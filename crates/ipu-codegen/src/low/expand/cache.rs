//! Per-build caches for pure, relative low fragments. Phase assembly, alias
//! mutation and deferred materialization stay in the caller.
use super::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;

const MAX_ENTRIES: usize = 32768;

struct Memo<K, V> {
    entries: Mutex<(HashMap<K, Arc<V>>, u64, u64)>,
}
impl<K, V> Default for Memo<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new((HashMap::new(), 0, 0)),
        }
    }
}
impl<K: Eq + Hash, V> Memo<K, V> {
    fn get(&self, key: &K) -> Option<Arc<V>> {
        let mut state = self.entries.lock().unwrap();
        let found = state.0.get(key).cloned();
        if found.is_some() {
            state.1 += 1;
        } else {
            state.2 += 1;
        }
        found
    }
    fn insert(&self, key: K, value: V) {
        let mut state = self.entries.lock().unwrap();
        if state.0.len() < MAX_ENTRIES {
            state.0.entry(key).or_insert_with(|| Arc::new(value));
        }
    }
    fn stats(&self) -> (usize, u64, u64) {
        let state = self.entries.lock().unwrap();
        (state.0.len(), state.1, state.2)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Geometry {
    tensor: TensorType,
    extents: Vec<ShardExtent>,
}
impl Geometry {
    fn of(shard: &BlockValue) -> Self {
        Self {
            tensor: shard.tensor_type.clone(),
            extents: shard.extents.clone(),
        }
    }
}
#[derive(PartialEq, Eq, Hash)]
struct CopyKey {
    source: Geometry,
    destination: Geometry,
    source_view: Vec<ShardExtent>,
    destination_view: Vec<ShardExtent>,
    order: CopyOrder,
    same_buffer: bool,
}
#[derive(PartialEq, Eq, Hash)]
struct ProductKey {
    kernel: TileKernelSpec,
    axes: crate::ProductAxes,
    bindings: Vec<Geometry>,
    inputs: Vec<(usize, Vec<ShardExtent>)>,
    output: usize,
}

pub(crate) struct ExpansionCache {
    enabled: bool,
    copies: Memo<CopyKey, Vec<CopyOperation<()>>>,
    products: Memo<ProductKey, Vec<KernelRun>>,
}
impl Default for ExpansionCache {
    fn default() -> Self {
        Self {
            enabled: true,
            copies: Memo::default(),
            products: Memo::default(),
        }
    }
}
impl ExpansionCache {
    #[cfg(test)]
    pub(super) fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub(crate) fn stats(&self) -> [(usize, u64, u64); 2] {
        [self.copies.stats(), self.products.stats()]
    }
    pub(super) fn copy(
        &self,
        shards: &[BlockValue],
        source: &ShardView,
        destination: &ShardView,
        order: CopyOrder,
    ) -> ExpansionResult<Arc<Vec<CopyOperation<()>>>> {
        let generate = || -> ExpansionResult<Vec<CopyOperation<()>>> {
            let a = view_byte_traversal(&shards[source.shard.index() as usize], source, order)?;
            let b = view_byte_traversal(
                &shards[destination.shard.index() as usize],
                destination,
                order,
            )?;
            let copies = LocalCopy::from_traversals(source.shard, destination.shard, &a, &b)?
                .into_iter()
                .map(|c| CopyOperation {
                    source: (),
                    destination: (),
                    source_offset: c.source_offset,
                    destination_offset: c.destination_offset,
                    bytes: c.bytes,
                    pattern: c.pattern,
                })
                .collect();
            Ok(copies)
        };
        let left = &shards[source.shard.index() as usize];
        let right = &shards[destination.shard.index() as usize];
        let whole_copy = source.extents == left.extents
            && destination.extents == right.extents
            && (order == CopyOrder::Physical
                || (order == CopyOrder::Semantic
                    && left.tensor_type.format.layout.order == ElementOrder::RowMajor
                    && right.tensor_type.format.layout.order == ElementOrder::RowMajor));
        if !self.enabled || whole_copy {
            return Ok(Arc::new(generate()?));
        }
        let key = CopyKey {
            source: Geometry::of(&shards[source.shard.index() as usize]),
            destination: Geometry::of(&shards[destination.shard.index() as usize]),
            source_view: source.extents.clone(),
            destination_view: destination.extents.clone(),
            order,
            same_buffer: source.shard == destination.shard,
        };
        if let Some(copies) = self.copies.get(&key) {
            return Ok(copies);
        }
        let copies = generate()?;
        // Generation is outside the lock, so unrelated finalists do not serialize.
        let result = Arc::new(copies);
        let mut state = self.copies.entries.lock().unwrap();
        if state.0.len() < MAX_ENTRIES {
            state.0.entry(key).or_insert_with(|| Arc::clone(&result));
        }
        Ok(result)
    }
}

impl TileGraphBuilder {
    pub(super) fn cached_product_calls(
        &mut self,
        provenance: WorkProvenance,
        tile: u16,
        kernel: &TileKernelSpec,
        inputs: &[ShardView],
        output: BlockValueId,
        axes: crate::ProductAxes,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        if !self.cache.enabled {
            return self.product_calls(provenance, tile, kernel, inputs, output, axes, body);
        }
        // A one-call GEMM is cheaper to emit than to key and rebind. Cache only
        // fragments which actually expand blocking or outer matrix iteration.
        if let TileKernelSpec::Gemm {
            inner_block,
            output_columns,
            ..
        } = kernel
            && *inner_block != 0
            && *output_columns != 0
            && let Some(left) = inputs.first()
        {
            let output_shard = &self.shards[output.index() as usize];
            let k = left.extents[axes.left_inner.resolve(left.extents.len())?];
            let c = output_shard.extents[axes.output_column.resolve(output_shard.extents.len())?];
            let step = output_shard
                .tensor_type
                .format
                .layout
                .order
                .gemm_output_group()
                .map_or(*output_columns, |group| group.min(*output_columns));
            let outer = if matches!(
                output_shard.tensor_type.format.layout.order,
                ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
            ) {
                1
            } else {
                output_shard.extents[..output_shard.extents.len().saturating_sub(2)]
                    .iter()
                    .map(|e| u64::from(e.physical_end - e.start))
                    .product::<u64>()
            };
            if k.physical_end - k.start <= *inner_block
                && c.physical_end - c.start <= step
                && outer <= 1
            {
                return self.product_calls(provenance, tile, kernel, inputs, output, axes, body);
            }
        }
        let mut bindings = Vec::new();
        let mut slot = |id| {
            bindings.iter().position(|&b| b == id).unwrap_or_else(|| {
                bindings.push(id);
                bindings.len() - 1
            })
        };
        let bound_inputs = inputs
            .iter()
            .map(|v| (slot(v.shard), v.extents.clone()))
            .collect();
        let output_slot = slot(output);
        // A deferred view must be resolved by the enclosing builder, not replayed
        // from a fragment created under a different materialization decision.
        if bindings.iter().any(|&id| {
            self.full_view(id)
                != ShardView {
                    shard: id,
                    extents: self.shards[id.index() as usize].extents.clone(),
                }
        }) {
            return self.product_calls(provenance, tile, kernel, inputs, output, axes, body);
        }
        let key = ProductKey {
            kernel: kernel.clone(),
            axes,
            inputs: bound_inputs,
            bindings: bindings
                .iter()
                .map(|id| Geometry::of(&self.shards[id.index() as usize]))
                .collect(),
            output: output_slot,
        };
        if let Some(runs) = self.cache.products.get(&key) {
            for template in runs.iter() {
                let mut run = template.clone();
                for view in run
                    .inputs
                    .iter_mut()
                    .flat_map(|operand| &mut operand.views)
                    .chain(std::iter::once(&mut run.output))
                {
                    view.shard = bindings[view.shard.index() as usize];
                }
                if let Some(metadata) = self.kernel_metadata.iter().find(|m| {
                    m.provenance == provenance
                        && m.kernel == run.kernel
                        && m.requirements == run.requirements
                }) {
                    run.metadata = Arc::clone(metadata);
                } else {
                    run.metadata = Arc::new(KernelRunMetadata {
                        provenance,
                        kernel: run.kernel.clone(),
                        requirements: run.requirements.clone(),
                    });
                    self.kernel_metadata.push(Arc::clone(&run.metadata));
                }
                self.append_single_kernel(body, tile, run)?;
            }
            return Ok(());
        }
        let start = self.kernel_runs.len();
        self.product_calls(provenance, tile, kernel, inputs, output, axes, body)?;
        let mut runs = self.kernel_runs[start..].to_vec();
        for run in &mut runs {
            for view in run
                .inputs
                .iter_mut()
                .flat_map(|operand| &mut operand.views)
                .chain(std::iter::once(&mut run.output))
            {
                view.shard = BlockValueId(
                    bindings
                        .iter()
                        .position(|&id| id == view.shard)
                        .expect("product uses bound operands") as u32,
                );
            }
        }
        self.cache.products.insert(key, runs);
        Ok(())
    }
}
