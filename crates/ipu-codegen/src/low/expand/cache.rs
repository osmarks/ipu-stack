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
struct CopyGeometry {
    precision: Precision,
    order: ElementOrder,
    allocation: Vec<ShardExtent>,
    view: Vec<ShardExtent>,
}
impl CopyGeometry {
    fn new(shard: &BlockValue, view: &ShardView) -> ExpansionResult<Self> {
        if shard.extents.len() != view.extents.len() {
            return Err(StorageError::InvalidView.into());
        }
        let mut allocation = shard.extents.clone();
        let mut view = view.extents.clone();
        for (a, v) in allocation.iter_mut().zip(&mut view) {
            let start = a.start;
            a.start = 0;
            a.logical_end -= start;
            a.physical_end -= start;
            v.start = v
                .start
                .checked_sub(start)
                .ok_or(StorageError::InvalidView)?;
            v.logical_end = v
                .logical_end
                .checked_sub(start)
                .ok_or(StorageError::InvalidView)?;
            v.physical_end = v
                .physical_end
                .checked_sub(start)
                .ok_or(StorageError::InvalidView)?;
        }
        Ok(Self {
            precision: shard.tensor_type.format.precision,
            order: shard.tensor_type.format.layout.order,
            allocation,
            view,
        })
    }
}
#[derive(PartialEq, Eq, Hash)]
struct CopyKey {
    source: CopyGeometry,
    destination: CopyGeometry,
    order: CopyOrder,
    same_buffer: bool,
}

struct ComputeFragment {
    primitive: crate::Primitive,
    boundary: Vec<(u16, Geometry)>,
    values: Vec<Vec<usize>>,
    runs: Vec<KernelRun>,
    aliases: Vec<(usize, usize)>,
}

pub(crate) struct ExpansionCache {
    enabled: bool,
    copies: Memo<CopyKey, Vec<CopyOperation<()>>>,
    computes: Memo<u64, Vec<Arc<ComputeFragment>>>,
}
impl Default for ExpansionCache {
    fn default() -> Self {
        Self {
            enabled: true,
            copies: Memo::default(),
            computes: Memo::default(),
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
        [self.copies.stats(), self.computes.stats()]
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
            source: CopyGeometry::new(left, source)?,
            destination: CopyGeometry::new(right, destination)?,
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
    pub(super) fn build_primitive(
        &mut self,
        operation: &MidOperation,
        primitive: &crate::Primitive,
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        use std::hash::{Hash, Hasher};
        let crate::Primitive::Compute {
            kernel,
            operands,
            product,
            reuse_input,
        } = primitive
        else {
            return self.build_primitive_uncached(operation, primitive, body);
        };
        if !self.cache.enabled {
            return self.build_primitive_uncached(operation, primitive, body);
        }
        let mut bindings = Vec::new();
        let mut slots = HashMap::new();
        let mut values = Vec::new();
        for &value in operation.inputs.iter().chain(&operation.results) {
            let mut ids = Vec::new();
            for &id in self.value_shards(value)? {
                // Leave deferred materialization to normal lowering. It can
                // redirect operands to buffers outside this operation boundary.
                if self.materialized_views.contains_key(&id) {
                    return self.build_primitive_uncached(operation, primitive, body);
                }
                let slot = *slots.entry(id).or_insert_with(|| {
                    bindings.push(id);
                    bindings.len() - 1
                });
                ids.push(slot);
            }
            values.push(ids);
        }
        // Hash borrowed geometry; allocate a stored boundary only on a miss.
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        kernel.hash(&mut hash);
        operands.hash(&mut hash);
        product.hash(&mut hash);
        reuse_input.hash(&mut hash);
        values.hash(&mut hash);
        for &id in &bindings {
            let shard = &self.shards[id.index() as usize];
            shard.tile.hash(&mut hash);
            shard.tensor_type.hash(&mut hash);
            shard.extents.hash(&mut hash);
        }
        let key = hash.finish();
        let bucket = self.cache.computes.get(&key);
        let hit = bucket.as_ref().and_then(|bucket| {
            bucket.iter().find(|entry| {
                entry.primitive == *primitive
                    && entry.values == values
                    && entry.boundary.len() == bindings.len()
                    && entry
                        .boundary
                        .iter()
                        .zip(&bindings)
                        .all(|((tile, geometry), id)| {
                            let shard = &self.shards[id.index() as usize];
                            *tile == shard.tile
                                && geometry.tensor == shard.tensor_type
                                && geometry.extents == shard.extents
                        })
            })
        });
        let provenance = operation_provenance(operation);
        if let Some(fragment) = hit {
            for &(output, input) in &fragment.aliases {
                self.shards[bindings[output].index() as usize].definition =
                    ShardDefinition::WritableAlias(bindings[input]);
            }
            let mut metadata = HashMap::new();
            for template in &fragment.runs {
                let mut run = template.clone();
                for view in run
                    .inputs
                    .iter_mut()
                    .flat_map(|operand| &mut operand.views)
                    .chain(std::iter::once(&mut run.output))
                {
                    view.shard = bindings[view.shard.index() as usize];
                }
                run.metadata = Arc::clone(
                    metadata
                        .entry(Arc::as_ptr(&template.metadata) as usize)
                        .or_insert_with(|| {
                            if let Some(m) = self.kernel_metadata.iter().find(|m| {
                                m.provenance == provenance
                                    && m.kernel == template.kernel
                                    && m.requirements == template.requirements
                            }) {
                                Arc::clone(m)
                            } else {
                                let m = Arc::new(KernelRunMetadata {
                                    provenance,
                                    kernel: template.kernel.clone(),
                                    requirements: template.requirements.clone(),
                                });
                                self.kernel_metadata.push(Arc::clone(&m));
                                m
                            }
                        }),
                );
                self.append_single_kernel(
                    body,
                    self.shards[run.output.shard.index() as usize].tile,
                    run,
                )?;
            }
            return Ok(());
        }
        let start = self.kernel_runs.len();
        self.build_primitive_uncached(operation, primitive, body)?;
        let mut aliases = Vec::new();
        if reuse_input.is_some() {
            for &output in self.value_shards(operation.results[0])? {
                let shard = &self.shards[output.index() as usize];
                if shard.extents.iter().any(|e| e.start == e.physical_end) {
                    continue;
                }
                let ShardDefinition::WritableAlias(input) = shard.definition else {
                    unreachable!("compute reuses the selected input")
                };
                aliases.push((slots[&output], slots[&input]));
            }
        }
        let mut runs = self.kernel_runs[start..].to_vec();
        for run in &mut runs {
            for view in run
                .inputs
                .iter_mut()
                .flat_map(|operand| &mut operand.views)
                .chain(std::iter::once(&mut run.output))
            {
                view.shard = BlockValueId(slots[&view.shard] as u32);
            }
        }
        let fragment = Arc::new(ComputeFragment {
            primitive: primitive.clone(),
            values,
            runs,
            aliases,
            boundary: bindings
                .iter()
                .map(|id| {
                    let s = &self.shards[id.index() as usize];
                    (s.tile, Geometry::of(s))
                })
                .collect(),
        });
        let mut state = self.cache.computes.entries.lock().unwrap();
        if state.0.len() < 256 {
            let bucket = state.0.entry(key).or_insert_with(|| Arc::new(Vec::new()));
            // Concurrent misses may have computed the same immutable fragment.
            if !bucket.iter().any(|entry| {
                entry.primitive == fragment.primitive
                    && entry.values == fragment.values
                    && entry.boundary == fragment.boundary
            }) {
                Arc::make_mut(bucket).push(fragment);
            }
        }
        Ok(())
    }
}
