use super::*;

#[derive(Clone, Debug, Default)]
pub struct ConversionPath {
    pub steps: Vec<Transform>,
    pub cycles: CycleEstimate,
    pub assumptions: BTreeSet<Assumption>,
}

/// Enumerate short conversion sequences independently of the high-graph search.
/// Only endpoint precisions are used: no speculative lossy intermediate casts.
/// Ownership and order may be crossed to form producer-local packed candidates.
pub fn enumerate_conversions(
    from: &TensorType,
    to: &TensorType,
    tiles: u16,
    options: &SearchOptions,
) -> Result<Vec<ConversionPath>, SearchError> {
    if from.shape != to.shape
        || !valid_tensor(from, tiles)
        || !valid_tensor(to, tiles)
        || !(1..=4).contains(&options.max_conversion_steps)
        || options.conversion_frontier == 0
    {
        return Err(invalid("invalid conversion endpoints or search limits"));
    }
    if from == to {
        return Ok(vec![ConversionPath::default()]);
    }
    let mut nodes = vec![from.clone(), to.clone()];
    for owner in [&from.format.layout, &to.format.layout] {
        for order in [from.format.layout.order, to.format.layout.order] {
            for precision in [from.format.precision, to.format.precision] {
                let mut tensor = from.clone();
                tensor.format.layout = owner.clone();
                tensor.format.layout.order = order;
                tensor.format.precision = precision;
                if valid_tensor(&tensor, tiles) && !nodes.contains(&tensor) {
                    nodes.push(tensor);
                }
            }
        }
    }
    let edges = nodes
        .iter()
        .map(|a| {
            nodes
                .iter()
                .map(|b| {
                    let mut choices = Vec::new();
                    if let Some(step) = edge(a, b, false) {
                        choices.push(step);
                    }
                    if equivalent(a, b)
                        && let Some(step) = edge(a, b, true)
                    {
                        choices.push(step);
                    }
                    choices
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut paths = Vec::new();
    let mut pending = vec![(0usize, vec![0usize], ConversionPath::default())];
    while let Some((at, visited, path)) = pending.pop() {
        if at == 1 {
            paths.push(path);
            continue;
        }
        if path.steps.len() >= options.max_conversion_steps {
            continue;
        }
        for next in 1..nodes.len() {
            if visited.contains(&next) {
                continue;
            }
            // Cast at most once, always from the original to the final precision.
            if nodes[at].format.precision == to.format.precision
                && nodes[next].format.precision != to.format.precision
            {
                continue;
            }
            for step in &edges[at][next] {
                let mut extended = path.clone();
                extended.cycles = extended.cycles.plus(step.cycles);
                extended
                    .assumptions
                    .extend(step.assumptions.iter().cloned());
                extended.steps.push(step.clone());
                let mut seen = visited.clone();
                seen.push(next);
                pending.push((next, seen, extended));
            }
        }
    }
    // Surface an excluded ordering separately from the kernel it needs.
    for path in &mut paths {
        let early = path
            .steps
            .iter()
            .take_while(|s| s.kind != TransformKind::Redistribute)
            .position(|s| s.from.format.precision != s.to.format.precision);
        if let Some(cast_index) = early
            && from.format.precision == Precision::F16
            && matches!(to.format.precision, Precision::F8F143 { .. })
            && from.format.layout.tiling != to.format.layout.tiling
        {
            let ordinary_eligible = from.fp8_producer_layout(&to.format).is_some();
            if !ordinary_eligible {
                path.steps[cast_index]
                    .assumptions
                    .insert(Assumption::ConversionEligibility);
                path.assumptions.insert(Assumption::ConversionEligibility);
            }
        }
    }
    paths.sort_by_key(|p| (p.cycles.optimistic, p.cycles.conservative, p.steps.len()));
    let mut retained: Vec<ConversionPath> = Vec::new();
    for path in paths {
        if retained.iter().any(|p| {
            p.cycles.optimistic <= path.cycles.optimistic
                && p.cycles.conservative <= path.cycles.conservative
                && p.assumptions.is_subset(&path.assumptions)
        }) {
            continue;
        }
        retained.push(path);
    }
    // Keep an assumption-free route even if the optimistic frontier is full.
    let baseline = retained.iter().find(|p| p.assumptions.is_empty()).cloned();
    retained.truncate(options.conversion_frontier);
    if let Some(baseline) = baseline
        && !retained.iter().any(|p| p.assumptions.is_empty())
    {
        retained.push(baseline);
    }
    Ok(retained)
}

fn equivalent(a: &TensorType, b: &TensorType) -> bool {
    if a.format.precision != b.format.precision {
        return false;
    }
    let orders = [a.format.layout.order, b.format.layout.order];
    if !orders.contains(&ElementOrder::RowMajor)
        || !orders.contains(&ElementOrder::Amp(AmpOrder::Left))
    {
        return false;
    }
    let Some(axes) = a
        .format
        .layout
        .resolve(&a.shape)
        .ok()
        .and_then(|r| r.axes().map(<[_]>::to_vec))
    else {
        return false;
    };
    let width = if matches!(a.format.precision, Precision::F8F143 { .. }) {
        32
    } else {
        16
    };
    axes.len() >= 2
        && axes[..axes.len() - 1]
            .iter()
            .all(|axis| axis.maximum_extent() == 1)
        && axes
            .last()
            .is_some_and(|axis| axis.extents_are_multiple_of(width))
}

fn edge(a: &TensorType, b: &TensorType, alias: bool) -> Option<Transform> {
    let same_owner = a.format.layout.tiling == b.format.layout.tiling
        && a.format.layout.memory_class == b.format.layout.memory_class;
    let same_order = a.format.layout.order == b.format.layout.order;
    let same_precision = a.format.precision == b.format.precision;
    if !same_owner && (!same_precision || (!same_order && !word_unpack(a, b))) {
        return None;
    }
    let mut assumptions = BTreeSet::new();
    let input_bytes = crate::estimate::maximum_shard_bytes(a);
    let output_bytes = crate::estimate::maximum_shard_bytes(b);
    let elements = input_bytes / a.format.precision.bytes();
    let local = 330 + input_bytes.saturating_add(output_bytes).div_ceil(8);
    let (kind, cycles) = if !same_owner {
        let estimate = Ipu21CostModel.rearrangement_cost(
            &a.shape,
            a.format.precision,
            layout_conversion_strategy(&a.format.layout, &b.format.layout),
            &a.format.layout,
            &b.format.layout,
        );
        let minimum = 600 + input_bytes.max(output_bytes).div_ceil(4);
        let price = if estimate.cycles >= u64::MAX / 16 {
            assumptions.insert(Assumption::MissingCostModel);
            minimum.saturating_mul(4)
        } else {
            estimate.cycles.max(minimum)
        };
        (
            TransformKind::Redistribute,
            CycleEstimate {
                optimistic: if assumptions.is_empty() {
                    price
                } else {
                    minimum
                },
                conservative: price,
            },
        )
    } else if alias && equivalent(a, b) {
        assumptions.insert(Assumption::MissingEquivalenceRule);
        (TransformKind::Alias, CycleEstimate::default())
    } else {
        let kind = match (same_order, same_precision) {
            (true, false) => TransformKind::Cast,
            (false, true) => TransformKind::Pack,
            (false, false) => TransformKind::CastAndPack,
            (true, true) => return None,
        };
        let supported = local_supported(a, b, &kind);
        if !supported {
            assumptions.insert(Assumption::MissingKernel(format!(
                "{:?} {:?} -> {:?} {:?}",
                a.format.precision,
                a.format.layout.order,
                b.format.precision,
                b.format.layout.order
            )));
        }
        let work = local.max(330 + elements.div_ceil(8));
        let conservative = if supported && !same_precision {
            Ipu21CostModel.cast_format_cycles(a, &b.format).max(work)
        } else if supported {
            Ipu21CostModel
                .rearrangement_cost(
                    &a.shape,
                    a.format.precision,
                    layout_conversion_strategy(&a.format.layout, &b.format.layout),
                    &a.format.layout,
                    &b.format.layout,
                )
                .cycles
                .max(work)
        } else {
            work.saturating_mul(4)
        };
        (
            kind,
            CycleEstimate {
                optimistic: if supported { conservative } else { work },
                conservative,
            },
        )
    };
    Some(Transform {
        from: a.clone(),
        to: b.clone(),
        kind,
        cycles,
        assumptions,
    })
}

fn local_supported(a: &TensorType, b: &TensorType, kind: &TransformKind) -> bool {
    if *kind == TransformKind::Pack && word_unpack(a, b) {
        return true;
    }
    let kernel = match kind {
        TransformKind::CastAndPack
            if a.format.precision == Precision::F16
                && matches!(b.format.precision, Precision::F8F143 { .. })
                && a.format.layout.order == ElementOrder::RowMajor
                && b.format.layout.order == ElementOrder::Amp(AmpOrder::Left)
                && a.fp8_producer_layout(&b.format).as_ref() == Some(&b.format.layout) =>
        {
            TileKernelSpec::Cast {
                from: a.format.precision,
                to: b.format.precision,
            }
        }
        TransformKind::Cast => {
            let fp8 = matches!(a.format.precision, Precision::F8F143 { .. })
                || matches!(b.format.precision, Precision::F8F143 { .. });
            if fp8
                && a.format.layout.order != ElementOrder::RowMajor
                && !(a.format.precision == Precision::F16
                    && matches!(b.format.precision, Precision::F8F143 { .. })
                    && a.fp8_cast_layout().as_ref() == Some(&b.format.layout))
            {
                return false;
            }
            TileKernelSpec::Cast {
                from: a.format.precision,
                to: b.format.precision,
            }
        }
        TransformKind::Pack => TileKernelSpec::Rearrange {
            from: a.format.layout.clone(),
            to: b.format.layout.clone(),
        },
        _ => return false,
    };
    let requirements =
        crate::KernelRequirements::new(&kernel, [a.format.clone()], b.format.clone());
    crate::tile_kernel_abi(&kernel, &requirements)
        .is_ok_and(|abi| abi.availability == crate::KernelAvailability::Implemented)
}

// AMP-left consists of contiguous row fragments, unlike AMP output's lane
// permutation. The existing mapped Copy lowers these to word copies/exchange;
// absence of a specialized unpack kernel does not mean the route is missing.
fn word_unpack(a: &TensorType, b: &TensorType) -> bool {
    a.format.precision == b.format.precision
        && a.format.layout.order == ElementOrder::Amp(AmpOrder::Left)
        && b.format.layout.order == ElementOrder::RowMajor
        && [a, b].iter().all(|t| {
            t.format.layout.resolve(&t.shape).ok().is_some_and(|r| {
                r.axes().and_then(|axes| axes.last()).is_some_and(|axis| {
                    axis.extents_are_multiple_of((4 / t.format.precision.bytes()) as u32)
                })
            })
        })
}
