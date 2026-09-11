//! Bounded independent product grids. Every combination is priced through its
//! complete mid implementation, including redistribution and partial sums.
use super::*;

pub(super) fn product_variants(
    base: OperatorPlan,
    inputs: &[TensorType],
    config: &PipelineConfig,
) -> Vec<OperatorPlan> {
    let OperatorDispatch::Attention { materialized, .. } = base.dispatch else {
        return vec![base];
    };
    let [qk_fp8, pv_fp8] = config.attention_fp8_scales;
    if (!materialized && (qk_fp8.is_some() || pv_fp8.is_some()))
        || [qk_fp8, pv_fp8]
            .into_iter()
            .flatten()
            .any(|s| !(-16..=15).contains(&s))
    {
        return vec![];
    }
    let policy = config.attention_products;
    if policy == AttentionProducts::SharedRows && (qk_fp8.is_some() || pv_fp8.is_some()) {
        return vec![];
    }
    if !materialized || policy == AttentionProducts::SharedRows {
        return if matches!(
            policy,
            AttentionProducts::Automatic | AttentionProducts::SharedRows
        ) {
            vec![base]
        } else {
            vec![]
        };
    }
    let heads = inputs[0].shape.0[0] as u16;
    let per_head = config.tile_count / heads;
    let query_rows = inputs[0].shape.0[1];
    let keys = inputs[1].shape.0[1];
    let values = inputs[2].shape.0[2];
    let grids = |columns: u32, inner: &[u16]| {
        let mut grids = Vec::new();
        for rows in 1u16..=16 {
            for &inner in inner {
                if u32::from(rows) > query_rows || u32::from(inner) > keys.div_ceil(16) {
                    continue;
                }
                let columns = (columns.div_ceil(16).min(u32::from(per_head / rows / inner))) as u16;
                if columns != 0 {
                    grids.push(Some(ProductGrid {
                        rows,
                        columns,
                        inner,
                    }));
                }
            }
        }
        grids
    };
    let mut qk = if policy == AttentionProducts::PvOnly {
        vec![]
    } else {
        grids(keys, &[1])
    };
    let mut pv = if policy == AttentionProducts::QkOnly {
        vec![]
    } else {
        grids(values, &[1, 2, 3, 4, 5, 6, 7, 8])
    };
    if matches!(
        policy,
        AttentionProducts::Automatic | AttentionProducts::PvOnly
    ) {
        qk.push(None);
    }
    if matches!(
        policy,
        AttentionProducts::Automatic | AttentionProducts::QkOnly
    ) {
        pv.push(None);
    }
    if qk_fp8.is_some() {
        qk.retain(Option::is_some);
    }
    if pv_fp8.is_some() {
        pv.retain(Option::is_some);
    }
    let mut result = Vec::new();
    for &query_key in &qk {
        for &probability_value in &pv {
            let mut plan = base.clone();
            if let OperatorDispatch::Attention {
                query_key: q,
                probability_value: p,
                ..
            } = &mut plan.dispatch
            {
                *q = query_key;
                *p = probability_value;
            }
            result.push(plan);
        }
    }
    result
}
