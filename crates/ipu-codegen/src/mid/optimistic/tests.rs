use super::*;

fn format(precision: Precision, layout: Layout) -> TensorFormat {
    TensorFormat { precision, layout }
}

#[test]
fn conversion_search_exposes_early_cast_pack_and_keeps_late_baseline() {
    for rows in [1, 4] {
        let from = TensorType::new([rows, 128], Precision::F16, Layout::row_sharded(1));
        let mut layout = Layout::row_sharded(1);
        layout.order = ElementOrder::Amp(AmpOrder::Left);
        layout.tiling = TensorTiling::replicated(4);
        let to = TensorType::new(
            [rows, 128],
            Precision::F8F143 { scale_exponent: -4 },
            layout,
        );
        let paths = enumerate_conversions(&from, &to, 4, &SearchOptions::default()).unwrap();
        assert!(
            paths.iter().any(|p| p.assumptions.is_empty()),
            "late baseline absent: {paths:#?}"
        );
        assert!(
            paths.iter().any(|p| p
                .steps
                .first()
                .is_some_and(|s| s.to.format.precision == to.format.precision
                    && s.to.format.layout.tiling.replicas == 1)),
            "early path absent: {paths:#?}"
        );
        if rows == 1 {
            assert!(
                paths
                    .iter()
                    .any(|p| p.assumptions.contains(&Assumption::MissingEquivalenceRule))
            );
        } else {
            assert!(
                paths
                    .iter()
                    .any(|p| p.steps.iter().any(|s| s.kind == TransformKind::CastAndPack))
            );
        }
    }
}

#[test]
fn region_search_preserves_live_residual_and_fixed_boundaries() {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [2, 4, 128]).unwrap();
    let bias = graph.parameter("bias", [1, 1, 128]).unwrap();
    let residual = graph.host_input("residual", [2, 4, 128]).unwrap();
    let gamma = graph.parameter("gamma", [1, 1, 128]).unwrap();
    let beta = graph.parameter("beta", [1, 1, 128]).unwrap();
    let a = graph.add(x, bias).unwrap();
    let sum = graph.add(a, residual).unwrap();
    let normalized = graph.layer_norm(sum, gamma, beta).unwrap();
    graph.set_outputs([sum, normalized]).unwrap();
    let layout = Layout::row_sharded(4);
    let mut config = PipelineConfig::new(4);
    for input in graph.inputs() {
        config.inputs.insert(
            input.value,
            format(
                Precision::F16,
                if input.shape.0[0] == 1 {
                    Layout::row_major(TensorTiling::replicated(4))
                } else {
                    layout.clone()
                },
            ),
        );
    }
    let outputs = BTreeMap::from([
        (sum, format(Precision::F16, layout.clone())),
        (normalized, format(Precision::F16, layout)),
    ]);
    let report = plan_graph(&graph, &config, outputs.clone(), &SearchOptions::default()).unwrap();
    assert!(!report.candidates.is_empty());
    for candidate in &report.candidates {
        for &id in &candidate.outputs {
            let value = &candidate.values[id];
            assert_eq!(value.tensor.format, outputs[&value.origin]);
        }
    }
    assert!(
        report
            .candidates
            .iter()
            .any(|candidate| candidate.steps.iter().any(|step| matches!(
                step.kind,
                StepKind::FusedElementwise { .. }
            ) && step.outputs.len() == 2)),
        "{report:#?}"
    );
    let bad = RegionRequest {
        operations: 0..3,
        inputs: config.inputs.clone(),
        outputs: BTreeMap::from([(normalized, outputs[&normalized].clone())]),
    };
    assert!(plan_region(&graph, &bad, &config, &SearchOptions::default()).is_err());
}

#[test]
fn materializations_are_shared_and_search_limits_are_explicit() {
    use super::search::plan_region;
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [1, 4, 128]).unwrap();
    let a = graph.gelu(x).unwrap();
    let b = graph.gelu(x).unwrap();
    graph.set_outputs([a, b]).unwrap();
    let config =
        PipelineConfig::new(4).with_input(x, format(Precision::F16, Layout::row_sharded(1)));
    let output = format(Precision::F16, Layout::row_sharded(4));
    let request = RegionRequest {
        operations: 0..2,
        inputs: config.inputs.clone(),
        outputs: BTreeMap::from([(a, output.clone()), (b, output)]),
    };
    let options = SearchOptions {
        beam_width: 2,
        ..SearchOptions::default()
    };
    let report = plan_region(&graph, &request, &config, &options).unwrap();
    assert!(report.truncated);
    for candidate in &report.candidates {
        let converted = candidate
            .steps
            .iter()
            .filter(|step| {
                matches!(step.kind, StepKind::Transform(_))
                    && candidate.values[step.outputs[0]].origin == x
            })
            .map(|step| &candidate.values[step.outputs[0]].tensor)
            .collect::<Vec<_>>();
        for (i, tensor) in converted.iter().enumerate() {
            assert!(!converted[..i].contains(tensor));
        }
    }
    assert!(
        plan_region(
            &graph,
            &request,
            &config,
            &SearchOptions {
                max_operations: 1,
                ..options
            }
        )
        .is_err()
    );
}

#[test]
fn existing_word_unpack_is_not_reported_as_a_missing_kernel() {
    let mut layout = Layout::row_sharded(1);
    layout.order = ElementOrder::Amp(AmpOrder::Left);
    let source = TensorType::new([4, 128], Precision::F16, layout);
    let target = TensorType::new([4, 128], Precision::F16, Layout::row_sharded(4));
    let paths = enumerate_conversions(&source, &target, 4, &SearchOptions::default()).unwrap();
    assert!(paths.iter().any(|p| p.assumptions.is_empty()));
    let values = [source, target]
        .into_iter()
        .enumerate()
        .map(|(i, tensor_type)| {
            let id = MidValueId::from_index(i as u32);
            MidValue {
                id,
                tensor_type,
                origin: ValueId::from_index(i as u32),
                storage_group: id,
                tile_offset: 0,
            }
        })
        .collect();
    let program = MidProgram {
        tile_count: 4,
        values,
        inputs: vec![MidInput {
            name: "x".into(),
            kind: GraphInputKind::Host,
            value: MidValueId(0),
        }],
        outputs: vec![MidValueId(1)],
        operations: vec![MidOperation {
            source: None,
            inputs: vec![MidValueId(0)],
            results: vec![MidValueId(1)],
            kind: MidOperationKind::Primitive(Primitive::Copy {
                mapping: CoordinateMapping::default(),
                reuse_local: true,
            }),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        }],
        ..MidProgram::default()
    };
    let tiles = crate::low::expand::expand_tiles(&program, false).unwrap();
    let low = crate::low::lower_to_tiles(&tiles, false);
    for run in &low.kernel_runs {
        crate::validate_kernel_run(run).unwrap();
    }
}

#[test]
fn high_region_finds_fp8_opportunities_without_losing_its_reference() {
    let mut g = ComputeGraph::new();
    let x = g.host_input("x", [1, 8, 128]).unwrap();
    let y = g.gelu(x).unwrap();
    let w0 = g.parameter("w0", [128, 128]).unwrap();
    let w1 = g.parameter("w1", [128, 128]).unwrap();
    let q = g.gemm(y, w0).unwrap();
    let k = g.gemm(y, w1).unwrap();
    g.set_outputs([q, k]).unwrap();
    let mut config = PipelineConfig::new(64);
    for input in g.inputs() {
        config = config.with_automatic_input(input.value, Precision::F16);
    }
    config
        .operator_candidates
        .retain(|c| !matches!(c.operator(), MidOperator::Gemm { .. }));
    config
        .operator_candidates
        .push(OperatorCandidate::fp8_gemm(64, -4));
    let outputs = BTreeMap::from([
        (q, format(Precision::F16, Layout::row_sharded(64))),
        (k, format(Precision::F16, Layout::row_sharded(64))),
    ]);
    let options = SearchOptions {
        max_expansions: 256,
        beam_width: 4,
        ..SearchOptions::default()
    };
    let report = plan_graph(&g, &config, outputs.clone(), &options).unwrap();
    assert!(report.reference().is_some());
    assert!(report.opportunities().next().is_some());
    assert!(report.expanded <= options.max_expansions);
    assert!(report.candidates.iter().any(|c| c.steps.iter().any(|step|
        matches!(&step.kind,StepKind::Transform(t) if matches!(t.to.format.precision,Precision::F8F143 {..}))
        && step.outputs.iter().any(|&id| c.values[id].origin == y
            && c.steps.iter().filter(|consumer| consumer.inputs.contains(&id)).count() > 1))));
    assert!(
        report.candidates[0]
            .to_dot()
            .starts_with("digraph optimistic_mid")
    );
    config.tile_memory_budget_bytes = 1;
    assert!(matches!(
        plan_graph(&g, &config, outputs, &options),
        Err(SearchError::NoCandidates { .. })
    ));
}
