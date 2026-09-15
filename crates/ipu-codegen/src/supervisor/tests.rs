use super::*;
use crate::inactive_exchange_program;

fn symbols() -> BTreeMap<String, u32> {
    [
        (WORKER_BARRIER_SYMBOL.into(), 0x50000),
        (COMPLETE_SYMBOL.into(), 0x50004),
        (HOST_RUN_SYMBOL.into(), 0x50008),
        (REPEAT_CALL_SYMBOL.into(), 0x5000c),
        (SAMPLE_CYCLE_SYMBOL.into(), 0x50010),
        (PATCH_REPEAT_TABLES_SYMBOL.into(), 0x50014),
        (PATCH_REPEAT_ARITHMETIC_SYMBOL.into(), 0x50018),
        ("gemm".into(), 0x51000),
    ]
    .into_iter()
    .collect()
}

#[test]
fn emits_resolved_exchange_and_compute_steps() {
    let program = TileProgram {
        tile: 7,
        steps: vec![
            TileStep::Exchange(ExchangeStep {
                active: false,
                incoming_base: 0,
                outgoing_base: None,
                preserve_base_registers: false,
                incoming_mux: None,
                incoming_format: 0,
                incoming_mux_pair: None,
                incoming_dcount: None,
                sync_in_program: false,
                program: PlacedExchangeRow {
                    address: 0x60000,
                    words: inactive_exchange_program(),
                },
                setup_patch: None,
                repeat_patches: Vec::new(),
                profile: StepProfile::default(),
            }),
            TileStep::Compute(ComputeStep {
                symbol: "gemm".into(),
                output_address: TileAddress::Absolute(0x70000),
                input_addresses: vec![
                    TileAddress::Absolute(0x71000),
                    TileAddress::Absolute(0x72000),
                ],
                arguments: vec![64],
                profile: StepProfile::default(),
            }),
        ],
    };
    let generated = emit(
        &program,
        &symbols(),
        &HostProgram::default(),
        &CodegenOptions {
            code_address: 0x52000,
            ..CodegenOptions::default()
        },
    )
    .unwrap();
    assert!(!generated.bytes.is_empty());
    assert_eq!(generated.exchange_rows.len(), 1);
    assert_eq!(generated.exchange_rows[0].address, 0x60000);
    let mut shared = program.clone();
    shared.steps.push(program.steps[0].clone());
    assert_eq!(
        validate::program(&shared, &HostProgram::default(), &CodegenOptions::default())
            .unwrap()
            .len(),
        1
    );
    let TileStep::Exchange(exchange) = shared.steps.last_mut().unwrap() else {
        unreachable!()
    };
    exchange.program.words.insert(0, 0);
    exchange.active = true;
    assert!(
        validate::program(&shared, &HostProgram::default(), &CodegenOptions::default()).is_err()
    );
}

#[test]
fn rejects_unresolved_or_malformed_inputs() {
    let program = TileProgram {
        tile: 0,
        steps: vec![TileStep::Exchange(ExchangeStep {
            active: false,
            incoming_base: 0,
            outgoing_base: None,
            preserve_base_registers: false,
            incoming_mux: None,
            incoming_format: 0,
            incoming_mux_pair: None,
            incoming_dcount: None,
            sync_in_program: false,
            program: PlacedExchangeRow {
                address: 3,
                words: Vec::new(),
            },
            setup_patch: None,
            repeat_patches: Vec::new(),
            profile: StepProfile::default(),
        })],
    };
    assert!(matches!(
        emit(
            &program,
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions::default()
        ),
        Err(CodegenError::Invalid(_))
    ));
    let mut program = TileProgram {
        tile: 0,
        steps: vec![TileStep::Compute(ComputeStep {
            symbol: "gemm".into(),
            output_address: TileAddress::RepeatPointer {
                index: 0,
                offset: 0,
            },
            input_addresses: vec![TileAddress::Absolute(0x70000)],
            arguments: vec![],
            profile: StepProfile::default(),
        })],
    };
    // Validation must reject an unavailable frame before emission can access it,
    // even when symbols are missing as well.
    assert!(matches!(
        emit(
            &program,
            &BTreeMap::new(),
            &HostProgram::default(),
            &CodegenOptions::default()
        ),
        Err(CodegenError::Invalid(_))
    ));
    program.steps.clear();
    assert!(matches!(
        emit(
            &program,
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions {
                invocations: 0,
                ..CodegenOptions::default()
            }
        ),
        Err(CodegenError::Invalid(_))
    ));
}

#[test]
fn arithmetic_patch_detection_is_exact_across_word_overflow() {
    let mut random = fastrand::Rng::with_seed(0x61726974686d);
    for _ in 0..256 {
        let initial = random.u32(..);
        let step = random.u32(..);
        let count = random.u32(3..=128);
        let mut words = (0..count)
            .map(|i| initial.wrapping_add(step.wrapping_mul(i)))
            .collect::<Vec<_>>();
        assert_eq!(arithmetic_progression(&words), Some((initial, step)));
        words[1] ^= 1;
        assert_eq!(arithmetic_progression(&words), None);
    }
    assert_eq!(arithmetic_progression(&[1, 2]), None);
    let legacy = serde_json::json!({"address": 0x60000, "words": [1, 4, 8]});
    assert!(matches!(
        serde_json::from_value::<ExchangePatchValues>(legacy).unwrap(),
        ExchangePatchValues::Table(_)
    ));
}

#[test]
fn randomized_repeat_patch_code_is_independent_of_iteration_count() {
    let mut random = fastrand::Rng::with_seed(0x7061_7463_685f_7265);
    let mut code_bytes = None;
    let mut arithmetic_code_bytes = None;
    for _ in 0..64 {
        let count = random.u32(2..=128);
        let values = (0..count).map(|_| random.u32(..)).collect::<Vec<_>>();
        let program = TileProgram {
            tile: 0,
            steps: vec![TileStep::Repeat(RepeatStep {
                count,
                iterated_pointers: vec![RepeatPointer {
                    initial_address: 0x70000,
                    stride_bytes: 64,
                }],
                body: vec![TileStep::Exchange(ExchangeStep {
                    active: true,
                    incoming_base: 0x70000,
                    outgoing_base: None,
                    preserve_base_registers: false,
                    incoming_mux: None,
                    incoming_format: 0,
                    incoming_mux_pair: None,
                    incoming_dcount: None,
                    sync_in_program: false,
                    program: PlacedExchangeRow {
                        address: 0x60000,
                        words: vec![0, ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION],
                    },
                    setup_patch: None,
                    repeat_patches: vec![ExchangePatch {
                        word_offset: 0,
                        values: ExchangePatchValues::Table(PlacedExchangeRow {
                            address: 0x61000,
                            words: values.clone(),
                        }),
                    }],
                    profile: StepProfile::default(),
                })],
                profile: StepProfile::default(),
            })],
        };
        let generated = emit(
            &program,
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions {
                code_address: 0x52000,
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert_eq!(generated.exchange_rows.len(), 2);
        assert_eq!(generated.exchange_rows[1].words, values);
        let emitted_words = generated
            .bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            emitted_words
                .iter()
                .filter(|word| **word == SYNC_SUPERVISOR_INSTRUCTION)
                .count(),
            1
        );
        assert_eq!(
            *code_bytes.get_or_insert(generated.bytes.len()),
            generated.bytes.len()
        );
        let mut program = program;
        let TileStep::Repeat(repeat) = &mut program.steps[0] else {
            unreachable!()
        };
        let TileStep::Exchange(exchange) = &mut repeat.body[0] else {
            unreachable!()
        };
        exchange.repeat_patches[0].values = ExchangePatchValues::Arithmetic {
            initial: 0xeffffff0,
            step: 0xffffc000,
        };
        let generated = emit(
            &program,
            &symbols(),
            &HostProgram::default(),
            &CodegenOptions {
                code_address: 0x52000,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            generated.exchange_rows.len(),
            1,
            "arithmetic patches allocate no value table"
        );
        assert_eq!(
            *arithmetic_code_bytes.get_or_insert(generated.bytes.len()),
            generated.bytes.len()
        );
        let TileStep::Repeat(repeat) = &mut program.steps[0] else {
            unreachable!()
        };
        let TileStep::Exchange(exchange) = &mut repeat.body[0] else {
            unreachable!()
        };
        // Mixed descriptors must survive code relocation without relocating
        // their destinations or mutable value tables.
        exchange.repeat_patches.push(ExchangePatch {
            word_offset: 1,
            values: ExchangePatchValues::Table(PlacedExchangeRow {
                address: 0x61000,
                words: values,
            }),
        });
        for base in [0x52000, 0x80000] {
            let mixed = emit(
                &program,
                &symbols(),
                &HostProgram::default(),
                &CodegenOptions {
                    code_address: base,
                    ..Default::default()
                },
            )
            .unwrap();
            let words = mixed
                .bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            let pool = words.len() - 5;
            assert_eq!(
                &words[pool..],
                &[0x60004, 0x61000, 0x60000, 0xeffffff0, 0xffffc000]
            );
            for offset in [0, 2] {
                let pointer = encode_setzi_m(2, base + (pool + offset) as u32 * 4).unwrap();
                assert!(words[..pool].contains(&pointer));
            }
        }
        let TileStep::Repeat(repeat) = &mut program.steps[0] else {
            unreachable!()
        };
        let TileStep::Exchange(exchange) = &mut repeat.body[0] else {
            unreachable!()
        };
        exchange.repeat_patches[1].word_offset = 0;
        assert!(
            emit(
                &program,
                &symbols(),
                &HostProgram::default(),
                &CodegenOptions::default()
            )
            .is_err()
        );
    }
}
