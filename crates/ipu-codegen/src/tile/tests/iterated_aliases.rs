use super::*;
use crate::mid::MidOperationKind;

use crate::*;

#[test]
fn pointer_resolution_preserves_signed_offsets_through_alias_chains() {
    let argument = BlockValueId::from_index(0);
    let tensor_type = TensorType::new([16], Precision::F16, Layout::logical_linear(1, 4));
    let mut shards = vec![BlockValue {
        id: argument,
        tile: 0,
        extents: tensor_type
            .format
            .layout
            .shard_extents(&tensor_type.shape)
            .unwrap()[0]
            .1
            .clone(),
        tensor_type,
        definition: ShardDefinition::Staging,
    }];
    let mut addresses = BTreeMap::from([(argument, 0x60000u32)]);
    let overrides = BTreeMap::from([(
        argument,
        TileAddress::RepeatPointer {
            index: 0,
            offset: 32,
        },
    )]);
    let mut source = argument;
    let mut displacement = 0;
    for delta in [None, None, Some(-32768), Some(65536)] {
        let mut alias = shards[source.index() as usize].clone();
        alias.id = BlockValueId::from_index(shards.len() as u32);
        alias.definition = match delta {
            Some(offset) => ShardDefinition::ShiftedAlias { source, offset },
            None if source == argument => ShardDefinition::Alias(source),
            None => ShardDefinition::WritableAlias(source),
        };
        displacement += delta.unwrap_or(0);
        addresses.insert(
            alias.id,
            addresses[&argument]
                .checked_add_signed(displacement)
                .unwrap(),
        );
        source = alias.id;
        shards.push(alias);
        assert_eq!(
            crate::low::storage::resolve_address(&shards, &addresses, &overrides, source, 0,)
                .unwrap(),
            TileAddress::RepeatPointer {
                index: 0,
                offset: displacement + 32
            }
        );
        assert_eq!(
            crate::low::storage::resolve_address(&shards, &addresses, &BTreeMap::new(), source, 0,)
                .unwrap(),
            TileAddress::Absolute(addresses[&source])
        );
    }
    let output = BlockValueId::from_index(shards.len() as u32);
    let mut block = shards[source.index() as usize].clone();
    block.id = output;
    block.definition = ShardDefinition::Staging;
    shards.push(block);
    addresses.insert(output, 0x80000);
    let view = |shard: BlockValueId| ShardView {
        shard,
        extents: shards[shard.index() as usize].extents.clone(),
    };
    let kernel = MidOperationKind::Gelu;
    let run = KernelRun::bind(
        WorkProvenance {
            operation: None,
            value: None,
            reason: WorkReason::OperatorKernel,
        },
        kernel,
        vec![view(source)],
        vec![view(output)],
        &shards,
        &mut vec![],
    )
    .unwrap();
    let call = materialize_kernel_run(&run, &shards, &addresses, &overrides).unwrap();
    assert_eq!(
        call.input_addresses,
        [TileAddress::RepeatPointer {
            index: 0,
            offset: displacement + 32
        }]
    );
}
