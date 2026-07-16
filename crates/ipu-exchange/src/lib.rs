use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::debug;

pub const PLAN_WORDS: usize = 9;
pub const MAX_TRANSFER_WORDS: u32 = 4148;
pub const EXCHANGE_WINDOW_BASE: u32 = 0x50000;

pub type PlanRow = [u32; PLAN_WORDS];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub sender: PlanRow,
    pub receiver: PlanRow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MulticastPlan {
    pub sender: PlanRow,
    pub receivers: Vec<PlanRow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    logical_to_physical: Vec<u16>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExchangeError {
    #[error("logical tile {0} is out of range")]
    Tile(u16),
    #[error("exchange endpoints must be distinct")]
    DuplicateTile,
    #[error("exchange count {0} is outside 1..={MAX_TRANSFER_WORDS}")]
    Count(u32),
    #[error("invalid multicast receiver set")]
    ReceiverSet,
    #[error("exchange schedule is not encodable: {0}")]
    Schedule(&'static str),
    #[error("tile address 0x{0:x} is not encodable")]
    Address(u32),
}

impl Topology {
    pub fn new(logical_to_physical: Vec<u16>) -> Result<Self, ExchangeError> {
        let mut physical = HashSet::new();
        if logical_to_physical.is_empty()
            || logical_to_physical
                .iter()
                .any(|tile| !physical.insert(*tile))
        {
            return Err(ExchangeError::ReceiverSet);
        }
        Ok(Self {
            logical_to_physical,
        })
    }

    pub fn c600() -> Self {
        Self {
            logical_to_physical: (0..1472).map(c600_logical_to_physical).collect(),
        }
    }

    pub fn tile_count(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn physical(&self, logical: u16) -> Result<u16, ExchangeError> {
        self.logical_to_physical
            .get(usize::from(logical))
            .copied()
            .ok_or(ExchangeError::Tile(logical))
    }

    pub fn point_to_point(
        &self,
        sender_logical: u16,
        receiver_logical: u16,
        count: u32,
    ) -> Result<Plan, ExchangeError> {
        validate_count(count)?;
        if sender_logical == receiver_logical {
            return Err(ExchangeError::DuplicateTile);
        }
        let sender = u32::from(self.physical(sender_logical)?);
        let receiver = u32::from(self.physical(receiver_logical)?);
        let direction = direction(sender, receiver);
        let mux_time = time_to_mux(sender, receiver);
        let receiver_phase = 2 * (receiver >> 6);
        let sender_delay = 111 - mux_time;
        if !(-1..=0x7ffff).contains(&sender_delay) {
            return Err(ExchangeError::Schedule("sender delay"));
        }

        let mut sender_row = [0; PLAN_WORDS];
        sender_row[0] = 0x4180_0003;
        let mut cursor = 1;
        if sender_delay >= 0 {
            sender_row[cursor] = delay(sender_delay as u32);
            cursor += 1;
        }
        let first_packet = count.min(64);
        sender_row[cursor] = encode_send(first_packet - 1, direction, 0)?;
        cursor += 1;
        if count > 64 {
            sender_row[cursor] = send_off(count - 65, direction, 0);
            cursor += 1;
        }
        let trailing_delay = 4 - sender_delay - count as i32;
        if trailing_delay >= 0 {
            sender_row[cursor] = delay(trailing_delay as u32);
            cursor += 1;
        }
        sender_row[cursor] = 0x43a0_0000;

        let mut receiver_row = [0; PLAN_WORDS];
        receiver_row[0] = 1;
        receiver_row[1] = 0x4180_0003;
        receiver_row[2] = delay_xpic(112, 0, 0);
        if count <= 51 {
            receiver_row[3] = delay_xpic(count - 1, 0, 0x640);
            receiver_row[4] = delay_pic(51 - count + receiver_phase, 0, 0);
            receiver_row[5] = delay(count + 4);
            receiver_row[6] = 0x43a0_0000;
        } else if count == 52 {
            receiver_row[3] = delay_pic(50 + receiver_phase, 0, 0);
            receiver_row[4] = delay_xpic(0, 0, 0x640);
            receiver_row[5] = delay(56);
            receiver_row[6] = 0x43a0_0000;
        } else {
            receiver_row[3] = delay_pic(51 + receiver_phase, 0, 0);
            receiver_row[4] = delay_xpic(count - 53, 0, 0x640);
            receiver_row[5] = delay(56);
            receiver_row[6] = 0x43a0_0000;
        }
        debug!(
            sender_logical,
            receiver_logical, count, "assembled point-to-point exchange"
        );
        Ok(Plan {
            sender: sender_row,
            receiver: receiver_row,
        })
    }

    pub fn multicast(
        &self,
        sender_logical: u16,
        receiver_logical: &[u16],
        count: u32,
        schedule_offset: u32,
    ) -> Result<MulticastPlan, ExchangeError> {
        validate_count(count)?;
        let source_physical = u32::from(self.physical(sender_logical)?);
        let mut used = HashSet::from([sender_logical]);
        if receiver_logical.is_empty()
            || receiver_logical
                .iter()
                .any(|receiver| !used.insert(*receiver) || self.physical(*receiver).is_err())
        {
            return Err(ExchangeError::ReceiverSet);
        }
        let mux_times: Vec<_> = receiver_logical
            .iter()
            .map(|receiver| {
                self.physical(*receiver)
                    .map(|physical| time_to_mux(source_physical, u32::from(physical)))
            })
            .collect::<Result<_, _>>()?;
        let minimum_mux = *mux_times.iter().min().ok_or(ExchangeError::ReceiverSet)?;
        let natural_start = (-minimum_mux).max(0) as u32;
        let start_cycle = natural_start
            .checked_add(schedule_offset)
            .filter(|cycle| *cycle <= 4095)
            .ok_or(ExchangeError::Schedule("multicast start cycle"))?;
        let sender_delay = start_cycle as i32 - 1;

        let mut sender = [0; PLAN_WORDS];
        let mut cursor = 0;
        sender[cursor] = 0x4180_0003;
        cursor += 1;
        if sender_delay >= 0 {
            sender[cursor] = delay(sender_delay as u32);
            cursor += 1;
        }
        sender[cursor] = encode_send(count.min(64) - 1, 3, 0)?;
        cursor += 1;
        if count > 64 {
            sender[cursor] = send_off(count - 65, 3, 0);
            cursor += 1;
        }
        let trailing_delay = 4 - sender_delay - count as i32;
        if trailing_delay >= 0 {
            sender[cursor] = delay(trailing_delay as u32);
            cursor += 1;
        }
        sender[cursor] = 0x43a0_0000;

        let mut receivers = Vec::with_capacity(receiver_logical.len());
        for (logical, mux_time) in receiver_logical.iter().zip(mux_times) {
            let physical = u32::from(self.physical(*logical)?);
            let receive_cycle = start_cycle as i32 + mux_time;
            if !(0..=4095).contains(&receive_cycle) {
                return Err(ExchangeError::Schedule("multicast receive cycle"));
            }
            let receiver_phase = 2 * (physical >> 6);
            let mut row = [0; PLAN_WORDS];
            row[0] = 0x4180_0003;
            row[1] = delay_xpic(receive_cycle as u32, 0, source_physical);
            if count <= 51 {
                row[2] = delay_xpic(count - 1, 0, 0x640);
                row[3] = delay_pic(51 - count + receiver_phase, 0, 0) | 0x0001_4000;
                row[4] = delay(count + 4);
                row[5] = 0x43a0_0000;
            } else if count == 52 {
                row[2] = delay_pic(50 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay_xpic(0, 0, 0x640);
                row[4] = delay(56);
                row[5] = 0x43a0_0000;
            } else {
                row[2] = delay_pic(51 + receiver_phase, 0, 0) | 0x0001_4000;
                row[3] = delay_xpic(count - 53, 0, 0x640);
                row[4] = delay(56);
                row[5] = 0x43a0_0000;
            }
            receivers.push(row);
        }
        debug!(
            sender_logical,
            receivers = receivers.len(),
            count,
            schedule_offset,
            "assembled multicast exchange"
        );
        Ok(MulticastPlan { sender, receivers })
    }
}

pub fn c600_logical_to_physical(logical: u16) -> u16 {
    let pair = logical / 2;
    let lane = logical & 1;
    let block = pair / 23;
    let mut row = pair % 23;
    if block & 1 != 0 {
        row = 22 - row;
    }
    let column = (block / 2) * 4 + (block & 1);
    row * 64 + column + lane * 2
}

pub fn patch_sender_address(row: &mut PlanRow, byte_address: u32) -> Result<(), ExchangeError> {
    if byte_address & 3 != 0 || byte_address >> 2 > 0x1f_ffff {
        return Err(ExchangeError::Address(byte_address));
    }
    let word_address = byte_address >> 2;
    for instruction in row {
        if *instruction & 0xf800_0000 == 0x7800_0000 {
            *instruction = (*instruction & !0x001f_fff8) | ((word_address << 3) & 0x001f_fff8);
            return Ok(());
        }
    }
    Err(ExchangeError::Address(byte_address))
}

pub fn patch_multicast_receiver_address(
    row: &mut PlanRow,
    byte_address: u32,
) -> Result<(), ExchangeError> {
    if byte_address < EXCHANGE_WINDOW_BASE || byte_address & 3 != 0 {
        return Err(ExchangeError::Address(byte_address));
    }
    let window_word = (byte_address - EXCHANGE_WINDOW_BASE) >> 2;
    if window_word > 0x1fff {
        return Err(ExchangeError::Address(byte_address));
    }
    let instruction = row
        .iter_mut()
        .find(|word| **word & 0xfc01_4000 == 0x6001_4000)
        .ok_or(ExchangeError::Address(byte_address))?;
    *instruction = (*instruction & !0x1fff) | window_word;
    Ok(())
}

fn validate_count(count: u32) -> Result<(), ExchangeError> {
    if (1..=MAX_TRANSFER_WORDS).contains(&count) {
        Ok(())
    } else {
        Err(ExchangeError::Count(count))
    }
}

fn route_displacement(source: u32, destination: u32) -> i32 {
    let source_raw = ((source >> 2) & 15) as i32;
    let destination_raw = ((destination >> 2) & 15) as i32;
    let source_column = if source_raw > 7 {
        source_raw ^ 15
    } else {
        source_raw
    };
    let destination_column = if destination_raw > 7 {
        destination_raw ^ 15
    } else {
        destination_raw
    };
    let source_mux = source_column + ((source_raw >> 3) ^ (source & 1) as i32);
    let base = (destination_column - source_mux) * 6;
    let destination_lane = destination & 3;
    let destination_half = destination_raw >> 3;
    if destination_lane > 1 {
        base + if destination_half == (destination & 1) as i32 {
            2
        } else {
            4
        }
    } else {
        base + if destination_half == destination_lane as i32 {
            1
        } else {
            5
        }
    }
}

fn direction(source: u32, destination: u32) -> u32 {
    if route_displacement(source, destination) < 1 {
        2
    } else {
        1
    }
}

fn time_to_mux(source: u32, destination: u32) -> i32 {
    let source_raw = ((source >> 2) & 15) as i32;
    let destination_raw = ((destination >> 2) & 15) as i32;
    let source_low = ((source >> 2) & 7) as i32;
    let displacement = route_displacement(source, destination);
    let source_edge = if source_raw > 7 {
        (source_raw * 4) ^ 60
    } else {
        source_raw * 4
    };
    let destination_edge = if destination_raw > 7 {
        (destination_raw * 4) ^ 60
    } else {
        destination_raw * 4
    };
    let local = ((source >> 2) & 8) as i32 | ((source >> 3) & 3) as i32;
    let crossing = local - destination_raw + ((source_low >> 1) ^ 3);
    let same_region = (source ^ destination) & 0x20 == 0;
    let turn = if same_region {
        source_low + 1
    } else {
        16 - source_low
    };
    let group_delta = (((source >> 6) & 31) as i32 - ((destination >> 6) & 31) as i32) * 2;
    crossing + source_edge + turn - destination_edge + group_delta + displacement.abs() - 34
}

fn delay(cycles: u32) -> u32 {
    0x40a0_0000 | (cycles & 0x7ffff)
}

fn delay_pic(a: u32, b: u32, c: u32) -> u32 {
    0x6000_0000 | ((a << 19) & 0x03f8_0000) | ((b << 18) & 0x0004_0000) | (c & 0x3ffff)
}

fn delay_xpic(a: u32, b: u32, c: u32) -> u32 {
    0x6400_0000 | ((a << 14) & 0x03ff_c000) | ((b << 13) & 0x0000_2000) | (c & 0x1fff)
}

pub fn encode_send(
    count_minus_one: u32,
    direction: u32,
    base_word: u32,
) -> Result<u32, ExchangeError> {
    if count_minus_one > 63 || direction > 7 || base_word > 0x3_ffff {
        return Err(ExchangeError::Schedule("send instruction operand"));
    }
    Ok(0x7800_0000
        | ((count_minus_one << 21) & 0x07e0_0000)
        | ((base_word << 3) & 0x001f_fff8)
        | direction)
}

fn send_off(count_minus_one: u32, direction: u32, base_word: u32) -> u32 {
    0x7000_0000
        | ((count_minus_one << 21) & 0x07e0_0000)
        | (((count_minus_one >> 6) << 14) & 0x000f_c000)
        | ((base_word << 3) & 0x0000_3ff8)
        | (direction & 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c600_mapping_is_a_permutation() {
        let topology = Topology::c600();
        let physical: HashSet<_> = (0..topology.tile_count() as u16)
            .map(|logical| topology.physical(logical).unwrap())
            .collect();
        assert_eq!(physical.len(), 1472);
        assert_eq!(topology.physical(46).unwrap(), 1409);
        assert!(physical.iter().all(|tile| *tile < 1472));
    }

    #[test]
    fn point_to_point_matches_cpp_oracle_vectors() {
        let topology = Topology::c600();
        let cases = [
            (
                0,
                1286,
                3,
                [
                    0x41800003, 0x40a00086, 0x78400001, 0x43a00000, 0, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x64008640, 0x61800000, 0x40a00007, 0x43a00000, 0, 0,
                ],
            ),
            (
                736,
                100,
                52,
                [
                    0x41800003, 0x40a0003e, 0x7e600002, 0x43a00000, 0, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x61d00000, 0x64000640, 0x40a00038, 0x43a00000, 0, 0,
                ],
            ),
            (
                1286,
                1471,
                65,
                [
                    0x41800003, 0x40a0007d, 0x7fe00002, 0x70000002, 0x43a00000, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x61980000, 0x64030640, 0x40a00038, 0x43a00000, 0, 0,
                ],
            ),
            (
                32,
                0,
                4148,
                [
                    0x41800003, 0x40a0006c, 0x7fe00001, 0x766fc001, 0x43a00000, 0, 0, 0, 0,
                ],
                [
                    1, 0x41800003, 0x641c0000, 0x61980000, 0x67ffc640, 0x40a00038, 0x43a00000, 0, 0,
                ],
            ),
        ];
        for (sender, receiver, count, expected_sender, expected_receiver) in cases {
            let plan = topology.point_to_point(sender, receiver, count).unwrap();
            assert_eq!(plan.sender, expected_sender);
            assert_eq!(plan.receiver, expected_receiver);
        }
    }

    #[test]
    fn multicast_matches_cpp_oracle_vectors() {
        let topology = Topology::c600();
        let plan = topology.multicast(0, &[274, 1286], 3, 0).unwrap();
        assert_eq!(plan.sender[0], 0x41800003);
        assert_eq!(plan.sender[1], 0x40a00016);
        assert_eq!(plan.sender[2], 0x78400003);
        assert_eq!(plan.receivers[0][1], 0x64000000);
        assert_eq!(plan.receivers[0][2], 0x64008640);
        assert_eq!(plan.receivers[0][3], 0x61814000);

        let boundary = topology.multicast(736, &[100, 900], 52, 0).unwrap();
        assert_eq!(boundary.receivers[0][2], 0x61d14000);
        assert_eq!(boundary.receivers[0][3], 0x64000640);
    }

    #[test]
    fn validates_limits_and_patches_addresses() {
        let topology = Topology::c600();
        assert_eq!(
            topology.point_to_point(0, 1, 0),
            Err(ExchangeError::Count(0))
        );
        assert_eq!(
            topology.multicast(0, &[1, 1], 1, 0),
            Err(ExchangeError::ReceiverSet)
        );
        let mut plan = topology.multicast(0, &[274], 65, 0).unwrap();
        patch_sender_address(&mut plan.sender, 0x52040).unwrap();
        patch_multicast_receiver_address(&mut plan.receivers[0], 0x53080).unwrap();
        assert_eq!(
            plan.sender[2] & 0x001f_fff8,
            ((0x52040 >> 2) << 3) & 0x001f_fff8
        );
        assert_eq!(plan.receivers[0][2] & 0x1fff, (0x53080 - 0x50000) >> 2);
    }

    #[test]
    fn encodes_supervisor_send_fields() {
        assert_eq!(encode_send(1, 3, 82_041).unwrap(), 0x782a_03cb);
        assert_eq!(encode_send(1, 3, 82_043).unwrap(), 0x782a_03db);
        assert!(encode_send(64, 3, 0).is_err());
        assert!(encode_send(1, 8, 0).is_err());
        assert!(encode_send(1, 3, 0x4_0000).is_err());
    }
}
