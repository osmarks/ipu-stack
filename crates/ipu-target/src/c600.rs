//! C600 compute-tile inventory and the SDK's logical ordering. This inventory
//! does not constrain custom Topology mappings to a whole-device permutation.
pub const COMPUTE_TILES: u16 = 1472;

pub fn logical_to_physical(logical: u16) -> u16 {
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
