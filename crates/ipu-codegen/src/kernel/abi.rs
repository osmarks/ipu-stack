//! Worker register convention and resident kernel entry points. Ownership follows
//! the callable ABI even when a helper is linked from the runtime object.

pub const OUTPUT_REGISTER: u8 = 2;
pub const FIRST_INPUT_REGISTER: u8 = 3;
pub const LAST_VALUE_REGISTER: u8 = 9;
pub const RETURN_REGISTER: u8 = 10;
