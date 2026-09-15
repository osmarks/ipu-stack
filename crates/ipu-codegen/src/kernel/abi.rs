//! Worker register convention and resident kernel entry points. Ownership follows
//! the callable ABI even when a helper is linked from the runtime object.

pub const OUTPUT_REGISTER: u8 = 2;
pub const FIRST_INPUT_REGISTER: u8 = 3;
pub const LAST_VALUE_REGISTER: u8 = 9;
pub const RETURN_REGISTER: u8 = 10;
pub const COPY_U16_SYMBOL: &str = "static_copy_u16";
pub const COPY_U32_SYMBOL: &str = "static_copy_u32";
pub const COPY_U64_SYMBOL: &str = "copy_u64";
pub const COPY_STRIDED_U32_SYMBOL: &str = "copy_strided_u32";
pub const COPY_STRIDED_U64_SYMBOL: &str = "copy_strided_u64";
pub const FILL_ZERO_U64_SYMBOL: &str = "fill_zero_u64";
