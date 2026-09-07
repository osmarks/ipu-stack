//! Graphcore F143 encoding (bias 8, unsigned zero, 0x80 NaN).

pub fn f143_scale(values: impl IntoIterator<Item = f32>) -> i8 {
    let maximum = values
        .into_iter()
        .filter(|value| value.is_finite())
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    if maximum == 0.0 {
        return 0;
    }
    (maximum / 240.0).log2().ceil().clamp(-32.0, 31.0) as i8
}

pub fn f143_from_f32(value: f32, scale: i8) -> u8 {
    let scale_multiplier = 2.0f32.powi(-i32::from(scale));
    f143_from_scaled_f32(value * scale_multiplier)
}

fn f143_from_scaled_f32(value: f32) -> u8 {
    let sign = u8::from(value.is_sign_negative()) << 7;
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return 0x80;
    }
    if magnitude == 0.0 {
        return 0;
    }
    if !magnitude.is_finite() || magnitude >= 240.0 {
        return sign | 0x7f;
    }
    if magnitude < 1.0 / 128.0 {
        let mantissa = (magnitude * 1024.0).round_ties_even() as u8;
        let mantissa = mantissa.min(8);
        return if mantissa == 0 { 0 } else { sign | mantissa };
    }

    let exponent = i32::try_from((magnitude.to_bits() >> 23) & 0xff).unwrap() - 127;
    let mut encoded_exponent = exponent + 8;
    let unit = f32::from_bits(((exponent + 127) as u32) << 23);
    let mut mantissa = ((magnitude / unit - 1.0) * 8.0).round_ties_even() as i32;
    if mantissa == 8 {
        mantissa = 0;
        encoded_exponent += 1;
    }
    if encoded_exponent > 15 {
        return sign | 0x7f;
    }
    sign | ((encoded_exponent as u8) << 3) | mantissa as u8
}

pub fn f143_to_f32(bits: u8, scale: i8) -> f32 {
    if bits == 0x80 {
        return f32::NAN;
    }
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0xf;
    let mantissa = bits & 7;
    let value = if exponent == 0 {
        f32::from(mantissa) * 2.0f32.powi(-10)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 8)
    };
    sign * value * 2.0f32.powi(i32::from(scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_finite_encoding_round_trips_at_hardware_scales() {
        for scale in -32..=31 {
            for bits in 0..=255 {
                if bits != 0x80 {
                    assert_eq!(f143_from_f32(f143_to_f32(bits, scale), scale), bits);
                }
            }
        }
        assert!(f143_to_f32(0x80, 0).is_nan());
        assert_eq!(f143_from_f32(f32::NAN, 0), 0x80);
    }

    #[test]
    fn underflow_does_not_create_negative_zero_nan() {
        for value in [-0.0, -0.0001, -1.0 / 2048.0] {
            assert_eq!(f143_from_f32(value, 0), 0);
        }
        assert_eq!(f143_from_f32(-3.0 / 2048.0, 0), 0x82);
        assert_eq!(f143_from_f32(1000.0, 0), 0x7f);
        assert_eq!(f143_from_f32(-1000.0, 0), 0xff);
    }

    #[test]
    fn scale_covers_the_finite_range() {
        assert_eq!(f143_scale([0.0]), 0);
        assert_eq!(f143_scale([240.0, -120.0]), 0);
        assert_eq!(f143_scale([241.0]), 1);
        assert_eq!(f143_scale([-15.0, f32::NAN]), -4);
    }
}
