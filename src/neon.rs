//! 256-bit integer vectors as pairs of 128-bit NEON registers, for the aarch64 kernels.
//!
//! The pairs give the NEON kernels the lane counts of their AVX2 versions (32 x 8, 16 x 16
//! and 8 x 32 bits), so striped layouts, range checks and fallbacks are the same on both
//! architectures, and the two halves are independent work for the wide NEON back ends.
//! NEON is part of the aarch64 baseline, so nothing here needs runtime detection.

use std::arch::aarch64::*;

/// Evaluates an expression of NEON intrinsics that take no pointers.
macro_rules! neon {
    ($e:expr) => {
        // SAFETY: NEON is part of the aarch64 baseline, and the intrinsics in `$e` only
        // compute on registers (callers use their own `unsafe` blocks for loads and stores).
        unsafe { $e }
    };
}
pub(crate) use neon;

/// 32 lanes of i8: lanes 0-15 in the first register, 16-31 in the second.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct I8x32(int8x16_t, int8x16_t);

impl I8x32 {
    #[inline]
    pub fn load(bytes: &[u8; 32]) -> Self {
        // SAFETY: a [u8; 32] is 32 readable bytes; NEON loads have no alignment requirement.
        unsafe { Self(vld1q_s8(bytes.as_ptr().cast()), vld1q_s8(bytes.as_ptr().add(16).cast())) }
    }

    #[inline]
    pub fn splat(x: i8) -> Self {
        neon!(Self(vdupq_n_s8(x), vdupq_n_s8(x)))
    }

    #[inline]
    pub fn to_array(self) -> [i8; 32] {
        // SAFETY: a #[repr(C)] pair of 16-byte registers and [i8; 32] are both 32 bytes of
        // plain integers.
        unsafe { std::mem::transmute::<Self, [i8; 32]>(self) }
    }

    /// The two 16-lane halves.
    #[inline]
    pub fn halves(self) -> [int8x16_t; 2] {
        [self.0, self.1]
    }

    /// `table[self & 15]` per lane, like `pshufb` with a 16-entry table in both halves.
    /// Lanes must hold values below 16.
    #[inline]
    pub fn lookup(self, table: int8x16_t) -> Self {
        neon!(Self(vqtbl1q_s8(table, vreinterpretq_u8_s8(self.0)), vqtbl1q_s8(table, vreinterpretq_u8_s8(self.1))))
    }

    #[inline]
    pub fn and(self, o: Self) -> Self {
        neon!(Self(vandq_s8(self.0, o.0), vandq_s8(self.1, o.1)))
    }

    /// All ones where the lanes are equal.
    #[inline]
    pub fn eq(self, o: Self) -> Self {
        neon!(Self(vreinterpretq_s8_u8(vceqq_s8(self.0, o.0)), vreinterpretq_s8_u8(vceqq_s8(self.1, o.1))))
    }

    #[inline]
    pub fn max(self, o: Self) -> Self {
        neon!(Self(vmaxq_s8(self.0, o.0), vmaxq_s8(self.1, o.1)))
    }

    /// Wrapping addition.
    #[inline]
    pub fn add(self, o: Self) -> Self {
        neon!(Self(vaddq_s8(self.0, o.0), vaddq_s8(self.1, o.1)))
    }

    /// Wrapping subtraction.
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        neon!(Self(vsubq_s8(self.0, o.0), vsubq_s8(self.1, o.1)))
    }
}

/// Lane operations of the 16- and 32-bit striped kernels, which are generic over them.
/// Arithmetic saturates in 16-bit lanes and wraps in 32-bit lanes, as in the AVX2 kernels.
pub trait Lanes<const L: usize>: Copy + std::fmt::Debug {
    type Elem: Copy + Into<i32>;
    /// The kernels' minus infinity.
    const NEG: Self::Elem;
    const MAX: Self::Elem;
    /// `x as Elem`.
    fn cast(x: i32) -> Self::Elem;
    fn splat(x: Self::Elem) -> Self;
    fn from_array(lanes: [Self::Elem; L]) -> Self;
    fn to_array(self) -> [Self::Elem; L];
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    /// All ones where the lanes are equal.
    fn eq(self, o: Self) -> Self;
    /// All ones where `self` is greater.
    fn gt(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    fn shl4(self) -> Self;
    /// Whether any lane is nonzero.
    fn any(self) -> bool;
    /// Moves every lane up by one (lane `l` to `l + 1`) and puts `x` in lane 0.
    fn shift_in(self, x: Self::Elem) -> Self;
    /// Replaces lane 0 with `x`.
    fn with_first(self, x: Self::Elem) -> Self;
    /// Sign-extends the first `L` bytes of `bytes`.
    fn widen(bytes: &[i8]) -> Self;
    /// Narrows lanes holding small non-negative values to bytes, written to `out[..L]`.
    fn narrow(self, out: &mut [u8]);
}

macro_rules! lanes {
    ($name:ident, $elem:ty, $half:ty, $lanes:expr, $neg:expr, $dup:ident, $add:ident, $sub:ident, $max:ident,
     $min:ident, $ceq:ident, $cgt:ident, $and:ident, $or:ident, $shl:ident, $ext:ident, $set_lane:ident,
     $to_signed:ident, $to_unsigned:ident, $maxv:ident, $widen:ident, $narrow:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug)]
        pub struct $name($half, $half);

        impl Lanes<$lanes> for $name {
            type Elem = $elem;
            const NEG: $elem = $neg;
            const MAX: $elem = <$elem>::MAX;

            #[inline]
            fn cast(x: i32) -> $elem {
                x as $elem
            }

            #[inline]
            fn splat(x: $elem) -> Self {
                neon!(Self($dup(x), $dup(x)))
            }

            #[inline]
            fn from_array(lanes: [$elem; $lanes]) -> Self {
                // SAFETY: a #[repr(C)] pair of 16-byte registers and the array are both 32
                // bytes of plain integers.
                unsafe { std::mem::transmute::<[$elem; $lanes], Self>(lanes) }
            }

            #[inline]
            fn to_array(self) -> [$elem; $lanes] {
                // SAFETY: as above, in the other direction.
                unsafe { std::mem::transmute::<Self, [$elem; $lanes]>(self) }
            }

            #[inline]
            fn add(self, o: Self) -> Self {
                neon!(Self($add(self.0, o.0), $add(self.1, o.1)))
            }

            #[inline]
            fn sub(self, o: Self) -> Self {
                neon!(Self($sub(self.0, o.0), $sub(self.1, o.1)))
            }

            #[inline]
            fn max(self, o: Self) -> Self {
                neon!(Self($max(self.0, o.0), $max(self.1, o.1)))
            }

            #[inline]
            fn min(self, o: Self) -> Self {
                neon!(Self($min(self.0, o.0), $min(self.1, o.1)))
            }

            #[inline]
            fn eq(self, o: Self) -> Self {
                neon!(Self($to_signed($ceq(self.0, o.0)), $to_signed($ceq(self.1, o.1))))
            }

            #[inline]
            fn gt(self, o: Self) -> Self {
                neon!(Self($to_signed($cgt(self.0, o.0)), $to_signed($cgt(self.1, o.1))))
            }

            #[inline]
            fn and(self, o: Self) -> Self {
                neon!(Self($and(self.0, o.0), $and(self.1, o.1)))
            }

            #[inline]
            fn or(self, o: Self) -> Self {
                neon!(Self($or(self.0, o.0), $or(self.1, o.1)))
            }

            #[inline]
            fn shl4(self) -> Self {
                neon!(Self($shl::<4>(self.0), $shl::<4>(self.1)))
            }

            #[inline]
            fn any(self) -> bool {
                neon!($maxv($to_unsigned($or(self.0, self.1))) != 0)
            }

            #[inline]
            fn shift_in(self, x: $elem) -> Self {
                const TOP: i32 = $lanes / 2 - 1;
                // vext(a, b, TOP) is a's last lane followed by b's first TOP lanes.
                neon!(Self($ext::<TOP>($dup(x), self.0), $ext::<TOP>(self.0, self.1)))
            }

            #[inline]
            fn with_first(self, x: $elem) -> Self {
                neon!(Self($set_lane::<0>(x, self.0), self.1))
            }

            #[inline]
            fn widen(bytes: &[i8]) -> Self {
                let (lo, hi) = $widen(bytes);
                Self(lo, hi)
            }

            #[inline]
            fn narrow(self, out: &mut [u8]) {
                $narrow(self.0, self.1, out);
            }
        }
    };
}

lanes!(
    I16x16,
    i16,
    int16x8_t,
    16,
    i16::MIN,
    vdupq_n_s16,
    vqaddq_s16,
    vqsubq_s16,
    vmaxq_s16,
    vminq_s16,
    vceqq_s16,
    vcgtq_s16,
    vandq_s16,
    vorrq_s16,
    vshlq_n_s16,
    vextq_s16,
    vsetq_lane_s16,
    vreinterpretq_s16_u16,
    vreinterpretq_u16_s16,
    vmaxvq_u16,
    widen_i16,
    narrow_i16
);
lanes!(
    I32x8,
    i32,
    int32x4_t,
    8,
    i32::MIN / 4,
    vdupq_n_s32,
    vaddq_s32,
    vsubq_s32,
    vmaxq_s32,
    vminq_s32,
    vceqq_s32,
    vcgtq_s32,
    vandq_s32,
    vorrq_s32,
    vshlq_n_s32,
    vextq_s32,
    vsetq_lane_s32,
    vreinterpretq_s32_u32,
    vreinterpretq_u32_s32,
    vmaxvq_u32,
    widen_i32,
    narrow_i32
);

/// Sign-extends 16 bytes to two registers of 16-bit lanes.
#[inline]
fn widen_i16(bytes: &[i8]) -> (int16x8_t, int16x8_t) {
    let bytes = &bytes[..16];
    // SAFETY: `bytes` is 16 readable bytes; the load is unaligned.
    let v = unsafe { vld1q_s8(bytes.as_ptr()) };
    neon!((vmovl_s8(vget_low_s8(v)), vmovl_high_s8(v)))
}

/// Sign-extends 8 bytes to two registers of 32-bit lanes.
#[inline]
fn widen_i32(bytes: &[i8]) -> (int32x4_t, int32x4_t) {
    let bytes = &bytes[..8];
    // SAFETY: `bytes` is 8 readable bytes; the load is unaligned.
    let v = unsafe { vld1_s8(bytes.as_ptr()) };
    neon!({
        let v = vmovl_s8(v);
        (vmovl_s16(vget_low_s16(v)), vmovl_high_s16(v))
    })
}

/// Narrows 16 lanes to bytes with unsigned saturation, written to `out[..16]`.
#[inline]
fn narrow_i16(lo: int16x8_t, hi: int16x8_t, out: &mut [u8]) {
    let out = &mut out[..16];
    let bytes = neon!(vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi)));
    // SAFETY: `out` is 16 writable bytes; the store is unaligned.
    unsafe { vst1q_u8(out.as_mut_ptr(), bytes) };
}

/// Narrows 8 lanes to bytes with unsigned saturation, written to `out[..8]`.
#[inline]
fn narrow_i32(lo: int32x4_t, hi: int32x4_t, out: &mut [u8]) {
    let out = &mut out[..8];
    let bytes = neon!(vqmovn_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi))));
    // SAFETY: `out` is 8 writable bytes; the store is unaligned.
    unsafe { vst1_u8(out.as_mut_ptr(), bytes) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lanes16() -> [i16; 16] {
        std::array::from_fn(|l| l as i16 * 100 - 700)
    }

    #[test]
    fn shift_in_moves_every_lane_up_one() {
        let v = I16x16::from_array(lanes16());
        let want: [i16; 16] = std::array::from_fn(|l| if l == 0 { 42 } else { lanes16()[l - 1] });
        assert_eq!(v.shift_in(42).to_array(), want);
        let w = I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(w.shift_in(-9).to_array(), [-9, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn sixteen_bit_arithmetic_saturates_and_thirty_two_bit_wraps() {
        let big = I16x16::splat(i16::MAX - 1);
        assert_eq!(big.add(I16x16::splat(5)).to_array(), [i16::MAX; 16]);
        assert_eq!(I16x16::splat(i16::MIN).sub(I16x16::splat(1)).to_array(), [i16::MIN; 16]);
        assert_eq!(I32x8::splat(i32::MAX).add(I32x8::splat(1)).to_array(), [i32::MIN; 8]);
    }

    #[test]
    fn comparisons_masks_and_first_lane() {
        let v = I16x16::from_array(lanes16());
        let gt = v.gt(I16x16::splat(0)).to_array();
        assert_eq!(gt, lanes16().map(|x| if x > 0 { -1 } else { 0 }));
        assert!(gt.iter().any(|&x| x != 0) && v.gt(I16x16::splat(0)).any());
        assert!(!v.gt(I16x16::splat(i16::MAX)).any());
        let only_last = I16x16::from_array(std::array::from_fn(|l| if l == 15 { 1 } else { 0 }));
        assert!(only_last.any());
        assert_eq!(v.with_first(7).to_array()[..2], [7, lanes16()[1]]);
        assert_eq!(I16x16::splat(3).shl4().to_array(), [48; 16]);
    }

    #[test]
    fn widen_and_narrow_keep_lane_order() {
        let bytes: [i8; 16] = std::array::from_fn(|l| l as i8 - 8);
        assert_eq!(I16x16::widen(&bytes).to_array(), bytes.map(i16::from));
        assert_eq!(I32x8::widen(&bytes).to_array(), std::array::from_fn(|l| i32::from(bytes[l])));
        let mut out = [0u8; 16];
        I16x16::from_array(std::array::from_fn(|l| l as i16 * 3)).narrow(&mut out);
        assert_eq!(out, std::array::from_fn(|l| l as u8 * 3));
        let mut out = [0u8; 8];
        I32x8::from_array([0, 1, 2, 3, 60, 61, 62, 63]).narrow(&mut out);
        assert_eq!(out, [0, 1, 2, 3, 60, 61, 62, 63]);
    }

    #[test]
    fn byte_lookup_and_compare() {
        let codes: [u8; 32] = std::array::from_fn(|l| (l % 16) as u8);
        let table_bytes: [i8; 16] = std::array::from_fn(|l| l as i8 * 2 - 10);
        // SAFETY: 16 readable bytes.
        let table = unsafe { vld1q_s8(table_bytes.as_ptr()) };
        let got = I8x32::load(&codes).lookup(table).to_array();
        assert_eq!(got, codes.map(|c| table_bytes[c as usize]));
        let eq = I8x32::load(&codes).eq(I8x32::splat(3)).to_array();
        assert_eq!(eq, codes.map(|c| if c == 3 { -1 } else { 0 }));
    }
}
