//! Score-only global alignment with Farrar's striped SIMD layout (AVX2 or NEON).
//!
//! Computes exactly the score of [`super::gotoh::score`]: the recurrences are the same
//! affine-gap model expressed as H/E/F, and trailing-gap-free scoring takes the best cell
//! of the last row or column. 16-bit lanes are used when the score range allows it.

use super::gotoh;
use super::scoring::{MATCH_SCORE, MATRIX_LETTERS, MISMATCH_SCORE, Scoring, Substitution, matrix_index};

/// Reusable per-thread buffers for [`score`].
#[derive(Debug, Default)]
pub struct ScoreBuffers {
    pub(super) codes_a: Vec<u8>,
    pub(super) codes_b: Vec<u8>,
    pub(super) table: Vec<i32>,
    pub(super) alphabet: usize,
    #[cfg(target_arch = "x86_64")]
    simd: avx2::Buffers,
    #[cfg(target_arch = "aarch64")]
    simd: neon::Buffers,
}

impl ScoreBuffers {
    /// Maps both sequences to dense letter codes and fills the letter-pair score table.
    pub(super) fn encode(&mut self, a: &[u8], b: &[u8], sub: &Substitution) {
        self.codes_a.clear();
        self.codes_b.clear();
        self.table.clear();
        match sub {
            Substitution::Matrix(m) => {
                self.alphabet = MATRIX_LETTERS;
                self.codes_a.extend(a.iter().map(|&c| matrix_index(c) as u8));
                self.codes_b.extend(b.iter().map(|&c| matrix_index(c) as u8));
                self.table.extend(m.iter().flat_map(|row| row.iter().copied()));
            }
            Substitution::MatchMismatch => {
                let mut code = [u8::MAX; 256];
                let mut next = 0u8;
                let mut map = |c: u8| {
                    if code[c as usize] == u8::MAX {
                        code[c as usize] = next;
                        next += 1;
                    }
                    code[c as usize]
                };
                for &c in a {
                    let x = map(c);
                    self.codes_a.push(x);
                }
                for &c in b {
                    let x = map(c);
                    self.codes_b.push(x);
                }
                let k = next as usize;
                self.alphabet = k;
                self.table.extend((0..k * k).map(|i| if i / k == i % k { MATCH_SCORE } else { MISMATCH_SCORE }));
            }
        }
    }
}

/// Global alignment score of `a` against `b`, identical to Biopython's `aligner.score`.
pub fn score(a: &[u8], b: &[u8], scoring: &Scoring, buf: &mut ScoreBuffers) -> i32 {
    if a.is_empty() || b.is_empty() {
        return gotoh::score(a, b, scoring);
    }
    debug_assert!(scoring.open <= scoring.extend, "lazy-F loop assumes open <= extend");
    #[cfg(target_arch = "aarch64")]
    {
        buf.encode(a, b, &scoring.sub);
        let (lo, hi) = score_range(a.len(), b.len(), scoring, &buf.table);
        // As for AVX2 below.
        let bias16 = (hi - lo < 64_000).then(|| -(lo + hi) / 2);
        use crate::neon::{I16x16, I32x8};
        return match (scoring.open == scoring.extend, bias16) {
            (true, Some(bias)) => neon::score_linear::<16, I16x16>(buf, scoring, bias),
            (true, None) => neon::score_linear::<8, I32x8>(buf, scoring, 0),
            (false, Some(bias)) => neon::score_affine::<16, I16x16>(buf, scoring, bias),
            (false, None) => neon::score_affine::<8, I32x8>(buf, scoring, 0),
        };
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        buf.encode(a, b, &scoring.sub);
        let (lo, hi) = score_range(a.len(), b.len(), scoring, &buf.table);
        // 16-bit lanes hold any span below 65536 once scores are shifted to centre it.
        let bias16 = (hi - lo < 64_000).then(|| -(lo + hi) / 2);
        // SAFETY: AVX2 support was checked at runtime just above.
        return unsafe {
            match (scoring.open == scoring.extend, bias16) {
                (true, Some(bias)) => avx2::score_linear_i16(buf, scoring, bias),
                (true, None) => avx2::score_linear_i32(buf, scoring, 0),
                (false, Some(bias)) => avx2::score_i16(buf, scoring, bias),
                (false, None) => avx2::score_i32(buf, scoring, 0),
            }
        };
    }
    #[allow(unreachable_code, reason = "aarch64 always returns above")]
    gotoh::score(a, b, scoring)
}

/// Bounds on every value the kernels store for an `m` x `n` alignment, with a margin.
///
/// The best path into cell (i, j) scores at least as well as `min(i, j)` worst
/// substitutions followed by one gap run of `|i - j|`, and at most a perfect match along
/// the shorter side.
pub(super) fn score_range(m: usize, n: usize, scoring: &Scoring, table: &[i32]) -> (i32, i32) {
    let (short, long) = (m.min(n) as i32, m.max(n) as i32);
    let min_sub = table.iter().copied().min().unwrap_or(0).min(0);
    let max_sub = table.iter().copied().max().unwrap_or(0).max(0);
    let (open, extend) = (scoring.open.min(0), scoring.extend.min(0));
    let interior = min_sub.min(extend) * long + open;
    let border = scoring.left_gap(m.max(n));
    // Gap states sit at most one more gap open below the cell they came from.
    (interior.min(border) + 2 * open - 64, max_sub * short + 64)
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{ScoreBuffers, Scoring};

    #[derive(Debug, Default)]
    pub(super) struct Buffers {
        profile: Vec<__m256i>,
        slot: Vec<usize>,
        h_prev: Vec<__m256i>,
        h_cur: Vec<__m256i>,
        e: Vec<__m256i>,
    }

    macro_rules! striped_kernel {
        ($name:ident, $elem:ty, $lanes:expr, $neg:expr, $add:ident, $max:ident, $set1:ident, $cmpgt:ident,
         $shift_bytes:expr, $insert:ident) => {
            /// # Safety
            ///
            /// The CPU must support AVX2.
            #[target_feature(enable = "avx2")]
            /// `bias` is added to every stored score so the range fits the lanes and
            /// subtracted again from the result.
            pub(super) unsafe fn $name(buf: &mut ScoreBuffers, scoring: &Scoring, bias: i32) -> i32 {
                const L: usize = $lanes;
                const NEG: $elem = $neg;
                let (m, n) = (buf.codes_a.len(), buf.codes_b.len());
                let segs = m.div_ceil(L);
                let k = buf.alphabet;
                let clamp = |v: i32| (v + bias).clamp(NEG as i32, <$elem>::MAX as i32) as $elem;
                let gap = |len: usize| scoring.left_gap(len);
                let to_vec = |lanes: [$elem; L]| -> __m256i {
                    // SAFETY: [$elem; L] and __m256i are both 32 bytes of plain integers.
                    unsafe { std::mem::transmute::<[$elem; L], __m256i>(lanes) }
                };
                let to_lanes = |v: __m256i| -> [$elem; L] {
                    // SAFETY: as above, in the other direction.
                    unsafe { std::mem::transmute::<__m256i, [$elem; L]>(v) }
                };
                let shift_in = |v: __m256i, x: $elem| -> __m256i {
                    let t = _mm256_permute2x128_si256::<0x08>(v, v);
                    $insert::<0>(_mm256_alignr_epi8::<{ 16 - $shift_bytes }>(v, t), x)
                };

                let simd = &mut buf.simd;
                // Query profile, built only for letters that occur in `b`.
                simd.slot.clear();
                simd.slot.resize(k, usize::MAX);
                simd.profile.clear();
                for &c in &buf.codes_b {
                    let c = c as usize;
                    if simd.slot[c] != usize::MAX {
                        continue;
                    }
                    simd.slot[c] = simd.profile.len() / segs;
                    for t in 0..segs {
                        let mut lanes = [0 as $elem; L];
                        for (lane, v) in lanes.iter_mut().enumerate() {
                            let row = lane * segs + t;
                            if row < m {
                                *v = buf.table[buf.codes_a[row] as usize * k + c] as $elem;
                            }
                        }
                        simd.profile.push(to_vec(lanes));
                    }
                }
                let lane_rows = |t: usize| -> [usize; L] { std::array::from_fn(|lane| lane * segs + t + 1) };
                simd.h_prev.clear();
                simd.e.clear();
                for t in 0..segs {
                    let rows = lane_rows(t);
                    simd.h_prev.push(to_vec(rows.map(|r| clamp(gap(r)))));
                    simd.e.push(to_vec(rows.map(|r| clamp(gap(r) + scoring.open))));
                }
                simd.h_cur.clear();
                simd.h_cur.resize(segs, $set1(NEG));

                let v_open = $set1(scoring.open as $elem);
                let v_ext = $set1(scoring.extend as $elem);
                let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
                let mut best_last_row = i32::MIN;
                for (j, &c) in buf.codes_b.iter().enumerate() {
                    let col = j + 1;
                    let prof = &simd.profile[simd.slot[c as usize] * segs..][..segs];
                    let top_prev = if col == 1 { 0 } else { gap(col - 1) };
                    let mut v_f = $insert::<0>($set1(NEG), clamp(gap(col) + scoring.open));
                    let mut v_h = shift_in(simd.h_prev[segs - 1], clamp(top_prev));
                    for t in 0..segs {
                        v_h = $add(v_h, prof[t]);
                        v_h = $max(v_h, simd.e[t]);
                        v_h = $max(v_h, v_f);
                        simd.h_cur[t] = v_h;
                        let v_h_open = $add(v_h, v_open);
                        simd.e[t] = $max($add(simd.e[t], v_ext), v_h_open);
                        v_f = $max($add(v_f, v_ext), v_h_open);
                        v_h = simd.h_prev[t];
                    }
                    // Lazy F: carry vertical gaps across lane boundaries until they stop mattering.
                    v_f = shift_in(v_f, NEG);
                    'lazy: for _ in 0..L {
                        for t in 0..segs {
                            let h = simd.h_cur[t];
                            let more = $cmpgt($add(v_f, v_ext), $add(h, v_open));
                            if _mm256_movemask_epi8(more) == 0 {
                                break 'lazy;
                            }
                            let h = $max(h, v_f);
                            simd.h_cur[t] = h;
                            simd.e[t] = $max(simd.e[t], $add(h, v_open));
                            v_f = $add(v_f, v_ext);
                        }
                        v_f = shift_in(v_f, NEG);
                    }
                    if scoring.free_right_gaps {
                        best_last_row = best_last_row.max(to_lanes(simd.h_cur[last_seg])[last_lane] as i32 - bias);
                    }
                    std::mem::swap(&mut simd.h_prev, &mut simd.h_cur);
                }
                // After the swap, h_prev holds the last column.
                if !scoring.free_right_gaps {
                    return to_lanes(simd.h_prev[last_seg])[last_lane] as i32 - bias;
                }
                let mut best = best_last_row.max(gap(n)).max(gap(m));
                for t in 0..segs {
                    let lanes = to_lanes(simd.h_prev[t]);
                    for (lane, &v) in lanes.iter().enumerate() {
                        if lane * segs + t < m {
                            best = best.max(v as i32 - bias);
                        }
                    }
                }
                best
            }
        };
    }

    /// Linear gap costs (`open == extend`): every gap step costs `g`, so a horizontal gap
    /// into a cell is just its left neighbour plus `g` and no gap state is stored.
    macro_rules! linear_kernel {
        ($name:ident, $elem:ty, $lanes:expr, $neg:expr, $add:ident, $max:ident, $set1:ident, $cmpgt:ident,
         $shift_bytes:expr, $insert:ident) => {
            /// # Safety
            ///
            /// The CPU must support AVX2.
            #[target_feature(enable = "avx2")]
            /// `bias` is added to every stored score so the range fits the lanes and
            /// subtracted again from the result.
            pub(super) unsafe fn $name(buf: &mut ScoreBuffers, scoring: &Scoring, bias: i32) -> i32 {
                const L: usize = $lanes;
                const NEG: $elem = $neg;
                let (m, n) = (buf.codes_a.len(), buf.codes_b.len());
                let segs = m.div_ceil(L);
                let k = buf.alphabet;
                let clamp = |v: i32| (v + bias).clamp(NEG as i32, <$elem>::MAX as i32) as $elem;
                let gap = |len: usize| scoring.left_gap(len);
                let to_vec = |lanes: [$elem; L]| -> __m256i {
                    // SAFETY: [$elem; L] and __m256i are both 32 bytes of plain integers.
                    unsafe { std::mem::transmute::<[$elem; L], __m256i>(lanes) }
                };
                let to_lanes = |v: __m256i| -> [$elem; L] {
                    // SAFETY: as above, in the other direction.
                    unsafe { std::mem::transmute::<__m256i, [$elem; L]>(v) }
                };
                let shift_in = |v: __m256i, x: $elem| -> __m256i {
                    let t = _mm256_permute2x128_si256::<0x08>(v, v);
                    $insert::<0>(_mm256_alignr_epi8::<{ 16 - $shift_bytes }>(v, t), x)
                };

                let simd = &mut buf.simd;
                simd.slot.clear();
                simd.slot.resize(k, usize::MAX);
                simd.profile.clear();
                for &c in &buf.codes_b {
                    let c = c as usize;
                    if simd.slot[c] != usize::MAX {
                        continue;
                    }
                    simd.slot[c] = simd.profile.len() / segs;
                    for t in 0..segs {
                        let mut lanes = [0 as $elem; L];
                        for (lane, v) in lanes.iter_mut().enumerate() {
                            let row = lane * segs + t;
                            if row < m {
                                *v = buf.table[buf.codes_a[row] as usize * k + c] as $elem;
                            }
                        }
                        simd.profile.push(to_vec(lanes));
                    }
                }
                simd.h_prev.clear();
                for t in 0..segs {
                    simd.h_prev.push(to_vec(std::array::from_fn(|lane| clamp(gap(lane * segs + t + 1)))));
                }
                simd.h_cur.clear();
                simd.h_cur.resize(segs, $set1(NEG));

                let g = scoring.extend;
                let v_g = $set1(g as $elem);
                let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
                let mut best_last_row = i32::MIN;
                for (j, &c) in buf.codes_b.iter().enumerate() {
                    let col = j + 1;
                    let prof = &simd.profile[simd.slot[c as usize] * segs..][..segs];
                    let top_prev = if col == 1 { 0 } else { gap(col - 1) };
                    let mut v_f = $insert::<0>($set1(NEG), clamp(gap(col) + g));
                    let mut v_diag = shift_in(simd.h_prev[segs - 1], clamp(top_prev));
                    for t in 0..segs {
                        let left = simd.h_prev[t];
                        let h = $max($max($add(v_diag, prof[t]), $add(left, v_g)), v_f);
                        simd.h_cur[t] = h;
                        v_f = $add(h, v_g);
                        v_diag = left;
                    }
                    // Lazy F: carry vertical gaps across lane boundaries until they stop mattering.
                    v_f = shift_in(v_f, NEG);
                    'lazy: for _ in 0..L {
                        for t in 0..segs {
                            let h = simd.h_cur[t];
                            if _mm256_movemask_epi8($cmpgt(v_f, h)) == 0 {
                                break 'lazy;
                            }
                            simd.h_cur[t] = $max(h, v_f);
                            v_f = $add(v_f, v_g);
                        }
                        v_f = shift_in(v_f, NEG);
                    }
                    if scoring.free_right_gaps {
                        best_last_row = best_last_row.max(to_lanes(simd.h_cur[last_seg])[last_lane] as i32 - bias);
                    }
                    std::mem::swap(&mut simd.h_prev, &mut simd.h_cur);
                }
                if !scoring.free_right_gaps {
                    return to_lanes(simd.h_prev[last_seg])[last_lane] as i32 - bias;
                }
                let mut best = best_last_row.max(gap(n)).max(gap(m));
                for t in 0..segs {
                    let lanes = to_lanes(simd.h_prev[t]);
                    for (lane, &v) in lanes.iter().enumerate() {
                        if lane * segs + t < m {
                            best = best.max(v as i32 - bias);
                        }
                    }
                }
                best
            }
        };
    }

    linear_kernel!(
        score_linear_i16,
        i16,
        16,
        i16::MIN,
        _mm256_adds_epi16,
        _mm256_max_epi16,
        _mm256_set1_epi16,
        _mm256_cmpgt_epi16,
        2,
        _mm256_insert_epi16
    );
    linear_kernel!(
        score_linear_i32,
        i32,
        8,
        i32::MIN / 4,
        _mm256_add_epi32,
        _mm256_max_epi32,
        _mm256_set1_epi32,
        _mm256_cmpgt_epi32,
        4,
        _mm256_insert_epi32
    );

    striped_kernel!(
        score_i16,
        i16,
        16,
        i16::MIN,
        _mm256_adds_epi16,
        _mm256_max_epi16,
        _mm256_set1_epi16,
        _mm256_cmpgt_epi16,
        2,
        _mm256_insert_epi16
    );
    striped_kernel!(
        score_i32,
        i32,
        8,
        i32::MIN / 4,
        _mm256_add_epi32,
        _mm256_max_epi32,
        _mm256_set1_epi32,
        _mm256_cmpgt_epi32,
        4,
        _mm256_insert_epi32
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! The AVX2 kernels' algorithms, generic over the lane width rather than expanded from
    //! macros: NEON needs no `target_feature`, so generic code inlines fully.

    use super::{ScoreBuffers, Scoring};
    use crate::neon::{I16x16, I32x8, Lanes};

    /// The vectors of one lane width.
    #[derive(Debug)]
    pub(super) struct Vecs<V> {
        profile: Vec<V>,
        slot: Vec<usize>,
        h_prev: Vec<V>,
        h_cur: Vec<V>,
        e: Vec<V>,
    }

    impl<V> Default for Vecs<V> {
        fn default() -> Self {
            Self { profile: Vec::new(), slot: Vec::new(), h_prev: Vec::new(), h_cur: Vec::new(), e: Vec::new() }
        }
    }

    #[derive(Debug, Default)]
    pub(super) struct Buffers {
        w16: Vecs<I16x16>,
        w32: Vecs<I32x8>,
    }

    /// A lane width, with its buffers.
    pub(super) trait Width<const L: usize>: Lanes<L> {
        fn vecs(buf: &mut Buffers) -> &mut Vecs<Self>;
    }

    impl Width<16> for I16x16 {
        fn vecs(buf: &mut Buffers) -> &mut Vecs<Self> {
            &mut buf.w16
        }
    }

    impl Width<8> for I32x8 {
        fn vecs(buf: &mut Buffers) -> &mut Vecs<Self> {
            &mut buf.w32
        }
    }

    /// Splits `buf` into its encoded sequences and the vectors of lane width `V`.
    fn split<const L: usize, V: Width<L>>(buf: &mut ScoreBuffers) -> (Encoded<'_>, &mut Vecs<V>) {
        let enc = Encoded { a: &buf.codes_a, b: &buf.codes_b, table: &buf.table, k: buf.alphabet };
        (enc, V::vecs(&mut buf.simd))
    }

    /// Letter codes and the letter-pair score table.
    #[derive(Clone, Copy)]
    struct Encoded<'a> {
        a: &'a [u8],
        b: &'a [u8],
        table: &'a [i32],
        k: usize,
    }

    impl<V> Vecs<V> {
        /// Query profile, built only for letters that occur in `b`.
        fn build_profile<const L: usize>(&mut self, enc: Encoded<'_>, segs: usize)
        where
            V: Lanes<L>,
        {
            let m = enc.a.len();
            self.slot.clear();
            self.slot.resize(enc.k, usize::MAX);
            self.profile.clear();
            for &c in enc.b {
                let c = c as usize;
                if self.slot[c] != usize::MAX {
                    continue;
                }
                self.slot[c] = self.profile.len() / segs;
                for t in 0..segs {
                    self.profile.push(V::from_array(std::array::from_fn(|lane| {
                        let row = lane * segs + t;
                        V::cast(if row < m { enc.table[enc.a[row] as usize * enc.k + c] } else { 0 })
                    })));
                }
            }
        }

        /// The score from the last column (in `h_prev` after the final swap), given the
        /// best last-row value seen when trailing gaps are free.
        fn finish<const L: usize>(
            &self,
            scoring: &Scoring,
            (m, n): (usize, usize),
            bias: i32,
            best_last_row: i32,
        ) -> i32
        where
            V: Lanes<L>,
        {
            let segs = m.div_ceil(L);
            let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
            if !scoring.free_right_gaps {
                return self.h_prev[last_seg].to_array()[last_lane].into() - bias;
            }
            let mut best = best_last_row.max(scoring.left_gap(n)).max(scoring.left_gap(m));
            for t in 0..segs {
                for (lane, &v) in self.h_prev[t].to_array().iter().enumerate() {
                    if lane * segs + t < m {
                        best = best.max(v.into() - bias);
                    }
                }
            }
            best
        }
    }

    /// `v + bias`, clamped to the lane range.
    fn clamp<const L: usize, V: Lanes<L>>(v: i32, bias: i32) -> V::Elem {
        V::cast((v + bias).clamp(V::NEG.into(), V::MAX.into()))
    }

    /// Affine gap costs. `bias` is added to every stored score so the range fits the lanes
    /// and subtracted again from the result.
    pub(super) fn score_affine<const L: usize, V: Width<L>>(
        buf: &mut ScoreBuffers,
        scoring: &Scoring,
        bias: i32,
    ) -> i32 {
        let (enc, simd) = split::<L, V>(buf);
        let (m, n) = (enc.a.len(), enc.b.len());
        let segs = m.div_ceil(L);
        let clamp = |v: i32| clamp::<L, V>(v, bias);
        let gap = |len: usize| scoring.left_gap(len);
        simd.build_profile::<L>(enc, segs);
        let lane_rows = |t: usize| -> [usize; L] { std::array::from_fn(|lane| lane * segs + t + 1) };
        simd.h_prev.clear();
        simd.e.clear();
        for t in 0..segs {
            let rows = lane_rows(t);
            simd.h_prev.push(V::from_array(rows.map(|r| clamp(gap(r)))));
            simd.e.push(V::from_array(rows.map(|r| clamp(gap(r) + scoring.open))));
        }
        simd.h_cur.clear();
        simd.h_cur.resize(segs, V::splat(V::NEG));

        let v_open = V::splat(V::cast(scoring.open));
        let v_ext = V::splat(V::cast(scoring.extend));
        let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
        let mut best_last_row = i32::MIN;
        for (j, &c) in enc.b.iter().enumerate() {
            let col = j + 1;
            let prof = &simd.profile[simd.slot[c as usize] * segs..][..segs];
            let top_prev = if col == 1 { 0 } else { gap(col - 1) };
            let mut v_f = V::splat(V::NEG).with_first(clamp(gap(col) + scoring.open));
            let mut v_h = simd.h_prev[segs - 1].shift_in(clamp(top_prev));
            for (t, &p) in prof.iter().enumerate() {
                v_h = v_h.add(p).max(simd.e[t]).max(v_f);
                simd.h_cur[t] = v_h;
                let v_h_open = v_h.add(v_open);
                simd.e[t] = simd.e[t].add(v_ext).max(v_h_open);
                v_f = v_f.add(v_ext).max(v_h_open);
                v_h = simd.h_prev[t];
            }
            // Lazy F: carry vertical gaps across lane boundaries until they stop mattering.
            v_f = v_f.shift_in(V::NEG);
            'lazy: for _ in 0..L {
                for t in 0..segs {
                    let h = simd.h_cur[t];
                    if !v_f.add(v_ext).gt(h.add(v_open)).any() {
                        break 'lazy;
                    }
                    let h = h.max(v_f);
                    simd.h_cur[t] = h;
                    simd.e[t] = simd.e[t].max(h.add(v_open));
                    v_f = v_f.add(v_ext);
                }
                v_f = v_f.shift_in(V::NEG);
            }
            if scoring.free_right_gaps {
                best_last_row = best_last_row.max(simd.h_cur[last_seg].to_array()[last_lane].into() - bias);
            }
            std::mem::swap(&mut simd.h_prev, &mut simd.h_cur);
        }
        simd.finish::<L>(scoring, (m, n), bias, best_last_row)
    }

    /// Linear gap costs (`open == extend`): a horizontal gap into a cell is just its left
    /// neighbour plus `g` and no gap state is stored.
    pub(super) fn score_linear<const L: usize, V: Width<L>>(
        buf: &mut ScoreBuffers,
        scoring: &Scoring,
        bias: i32,
    ) -> i32 {
        let (enc, simd) = split::<L, V>(buf);
        let (m, n) = (enc.a.len(), enc.b.len());
        let segs = m.div_ceil(L);
        let clamp = |v: i32| clamp::<L, V>(v, bias);
        let gap = |len: usize| scoring.left_gap(len);
        simd.build_profile::<L>(enc, segs);
        simd.h_prev.clear();
        for t in 0..segs {
            simd.h_prev.push(V::from_array(std::array::from_fn(|lane| clamp(gap(lane * segs + t + 1)))));
        }
        simd.h_cur.clear();
        simd.h_cur.resize(segs, V::splat(V::NEG));

        let g = scoring.extend;
        let v_g = V::splat(V::cast(g));
        let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
        let mut best_last_row = i32::MIN;
        for (j, &c) in enc.b.iter().enumerate() {
            let col = j + 1;
            let prof = &simd.profile[simd.slot[c as usize] * segs..][..segs];
            let top_prev = if col == 1 { 0 } else { gap(col - 1) };
            let mut v_f = V::splat(V::NEG).with_first(clamp(gap(col) + g));
            let mut v_diag = simd.h_prev[segs - 1].shift_in(clamp(top_prev));
            for (t, &p) in prof.iter().enumerate() {
                let left = simd.h_prev[t];
                let h = v_diag.add(p).max(left.add(v_g)).max(v_f);
                simd.h_cur[t] = h;
                v_f = h.add(v_g);
                v_diag = left;
            }
            // Lazy F: carry vertical gaps across lane boundaries until they stop mattering.
            v_f = v_f.shift_in(V::NEG);
            'lazy: for _ in 0..L {
                for t in 0..segs {
                    let h = simd.h_cur[t];
                    if !v_f.gt(h).any() {
                        break 'lazy;
                    }
                    simd.h_cur[t] = h.max(v_f);
                    v_f = v_f.add(v_g);
                }
                v_f = v_f.shift_in(V::NEG);
            }
            if scoring.free_right_gaps {
                best_last_row = best_last_row.max(simd.h_cur[last_seg].to_array()[last_lane].into() - bias);
            }
            std::mem::swap(&mut simd.h_prev, &mut simd.h_cur);
        }
        simd.finish::<L>(scoring, (m, n), bias, best_last_row)
    }
}

#[cfg(test)]
mod tests {
    use super::super::scoring::{GapKind, MatrixKind};
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    fn random_seq(seed: &mut u64, len: usize, letters: &[u8]) -> Vec<u8> {
        (0..len).map(|_| letters[lcg(seed) as usize % letters.len()]).collect()
    }

    #[test]
    #[ignore = "single-thread throughput of the linear-gap kernel on 1000 x 1000 TVR alignments"]
    fn bench_linear_1000() {
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false);
        let mut seed = 1;
        let letters = b"CCCCCCCCCCCCCCCCCCCCCCDEFAAAGHC";
        let seqs: Vec<Vec<u8>> = (0..64).map(|_| random_seq(&mut seed, 1000, letters)).collect();
        let mut buf = ScoreBuffers::default();
        let t = std::time::Instant::now();
        let mut total = 0i64;
        for i in 0..seqs.len() {
            for j in 0..seqs.len() {
                total += i64::from(score(&seqs[i], &seqs[j], &s, &mut buf));
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let cells = (seqs.len() * seqs.len()) as f64 * 1e6;
        eprintln!("{:.2} Gcells/s single thread (checksum {total})", cells / secs / 1e9);
        use rayon::prelude::*;
        let t = std::time::Instant::now();
        let n = rayon::current_num_threads();
        let total: i64 = (0..n)
            .into_par_iter()
            .map(|_| {
                let mut buf = ScoreBuffers::default();
                let mut t = 0i64;
                for a in &seqs {
                    for b in &seqs {
                        t += i64::from(score(a, b, &s, &mut buf));
                    }
                }
                t
            })
            .sum();
        let secs = t.elapsed().as_secs_f64();
        eprintln!("{n} threads: {:.2} Gcells/s (checksum {total})", n as f64 * cells / secs / 1e9);
    }

    #[test]
    fn striped_matches_reference() {
        let schemes = [
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false),
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, true),
            Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, true),
            Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, false),
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, true),
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, false),
            Scoring::new(Substitution::MatchMismatch, GapKind::Msa, true),
        ];
        let mut seed = 42;
        let mut buf = ScoreBuffers::default();
        for trial in 0..400 {
            let letters: &[u8] = if trial % 3 == 0 { b"ACGT" } else { b"CCCCCCADEFAC" };
            let la = 1 + lcg(&mut seed) as usize % if trial % 10 == 0 { 9000 } else { 300 };
            let lb = 1 + lcg(&mut seed) as usize % if trial % 10 == 0 { 9000 } else { 300 };
            let a = random_seq(&mut seed, la, letters);
            let mut b = a.clone();
            b.truncate(lb.min(a.len()));
            if trial % 2 == 0 {
                b = random_seq(&mut seed, lb, letters);
            }
            for s in &schemes {
                assert_eq!(score(&a, &b, s, &mut buf), gotoh::score(&a, &b, s), "trial {trial} lens {la},{lb}");
            }
        }
    }

    /// Pairs just inside and outside the 16-bit range, at both extremes of the score span
    /// (identical sequences and sequences with no letter in common).
    #[test]
    fn biased_16_bit_lanes_at_the_range_limit() {
        let schemes = [
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false),
            Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, true),
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, false),
        ];
        let mut seed = 7;
        let mut buf = ScoreBuffers::default();
        for (la, lb) in [(6900, 6900), (7150, 7150), (2500, 9800), (9800, 120)] {
            let random = random_seq(&mut seed, la, b"CCCCCCADEFAC");
            let pairs = [
                (random.clone(), random[..lb.min(la)].to_vec()),
                (vec![b'C'; la], vec![b'D'; lb]),
                (random, random_seq(&mut seed, lb, b"CCCCCCADEFAC")),
            ];
            for (a, b) in &pairs {
                for s in &schemes {
                    assert_eq!(score(a, b, s, &mut buf), gotoh::score(a, b, s), "lens {},{}", a.len(), b.len());
                }
            }
        }
    }
}
