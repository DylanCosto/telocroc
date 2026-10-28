//! Score-only global alignment of up to 32 independent pairs at once (AVX2 or NEON), one
//! pair per 8-bit lane.
//!
//! Substitution scores must have the form `x == y ? same[x] : other[x] + other[y]`, as the
//! TVR matrix and match/mismatch scoring do. Subtracting `other` of every letter consumed
//! so far, `H'(i, j) = H(i, j) - sum(other[a[..i]]) - sum(other[b[..j]])`, turns a mismatch
//! into a step of 0, a match into `same[x] - 2 other[x]`, and each gap into `g - other` of
//! the letter it skips; the per-letter values are looked up with `pshufb` (`tbl` on NEON).
//!
//! Lanes hold *differences* of `H'` between neighbouring cells rather than scores (the
//! difference recurrence of Suzuki and Kasahara). With linear gaps every difference is
//! bounded by a few times the largest step, so 8 bits are exact for any sequence length.
//! The score is the top-right border plus the vertical differences down the last column.
//!
//! [`BatchScorer::new`] checks the preconditions; callers fall back to
//! [`super::striped::score`] when it returns `None`.

use super::scoring::{AMINO, GAP, Scoring, Substitution};

/// Pairs scored per call.
pub const LANES: usize = 32;

/// Letters the decomposition is built over (the matrix alphabet).
const ALPHABET: &[u8; 21] = b"ACDEFGHIKLMNPQRSTVWY-";

/// Scoring prepared for batched alignment.
#[derive(Clone, Debug)]
pub struct BatchScorer {
    /// Per byte: the lane code (`letter id << class bits | class`), or `None` if the letter
    /// is outside the alphabet.
    code: [Option<u8>; 256],
    /// Indexed by class: `same - 2 other` (the shifted match step), `g - other` (the
    /// shifted gap step) and `other` (to undo the shift).
    bonus: [i8; 16],
    step: [i8; 16],
    other: [i8; 16],
    class_mask: u8,
    gap: i8,
}

impl BatchScorer {
    /// Returns `None` when batched scoring cannot reproduce `scoring` exactly: affine or
    /// free trailing gaps, a substitution table that does not decompose, or scores too
    /// large for 8-bit differences. Also `None` without AVX2 or NEON.
    pub fn new(scoring: &Scoring) -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return None;
        if scoring.open != scoring.extend || scoring.free_right_gaps || !(-32..0).contains(&scoring.open) {
            return None;
        }
        let letters: Vec<u8> = match &scoring.sub {
            Substitution::Matrix(_) => ALPHABET.to_vec(),
            Substitution::MatchMismatch => (0..=255).collect(),
        };
        debug_assert!(AMINO.iter().chain([&GAP]).all(|c| ALPHABET.contains(c)));
        let score = |a: u8, b: u8| scoring.sub.score(a, b);
        // other[x] + other[y] = score(x, y) for x != y fixes other[x] from any two
        // further letters: other[x] = (s(x,y) + s(x,z) - s(y,z)) / 2.
        let half = |x: u8| -> Option<i32> {
            let mut rest = letters.iter().copied().filter(|&c| c != x);
            let (y, z) = (rest.next()?, rest.next()?);
            let twice = score(x, y) + score(x, z) - score(y, z);
            (twice % 2 == 0).then_some(twice / 2)
        };
        let mut classes: Vec<(i32, i32)> = Vec::new();
        let mut class_of = Vec::with_capacity(letters.len());
        for &x in &letters {
            let key = (score(x, x), half(x)?);
            let class = classes.iter().position(|&k| k == key).unwrap_or_else(|| {
                classes.push(key);
                classes.len() - 1
            });
            class_of.push(class);
        }
        for (i, &x) in letters.iter().enumerate() {
            for (j, &y) in letters.iter().enumerate() {
                let want =
                    if i == j { classes[class_of[i]].0 } else { classes[class_of[i]].1 + classes[class_of[j]].1 };
                if score(x, y) != want {
                    return None;
                }
            }
        }
        let class_bits = classes.len().next_power_of_two().trailing_zeros();
        // Codes must stay below 128 so pshufb reads the class from the low bits (tbl reads
        // the masked class).
        if classes.len() > 16 || letters.len() > 1 << (7 - class_bits) {
            return None;
        }
        let g = scoring.extend;
        let bonus = classes.iter().map(|&(s, o)| s - 2 * o);
        let step = classes.iter().map(|&(_, o)| g - o);
        let (bonus_lo, bonus_hi) = (bonus.clone().min()?.min(0), bonus.max()?.max(0));
        let (step_lo, step_hi) = (step.clone().min()?, step.max()?);
        // Every difference lies in [step_lo, bonus_hi - step_lo] (a cell is at least its
        // gap predecessor and at most the best step out of the diagonal); with the steps
        // added to them, all intermediate values must fit in i8.
        if step_lo < -30 || step_hi > 30 || bonus_lo < -30 || bonus_hi - 2 * step_lo > 100 {
            return None;
        }
        let mut code = [None; 256];
        for (id, (&x, &class)) in letters.iter().zip(&class_of).enumerate() {
            code[x as usize] = Some(((id as u8) << class_bits) | class as u8);
        }
        let (mut bonus, mut step, mut other) = ([0i8; 16], [0i8; 16], [0i8; 16]);
        for (c, &(s, o)) in classes.iter().enumerate() {
            bonus[c] = (s - 2 * o) as i8;
            step[c] = (g - o) as i8;
            other[c] = o as i8;
        }
        Some(Self { code, bonus, step, other, class_mask: (1 << class_bits) - 1, gap: g as i8 })
    }

    fn other_of(&self, code: u8) -> i8 {
        self.other[(code & self.class_mask) as usize]
    }

    /// Scores of up to [`LANES`] pairs, each identical to [`super::gotoh::score`], written
    /// to `out[..pairs.len()]`.
    ///
    /// # Panics
    ///
    /// If there are more than [`LANES`] pairs, a sequence is empty, or a letter is outside
    /// the scoring alphabet.
    pub fn score(&self, pairs: &[(&[u8], &[u8])], buf: &mut BatchBuffers, out: &mut [i32]) {
        assert!(pairs.len() <= LANES, "{} pairs in one batch", pairs.len());
        assert!(pairs.iter().all(|(a, b)| !a.is_empty() && !b.is_empty()), "empty sequence in a batch");
        if pairs.is_empty() {
            return;
        }
        let rows = pairs.iter().map(|p| p.0.len()).max().unwrap_or(0);
        let cols = pairs.iter().map(|p| p.1.len()).max().unwrap_or(0);
        let code = |c: u8| self.code[c as usize].unwrap_or_else(|| panic!("letter {:?} is not scored", c as char));
        // Lanes past their sequence's end hold letter code 0; cells there never reach a result.
        buf.a.clear();
        buf.a.resize(rows, [0; LANES]);
        buf.b.clear();
        buf.b.resize(cols, [0; LANES]);
        buf.ends.clear();
        buf.ends.resize(cols, 0);
        for (lane, (a, b)) in pairs.iter().enumerate() {
            let mut other_a = 0;
            for (row, &c) in buf.a.iter_mut().zip(a.iter()) {
                row[lane] = code(c);
                other_a += i32::from(self.other_of(row[lane]));
            }
            // The score is completed from these once column n is reached.
            out[lane] = i32::from(self.gap) * b.len() as i32 + other_a;
            for (col, &c) in buf.b.iter_mut().zip(b.iter()) {
                col[lane] = code(c);
            }
            buf.ends[b.len() - 1] |= 1 << lane;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `new` returned `Some`, so the CPU supports AVX2.
        unsafe {
            avx2::fill(self, pairs, buf, out);
        }
        #[cfg(target_arch = "aarch64")]
        neon::fill(self, pairs, buf, out);
    }
}

/// Whether this machine has a batched kernel (AVX2 or NEON).
#[cfg(test)]
pub(super) fn has_kernel() -> bool {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    cfg!(target_arch = "aarch64")
}

/// Reusable per-thread buffers for [`BatchScorer::score`].
#[derive(Debug, Default)]
pub struct BatchBuffers {
    /// Letter codes of each row (the `a` sequences), one byte per lane.
    a: Vec<[u8; LANES]>,
    /// Letter codes of each column (the `b` sequences).
    b: Vec<[u8; LANES]>,
    /// Per column, a bit for each lane whose `b` sequence ends there.
    ends: Vec<u32>,
    /// Vertical differences of the current column, one row per entry.
    #[cfg(target_arch = "x86_64")]
    v: Vec<std::arch::x86_64::__m256i>,
    #[cfg(target_arch = "aarch64")]
    v: Vec<crate::neon::I8x32>,
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{BatchBuffers, BatchScorer, LANES};

    /// Lookup tables, broadcast to every lane.
    struct Cells {
        bonus: __m256i,
        step: __m256i,
        mask: __m256i,
    }

    /// A row's letter codes with their match and gap steps.
    type Row = (__m256i, __m256i, __m256i);
    /// A column's letter codes with their gap steps.
    type Col = (__m256i, __m256i);

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load(lanes: &[u8; LANES]) -> __m256i {
        // SAFETY: a [u8; 32] is 32 readable bytes; the load is unaligned.
        unsafe { _mm256_loadu_si256(lanes.as_ptr().cast()) }
    }

    impl Cells {
        #[inline]
        #[target_feature(enable = "avx2")]
        fn row(&self, codes: &[u8; LANES]) -> Row {
            let x = load(codes);
            let class = _mm256_and_si256(x, self.mask);
            (x, _mm256_shuffle_epi8(self.bonus, class), _mm256_shuffle_epi8(self.step, class))
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        fn col(&self, codes: &[u8; LANES]) -> Col {
            let (y, _, step_y) = self.row(codes);
            (y, step_y)
        }

        /// One cell from the vertical difference to its left (`left`, previous column)
        /// and the horizontal difference above it (`up`, previous row). Returns the
        /// cell's vertical and horizontal differences.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn step(&self, (x, bonus_x, step_x): Row, (y, step_y): Col, left: __m256i, up: __m256i) -> (__m256i, __m256i) {
            // A gap from the left skips b[j] and one from above skips a[i].
            let diag = _mm256_and_si256(_mm256_cmpeq_epi8(x, y), bonus_x);
            let gaps = _mm256_max_epi8(_mm256_add_epi8(left, step_y), _mm256_add_epi8(up, step_x));
            let z = _mm256_max_epi8(diag, gaps);
            (_mm256_sub_epi8(z, up), _mm256_sub_epi8(z, left))
        }
    }

    /// Columns per pass over the rows (at most 4).
    const COLUMNS: usize = 4;

    /// Advances `v` across the `K` columns `cols`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn pass<const K: usize>(cells: &Cells, a: &[[u8; LANES]], cols: &[[u8; LANES]], v: &mut [__m256i]) {
        let ys: [Col; K] = std::array::from_fn(|c| cells.col(&cols[c]));
        // Top border: H'(0, j) - H'(0, j - 1) is the gap step of b[j].
        let mut ups: [__m256i; K] = std::array::from_fn(|c| ys[c].1);
        for (cell, codes) in v.iter_mut().zip(a) {
            let x = cells.row(codes);
            let mut left = *cell;
            for c in 0..K {
                (left, ups[c]) = cells.step(x, ys[c], left, ups[c]);
            }
            *cell = left;
        }
    }

    /// Per lane, the sum of `v[..m]` where `m` is the lane's `a` length, for the lanes in
    /// `ending` (other lanes' entries are unspecified).
    #[target_feature(enable = "avx2")]
    fn column_sums(v: &[__m256i], pairs: &[(&[u8], &[u8])], ending: u32) -> [i32; LANES] {
        let mut sums = [0i32; LANES];
        let mut lens = (0..pairs.len()).filter(|&l| ending & (1 << l) != 0).map(|l| pairs[l].0.len());
        let first = lens.next().unwrap_or(0);
        if lens.all(|m| m == first) {
            // One row count for every lane: widen and add whole rows.
            let mut acc = [_mm256_setzero_si256(); 4];
            for &d in &v[..first] {
                let halves = [_mm256_castsi256_si128(d), _mm256_extracti128_si256::<1>(d)];
                for (h, half) in halves.into_iter().enumerate() {
                    acc[2 * h] = _mm256_add_epi32(acc[2 * h], _mm256_cvtepi8_epi32(half));
                    acc[2 * h + 1] = _mm256_add_epi32(acc[2 * h + 1], _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(half)));
                }
            }
            for (k, a) in acc.into_iter().enumerate() {
                // SAFETY: __m256i and [i32; 8] are both 32 bytes of plain integers.
                sums[8 * k..8 * k + 8].copy_from_slice(&unsafe { std::mem::transmute::<__m256i, [i32; 8]>(a) });
            }
            return sums;
        }
        let mut rest = ending;
        while rest != 0 {
            let lane = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            sums[lane] = v[..pairs[lane].0.len()]
                .iter()
                .map(|&d| {
                    // SAFETY: __m256i and [i8; 32] are both 32 bytes of plain integers.
                    i32::from(unsafe { std::mem::transmute::<__m256i, [i8; LANES]>(d) }[lane])
                })
                .sum();
        }
        sums
    }

    /// Completes `out` (pre-filled with each lane's border terms) for every pair.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fill(scorer: &BatchScorer, pairs: &[(&[u8], &[u8])], buf: &mut BatchBuffers, out: &mut [i32]) {
        let table = |t: &[i8; 16]| -> __m256i {
            // SAFETY: 16 readable bytes, broadcast to both halves for pshufb.
            unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(t.as_ptr().cast())) }
        };
        let cells = Cells {
            bonus: table(&scorer.bonus),
            step: table(&scorer.step),
            mask: _mm256_set1_epi8(scorer.class_mask as i8),
        };
        let (a, b) = (&buf.a, &buf.b);
        // Left border: H'(i, 0) - H'(i - 1, 0) is the gap step of a[i].
        buf.v.clear();
        buf.v.extend(a.iter().map(|codes| cells.row(codes).2));
        let v = &mut buf.v[..];
        let mut j = 0;
        while j < b.len() {
            // Several columns per pass over the rows, so each row's codes and difference
            // are loaded once; the out-of-order core overlaps consecutive rows. A pass
            // ends at any column where a lane ends, so its differences are still in `v`
            // when that lane is summed.
            let span = (1..=COLUMNS).take_while(|&k| j + k <= b.len() && (k == 1 || buf.ends[j + k - 2] == 0)).last();
            match span.unwrap_or(1) {
                4 => pass::<4>(&cells, a, &b[j..j + 4], v),
                3 => pass::<3>(&cells, a, &b[j..j + 3], v),
                2 => pass::<2>(&cells, a, &b[j..j + 2], v),
                _ => pass::<1>(&cells, a, &b[j..j + 1], v),
            }
            j += span.unwrap_or(1);
            // H'(m, n) is H'(0, n) = g n - sum(other[b]) plus the vertical differences down
            // column n; adding back both sums of `other` gives g n + sum(other[a]) + down.
            let ending = buf.ends[j - 1];
            if ending != 0 {
                for (lane, down) in column_sums(v, pairs, ending).into_iter().enumerate() {
                    if ending & (1 << lane) != 0 {
                        out[lane] += down;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{BatchBuffers, BatchScorer, LANES};
    use crate::neon::{I8x32, neon};

    /// Lookup tables and the class mask, as in the AVX2 kernel.
    struct Cells {
        bonus: int8x16_t,
        step: int8x16_t,
        mask: I8x32,
    }

    /// A row's letter codes with their match and gap steps.
    type Row = (I8x32, I8x32, I8x32);
    /// A column's letter codes with their gap steps.
    type Col = (I8x32, I8x32);

    impl Cells {
        #[inline]
        fn row(&self, codes: &[u8; LANES]) -> Row {
            let x = I8x32::load(codes);
            let class = x.and(self.mask);
            (x, class.lookup(self.bonus), class.lookup(self.step))
        }

        #[inline]
        fn col(&self, codes: &[u8; LANES]) -> Col {
            let (y, _, step_y) = self.row(codes);
            (y, step_y)
        }

        /// One cell from the vertical difference to its left (`left`, previous column)
        /// and the horizontal difference above it (`up`, previous row). Returns the
        /// cell's vertical and horizontal differences.
        #[inline]
        fn step(&self, (x, bonus_x, step_x): Row, (y, step_y): Col, left: I8x32, up: I8x32) -> (I8x32, I8x32) {
            // A gap from the left skips b[j] and one from above skips a[i].
            let diag = x.eq(y).and(bonus_x);
            let gaps = left.add(step_y).max(up.add(step_x));
            let z = diag.max(gaps);
            (z.sub(up), z.sub(left))
        }
    }

    /// Columns per pass over the rows (at most 4).
    const COLUMNS: usize = 4;

    /// Advances `v` across the `K` columns `cols`.
    #[inline]
    fn pass<const K: usize>(cells: &Cells, a: &[[u8; LANES]], cols: &[[u8; LANES]], v: &mut [I8x32]) {
        let ys: [Col; K] = std::array::from_fn(|c| cells.col(&cols[c]));
        // Top border: H'(0, j) - H'(0, j - 1) is the gap step of b[j].
        let mut ups: [I8x32; K] = std::array::from_fn(|c| ys[c].1);
        for (cell, codes) in v.iter_mut().zip(a) {
            let x = cells.row(codes);
            let mut left = *cell;
            for c in 0..K {
                (left, ups[c]) = cells.step(x, ys[c], left, ups[c]);
            }
            *cell = left;
        }
    }

    /// Per lane, the sum of `v[..m]` where `m` is the lane's `a` length, for the lanes in
    /// `ending` (other lanes' entries are unspecified).
    fn column_sums(v: &[I8x32], pairs: &[(&[u8], &[u8])], ending: u32) -> [i32; LANES] {
        let mut sums = [0i32; LANES];
        let mut lens = (0..pairs.len()).filter(|&l| ending & (1 << l) != 0).map(|l| pairs[l].0.len());
        let first = lens.next().unwrap_or(0);
        if lens.all(|m| m == first) {
            // One row count for every lane: widen and add whole rows, four lanes per
            // accumulator in lane order.
            let mut acc = [neon!(vdupq_n_s32(0)); 8];
            for &d in &v[..first] {
                for (h, half) in d.halves().into_iter().enumerate() {
                    let acc = &mut acc[4 * h..4 * h + 4];
                    neon!({
                        let (lo, hi) = (vmovl_s8(vget_low_s8(half)), vmovl_high_s8(half));
                        acc[0] = vaddw_s16(acc[0], vget_low_s16(lo));
                        acc[1] = vaddw_high_s16(acc[1], lo);
                        acc[2] = vaddw_s16(acc[2], vget_low_s16(hi));
                        acc[3] = vaddw_high_s16(acc[3], hi);
                    });
                }
            }
            for (k, a) in acc.into_iter().enumerate() {
                // SAFETY: `sums[4k..4k + 4]` is 4 writable i32s; the store is unaligned.
                unsafe { vst1q_s32(sums[4 * k..4 * k + 4].as_mut_ptr(), a) };
            }
            return sums;
        }
        let mut rest = ending;
        while rest != 0 {
            let lane = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            sums[lane] = v[..pairs[lane].0.len()].iter().map(|&d| i32::from(d.to_array()[lane])).sum();
        }
        sums
    }

    /// Completes `out` (pre-filled with each lane's border terms) for every pair.
    pub(super) fn fill(scorer: &BatchScorer, pairs: &[(&[u8], &[u8])], buf: &mut BatchBuffers, out: &mut [i32]) {
        // SAFETY: 16 readable bytes.
        let table = |t: &[i8; 16]| unsafe { vld1q_s8(t.as_ptr()) };
        let cells = Cells {
            bonus: table(&scorer.bonus),
            step: table(&scorer.step),
            mask: I8x32::splat(scorer.class_mask as i8),
        };
        let (a, b) = (&buf.a, &buf.b);
        // Left border: H'(i, 0) - H'(i - 1, 0) is the gap step of a[i].
        buf.v.clear();
        buf.v.extend(a.iter().map(|codes| cells.row(codes).2));
        let v = &mut buf.v[..];
        let mut j = 0;
        while j < b.len() {
            // Several columns per pass over the rows; a pass ends at any column where a
            // lane ends, so its differences are still in `v` when that lane is summed.
            let span = (1..=COLUMNS).take_while(|&k| j + k <= b.len() && (k == 1 || buf.ends[j + k - 2] == 0)).last();
            match span.unwrap_or(1) {
                4 => pass::<4>(&cells, a, &b[j..j + 4], v),
                3 => pass::<3>(&cells, a, &b[j..j + 3], v),
                2 => pass::<2>(&cells, a, &b[j..j + 2], v),
                _ => pass::<1>(&cells, a, &b[j..j + 1], v),
            }
            j += span.unwrap_or(1);
            // As in the AVX2 kernel: g n + sum(other[a]) + the differences down column n.
            let ending = buf.ends[j - 1];
            if ending != 0 {
                for (lane, down) in column_sums(v, pairs, ending).into_iter().enumerate() {
                    if ending & (1 << lane) != 0 {
                        out[lane] += down;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::gotoh;
    use super::super::scoring::{GapKind, MatrixKind};
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    fn random_seq(seed: &mut u64, len: usize, letters: &[u8]) -> Vec<u8> {
        (0..len).map(|_| letters[lcg(seed) as usize % letters.len()]).collect()
    }

    fn check(scorer: &BatchScorer, scoring: &Scoring, pairs: &[(Vec<u8>, Vec<u8>)]) {
        let refs: Vec<(&[u8], &[u8])> = pairs.iter().map(|(a, b)| (a.as_slice(), b.as_slice())).collect();
        let mut out = [0; LANES];
        scorer.score(&refs, &mut BatchBuffers::default(), &mut out);
        for (lane, (a, b)) in pairs.iter().enumerate() {
            assert_eq!(out[lane], gotoh::score(a, b, scoring), "lane {lane}, lens {},{}", a.len(), b.len());
        }
    }

    #[test]
    #[ignore = "throughput against the striped kernel on 1000 x 1000 TVR alignments"]
    fn bench_batch_1000() {
        use super::super::striped::{self, ScoreBuffers};
        use rayon::prelude::*;
        let scoring = Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false);
        let Some(scorer) = BatchScorer::new(&scoring) else {
            assert!(!has_kernel(), "TVR scoring decomposes");
            return;
        };
        let mut seed = 1;
        let letters = b"CCCCCCCCCCCCCCCCCCCCCCDEFAAAGHC";
        let seqs: Vec<Vec<u8>> = (0..64).map(|_| random_seq(&mut seed, 1000, letters)).collect();
        let pairs: Vec<(&[u8], &[u8])> =
            seqs.iter().flat_map(|a| seqs.iter().map(move |b| (a.as_slice(), b.as_slice()))).collect();
        let cells = pairs.len() as f64 * 1e6;
        let striped_all = |pairs: &[(&[u8], &[u8])]| -> i64 {
            let mut buf = ScoreBuffers::default();
            pairs.iter().map(|&(a, b)| i64::from(striped::score(a, b, &scoring, &mut buf))).sum()
        };
        let batch_all = |pairs: &[(&[u8], &[u8])]| -> i64 {
            let mut buf = BatchBuffers::default();
            let mut out = [0; LANES];
            pairs
                .chunks(LANES)
                .map(|c| {
                    scorer.score(c, &mut buf, &mut out);
                    out[..c.len()].iter().map(|&s| i64::from(s)).sum::<i64>()
                })
                .sum()
        };
        for (name, run) in
            [("striped", &striped_all as &(dyn Fn(&[(&[u8], &[u8])]) -> i64 + Sync)), ("batch", &batch_all)]
        {
            let t = std::time::Instant::now();
            let total = run(&pairs);
            let single = cells / t.elapsed().as_secs_f64() / 1e9;
            let n = rayon::current_num_threads();
            let t = std::time::Instant::now();
            let all: i64 = (0..n).into_par_iter().map(|_| run(&pairs)).sum();
            let multi = n as f64 * cells / t.elapsed().as_secs_f64() / 1e9;
            eprintln!(
                "{name}: {single:.1} Gcells/s single, {multi:.1} Gcells/s on {n} threads (checksum {total} {all})"
            );
        }
    }

    #[test]
    fn accepts_the_distance_schemes_and_rejects_the_rest() {
        let tvr = Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false);
        let mm = Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, false);
        assert_eq!(BatchScorer::new(&tvr).is_some(), has_kernel());
        assert!(BatchScorer::new(&mm).is_none(), "256 byte letters do not fit 7-bit codes");
        assert!(
            BatchScorer::new(&Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, true)).is_none()
        );
        assert!(
            BatchScorer::new(&Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, false)).is_none()
        );
    }

    #[test]
    fn batches_match_the_reference() {
        let scoring = Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false);
        let Some(scorer) = BatchScorer::new(&scoring) else {
            assert!(!has_kernel(), "TVR scoring decomposes");
            return;
        };
        let mut seed = 11;
        for trial in 0..60 {
            let letters: &[u8] = if trial % 2 == 0 { b"CCCCCCCCCCCCCADEFGHC" } else { b"ACDEFGHIKLMNPQRSTVWY-" };
            let lanes = 1 + lcg(&mut seed) as usize % LANES;
            let max_len = if trial % 10 == 0 { 3000 } else { 200 };
            let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..lanes)
                .map(|_| {
                    let la = 1 + lcg(&mut seed) as usize % max_len;
                    let lb = 1 + lcg(&mut seed) as usize % max_len;
                    let a = random_seq(&mut seed, la, letters);
                    let b = if trial % 3 == 0 { a[..lb.min(la)].to_vec() } else { random_seq(&mut seed, lb, letters) };
                    (a, b)
                })
                .collect();
            check(&scorer, &scoring, &pairs);
        }
    }

    /// No length limit: 8-bit differences stay exact where 16-bit scores would not fit.
    #[test]
    fn long_pairs_and_extreme_scores_match_the_reference() {
        let scoring = Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false);
        let Some(scorer) = BatchScorer::new(&scoring) else {
            assert!(!has_kernel(), "TVR scoring decomposes");
            return;
        };
        let mut seed = 5;
        let pairs = vec![
            (vec![b'D'; 12000], vec![b'D'; 12000]),
            (vec![b'D'; 9000], vec![b'E'; 9500]),
            (vec![b'C'; 30], vec![b'D'; 11000]),
            (random_seq(&mut seed, 10000, b"CCCCCDEF"), random_seq(&mut seed, 1, b"CD")),
            (vec![b'A'], vec![b'A']),
        ];
        check(&scorer, &scoring, &pairs);
    }
}
