//! Global alignment with traceback in Farrar's striped SIMD layout (AVX2 or NEON), producing the
//! same first optimal alignment as [`super::gotoh::align`] (Biopython's).
//!
//! Vertical gaps are resolved per column with a scan across lanes (Daily's "striped
//! scan"), which gives the exact vertical-gap (Ix) value of every cell at a fixed cost.
//! Each cell then records three 2-bit fields:
//!
//! - bits 0-1: the best state of this cell (the predecessor of M one cell down-right),
//! - bits 2-3: the predecessor of this cell's Ix state (from the cell above),
//! - bits 4-5: the predecessor of the Iy state one cell to the right (from this cell).
//!
//! A field holds two flags, "the M candidate equals the maximum" (low bit) and "the Ix
//! candidate does" (high bit), which are cheaper to compute than the choice itself; the
//! traceback decodes them with Biopython's priority M > Ix > Iy ([`decode`]).

use super::gotoh;
use super::scoring::Scoring;
use super::striped::{ScoreBuffers, score_range};

/// Reusable per-thread buffers for [`align`].
#[derive(Debug, Default)]
pub struct TraceBuffers {
    enc: ScoreBuffers,
    #[cfg(target_arch = "x86_64")]
    simd: avx2::Buffers,
    #[cfg(target_arch = "aarch64")]
    simd: neon::Buffers,
    /// Trace bytes, starting at `trace_start` (32-byte aligned for streaming stores).
    trace: Vec<u8>,
    trace_start: usize,
}

/// `str(alignment[0]), str(alignment[1])` for Biopython's first optimal alignment.
///
/// # Panics
///
/// Panics if either sequence is empty.
pub fn align(a: &[u8], b: &[u8], scoring: &Scoring, buf: &mut TraceBuffers) -> (Vec<u8>, Vec<u8>) {
    assert!(!a.is_empty() && !b.is_empty(), "cannot align an empty sequence");
    #[cfg(target_arch = "aarch64")]
    {
        buf.enc.encode(a, b, &scoring.sub);
        let frame = frame_i16(a.len(), b.len(), scoring, &buf.enc.table);
        let lanes = if neon::fill_i16(buf, scoring, frame) {
            16
        } else {
            neon::fill_i32(buf, scoring);
            8
        };
        return traceback(a, b, &buf.trace[buf.trace_start..], a.len().div_ceil(lanes), lanes);
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        buf.enc.encode(a, b, &scoring.sub);
        let frame = frame_i16(a.len(), b.len(), scoring, &buf.enc.table);
        // SAFETY: AVX2 support was checked at runtime just above.
        let done = unsafe { avx2::fill_i16(buf, scoring, frame) };
        let lanes = if done {
            16
        } else {
            // SAFETY: as above.
            unsafe { avx2::fill_i32(buf, scoring) };
            8
        };
        return traceback(a, b, &buf.trace[buf.trace_start..], a.len().div_ceil(lanes), lanes);
    }
    #[allow(unreachable_code, reason = "aarch64 always returns above")]
    gotoh::align(a, b, scoring, &mut gotoh::AlignBuffers::default())
}

/// Columns between per-lane rebases in [`Frame::PerLane`].
const REBASE_COLUMNS: usize = 64;

/// How 16-bit lanes store scores: true score = stored + the lane's offset.
#[derive(Clone, Copy, Debug)]
enum Frame {
    /// One offset, `-bias`, for every lane.
    Centred(i32),
    /// Each lane's offset follows the score of its first row, rebased every
    /// [`REBASE_COLUMNS`] columns, so only the spread within a lane has to fit.
    PerLane,
}

/// Picks the 16-bit frame for an `m` x `n` alignment: one centred offset when the whole
/// score range fits, otherwise per-lane offsets. The per-lane fill checks its range and
/// reports failure (early, once lane offsets drift too far apart); worst-case bounds
/// reject far more alignments than actually overflow, so it is always tried.
fn frame_i16(m: usize, n: usize, scoring: &Scoring, table: &[i32]) -> Frame {
    // The padding rows below the last row drift by at most one gap step each.
    let (lo, hi) = score_range(m, n, scoring, table);
    let lo = lo + 16 * scoring.open.min(scoring.extend).min(0);
    if hi - lo < 58_000 { Frame::Centred(-(lo + hi) / 2) } else { Frame::PerLane }
}

const M: u8 = 0;
const IX: u8 = 1;
const IY: u8 = 2;

/// The first of (M, Ix, Iy) that reaches the maximum, from a field's two flags (M ties,
/// Ix ties); Iy when neither does.
fn decode(field: u8) -> u8 {
    const BY_FLAGS: [u8; 4] = [IY, M, IX, M];
    BY_FLAGS[usize::from(field & 3)]
}

/// Walks the striped trace from (len a, len b) back to the origin.
fn traceback(a: &[u8], b: &[u8], trace: &[u8], segs: usize, lanes: usize) -> (Vec<u8>, Vec<u8>) {
    let col_len = segs * lanes;
    let cell = |i: usize, j: usize| {
        let r = i - 1;
        trace[(j - 1) * col_len + (r % segs) * lanes + r / segs]
    };
    let best_state = |i: usize, j: usize| match (i, j) {
        (0, 0) => M,
        (0, _) => IY,
        (_, 0) => IX,
        _ => decode(cell(i, j)),
    };
    let (mut i, mut j) = (a.len(), b.len());
    let mut state = best_state(i, j);
    let mut row_a = Vec::with_capacity(a.len() + b.len());
    let mut row_b = Vec::with_capacity(a.len() + b.len());
    while i > 0 || j > 0 {
        match state {
            M => {
                row_a.push(a[i - 1]);
                row_b.push(b[j - 1]);
                i -= 1;
                j -= 1;
                state = best_state(i, j);
            }
            IX => {
                state = match j {
                    0 if i == 1 => M,
                    0 => IX,
                    _ => decode(cell(i, j) >> 2),
                };
                row_a.push(a[i - 1]);
                row_b.push(b'-');
                i -= 1;
            }
            _ => {
                state = match (i, j) {
                    (0, 1) => M,
                    (0, _) => IY,
                    (_, 1) => IX,
                    _ => decode(cell(i, j - 1) >> 4),
                };
                row_a.push(b'-');
                row_b.push(b[j - 1]);
                j -= 1;
            }
        }
    }
    row_a.reverse();
    row_b.reverse();
    (row_a, row_b)
}

/// The last row and column of a free-trailing-gap alignment, in exact scalar
/// arithmetic, writing the same trace bits as the vector fill would.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct ScalarEnds {
    m: usize,
    n: usize,
    segs: usize,
    lanes: usize,
    col_bytes: usize,
    end_open: i32,
    end_ext: i32,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl ScalarEnds {
    fn cell(&self, row: usize, col: usize) -> usize {
        (col - 1) * self.col_bytes + (row - 1) % self.segs * self.lanes + (row - 1) / self.segs
    }

    /// A trace field's flags: whether the M and the Ix candidates equal `target`.
    fn flags(m: i32, ix: i32, target: i32) -> u8 {
        u8::from(m == target) | (u8::from(ix == target) << 1)
    }

    /// `last_row`: (M, Ix) of row m in columns 1..n; `h`, `iy`: H and the Iy entering
    /// column n, by row (row m's Iy is recomputed here).
    fn fill(
        &self,
        enc: &ScoreBuffers,
        scoring: &Scoring,
        last_row: &[(i32, i32)],
        h: &[i32],
        iy: &mut [i32],
        trace: &mut [u8],
    ) {
        let (m, n, k) = (self.m, self.n, enc.alphabet);
        let sub = |row: usize, col: usize| enc.table[enc.codes_a[row - 1] as usize * k + enc.codes_b[col - 1] as usize];
        // Row m, columns 1..n: Iy extends along the row at the trailing-gap cost.
        let mut iy_m = scoring.left_gap(m) + self.end_open;
        for (col, &(vm, ix)) in (1..).zip(last_row) {
            let hm = vm.max(ix).max(iy_m);
            let iy_next = (iy_m + self.end_ext).max(hm + self.end_open);
            let byte = &mut trace[self.cell(m, col)];
            let next = Self::flags(vm + self.end_open, ix + self.end_open, iy_next);
            *byte = (*byte & 0b1100) | Self::flags(vm, ix, hm) | (next << 4);
            iy_m = iy_next;
        }
        iy[m] = iy_m;
        // Column n, top to bottom: Ix extends down the column at the trailing-gap cost.
        let mut diag = if n == 1 { 0 } else { scoring.left_gap(n - 1) };
        let mut ix = scoring.left_gap(n) + self.end_open;
        let (mut up_m, mut up_ix) = (i32::MIN / 4, i32::MIN / 4);
        for row in 1..=m {
            if row > 1 {
                ix = (ix + self.end_ext).max(up_m.max(iy[row - 1]) + self.end_open);
            }
            let vm = diag + sub(row, n);
            let hc = vm.max(iy[row]).max(ix);
            // Row 1's Ix comes from the top border, decoded from no flags.
            let from = if row == 1 { 0 } else { Self::flags(up_m + self.end_open, up_ix + self.end_ext, ix) };
            let (eo, ee) = if row == m { (self.end_open, self.end_ext) } else { (scoring.open, scoring.extend) };
            let iy_next = (iy[row] + ee).max(hc + eo);
            let next = Self::flags(vm + eo, ix + eo, iy_next);
            trace[self.cell(row, n)] = Self::flags(vm, ix, hc) | (from << 2) | (next << 4);
            (up_m, up_ix, diag) = (vm, ix, h[row]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{Frame, REBASE_COLUMNS, ScalarEnds, Scoring, TraceBuffers};

    #[derive(Debug, Default)]
    pub(super) struct Buffers {
        /// Substitution scores per letter of `b`, as bytes in striped order.
        profile: Vec<i8>,
        slot: Vec<usize>,
        h_prev: Vec<__m256i>,
        h_cur: Vec<__m256i>,
        e: Vec<__m256i>,
        f_local: Vec<__m256i>,
        /// M of each segment in the current column, from pass 1 for pass 2.
        m_local: Vec<__m256i>,
        /// (M, Ix) of the last row in each column, for [`Ends::Scalar`].
        last_row: Vec<(i32, i32)>,
        /// H and Iy of every row in the second-to-last column, likewise.
        last_h: Vec<i32>,
        last_iy: Vec<i32>,
    }

    /// Narrows 16 lanes of small non-negative i16 values to bytes and streams them to
    /// `out` (16 bytes), bypassing the cache.
    #[target_feature(enable = "avx2")]
    fn store_bytes_i16(v: __m256i, out: &mut [u8]) {
        let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(v, v));
        let out = &mut out[..16];
        assert!((out.as_ptr() as usize).is_multiple_of(16), "trace rows are 16-byte aligned");
        // SAFETY: `out` is 16 writable bytes at a 16-byte aligned address.
        unsafe { _mm_stream_si128(out.as_mut_ptr().cast(), _mm256_castsi256_si128(packed)) };
    }

    /// Narrows 8 lanes of small non-negative i32 values to bytes and streams them to `out`
    /// (8 bytes), bypassing the cache.
    #[target_feature(enable = "avx2")]
    fn store_bytes_i32(v: __m256i, out: &mut [u8]) {
        let w16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi32(v, v));
        let w8 = _mm256_packus_epi16(w16, w16);
        let out = &mut out[..8];
        // SAFETY: `out` is 8 writable bytes; MOVNTI has no alignment requirement.
        unsafe { _mm_stream_si64(out.as_mut_ptr().cast(), _mm_cvtsi128_si64(_mm256_castsi256_si128(w8))) };
    }

    /// Loads 16 profile bytes as i16 lanes.
    #[target_feature(enable = "avx2")]
    fn load_profile_i16(p: &[i8]) -> __m256i {
        let p = &p[..16];
        // SAFETY: `p` is 16 readable bytes; the load is unaligned.
        _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(p.as_ptr().cast()) })
    }

    /// Loads 8 profile bytes as i32 lanes.
    #[target_feature(enable = "avx2")]
    fn load_profile_i32(p: &[i8]) -> __m256i {
        let p = &p[..8];
        // SAFETY: `p` is 8 readable bytes; the load is unaligned.
        _mm256_cvtepi8_epi32(unsafe { _mm_loadl_epi64(p.as_ptr().cast()) })
    }

    macro_rules! trace_kernel {
        ($name:ident, $elem:ty, $lanes:expr, $neg:expr, $limit:expr, $add:ident, $sub:ident, $max:ident, $min:ident,
         $set1:ident, $cmpeq:ident, $slli:ident, $store_bytes:ident, $load_profile:ident, $shift_bytes:expr,
         $insert:ident) => {
            /// Fills the trace. Returns `false` if scores left the range where this lane
            /// width is exact.
            ///
            /// # Safety
            ///
            /// The CPU must support AVX2.
            #[target_feature(enable = "avx2")]
            /// Scores are stored relative to per-lane offsets set by `frame` (comparisons and
            /// the trace do not depend on them). With `TRACK` false the range is not checked,
            /// for frames whose bounds already guarantee it.
            pub(super) unsafe fn $name<const TRACK: bool>(
                buf: &mut TraceBuffers,
                scoring: &Scoring,
                frame: Frame,
            ) -> bool {
                const L: usize = $lanes;
                const NEG: $elem = $neg;
                let enc = &buf.enc;
                let (m, n) = (enc.codes_a.len(), enc.codes_b.len());
                let segs = m.div_ceil(L);
                let k = enc.alphabet;
                let clamp = |v: i32| v.clamp(NEG as i32, <$elem>::MAX as i32) as $elem;
                let gap = |len: usize| scoring.left_gap(len);
                // true score = stored + off[lane]
                let mut off: [i32; L] = match frame {
                    Frame::Centred(bias) => [-bias; L],
                    Frame::PerLane => std::array::from_fn(|lane| gap(lane * segs + 1)),
                };
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
                // Added after shifting a vector down one lane, to move each value into the
                // frame of the lane it lands in. Lane 0 receives the inserted value as is.
                let lane_shift = |off: &[i32; L]| -> ([i32; L], i32) {
                    let d: [i32; L] = std::array::from_fn(|l| if l == 0 { 0 } else { off[l - 1] - off[l] });
                    (d, d.iter().map(|x| x.abs()).max().unwrap_or(0))
                };
                let (mut shift, mut max_shift) = lane_shift(&off);
                if max_shift >= $limit / 2 {
                    return false;
                }
                let mut v_shift = to_vec(shift.map(|x| x as $elem));
                // A trace field's two flags (M ties, Ix ties), already at the field's bits.
                let field_bits = |shift: u32| ($set1(1 << shift), $set1(2 << shift));
                let (best_bits, ix_bits) = (field_bits(0), field_bits(2));
                let flags = |(m_bit, ix_bit): (__m256i, __m256i), eq_m: __m256i, eq_f: __m256i| {
                    _mm256_or_si256(_mm256_and_si256(eq_m, m_bit), _mm256_and_si256(eq_f, ix_bit))
                };
                let v_zero = $set1(0);
                // Free trailing gaps let the last row and column climb to their running
                // maximum, far from their lanes' offsets in the per-lane frame. There the
                // fill uses ordinary gaps (nothing else depends on those cells) and the last
                // row and column are redone exactly in scalar code.
                let scalar_ends = matches!(frame, Frame::PerLane) && scoring.free_right_gaps;
                let (end_open, end_ext) = if scoring.free_right_gaps { (0, 0) } else { (scoring.open, scoring.extend) };
                let (right_open, right_ext) =
                    if scalar_ends { (scoring.open, scoring.extend) } else { (end_open, end_ext) };
                let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
                let (v_open, v_ext) = ($set1(scoring.open as $elem), $set1(scoring.extend as $elem));
                // Horizontal (Iy) gap scores for the segment holding the last row, whose
                // trailing gaps may be free.
                let (last_open, last_ext) = {
                    let (mut o, mut e) = ([scoring.open as $elem; L], [scoring.extend as $elem; L]);
                    o[last_lane] = right_open as $elem;
                    e[last_lane] = right_ext as $elem;
                    (to_vec(o), to_vec(e))
                };

                let s = &mut buf.simd;
                s.slot.clear();
                s.slot.resize(k, usize::MAX);
                s.profile.clear();
                for &c in &enc.codes_b {
                    let c = c as usize;
                    if s.slot[c] != usize::MAX {
                        continue;
                    }
                    s.slot[c] = s.profile.len() / (segs * L);
                    for t in 0..segs {
                        for lane in 0..L {
                            let row = lane * segs + t;
                            let v = if row < m { enc.table[enc.codes_a[row] as usize * k + c] } else { 0 };
                            s.profile.push(i8::try_from(v).expect("substitution scores fit in a byte"));
                        }
                    }
                }
                // Column 0: H = Ix = leading vertical gap; the first Iy opens from it.
                let rows = |t: usize| -> [usize; L] { std::array::from_fn(|lane| lane * segs + t + 1) };
                s.h_prev.clear();
                s.e.clear();
                for t in 0..segs {
                    let r = rows(t);
                    s.h_prev.push(to_vec(std::array::from_fn(|lane| clamp(gap(r[lane]) - off[lane]))));
                    s.e.push(to_vec(std::array::from_fn(|lane| {
                        let open = if t == last_seg && lane == last_lane { right_open } else { scoring.open };
                        clamp(gap(r[lane]) + open - off[lane])
                    })));
                }
                s.h_cur.clear();
                s.h_cur.resize(segs, $set1(NEG));
                s.f_local.clear();
                s.f_local.resize(segs, $set1(NEG));
                s.m_local.clear();
                s.m_local.resize(segs, $set1(NEG));
                // every byte is overwritten below, so only grow the buffer
                let col_bytes = segs * L;
                if buf.trace.len() < n * col_bytes + 32 {
                    buf.trace.resize(n * col_bytes + 32, 0);
                }
                buf.trace_start = buf.trace.as_ptr().align_offset(32);
                let trace = &mut buf.trace[buf.trace_start..];

                let mut v_min = $set1(0);
                let mut v_max = $set1(0);
                s.last_row.clear();
                for (j, &c) in enc.codes_b.iter().enumerate() {
                    if scalar_ends && j == n - 1 {
                        break;
                    }
                    if matches!(frame, Frame::PerLane) && j > 0 && j % REBASE_COLUMNS == 0 {
                        // Each lane's first row becomes its new zero.
                        let d = to_lanes(s.h_prev[0]);
                        let v_d = to_vec(d);
                        for t in 0..segs {
                            s.h_prev[t] = $sub(s.h_prev[t], v_d);
                            s.e[t] = $sub(s.e[t], v_d);
                        }
                        for (o, &x) in off.iter_mut().zip(&d) {
                            *o += x as i32;
                        }
                        let (d, m) = lane_shift(&off);
                        (shift, max_shift) = (d, max_shift.max(m));
                        if max_shift >= $limit / 2 {
                            return false;
                        }
                        v_shift = to_vec(shift.map(|x| x as $elem));
                    }
                    let col = j + 1;
                    let (f_open_s, f_ext_s) =
                        if col == n { (right_open, right_ext) } else { (scoring.open, scoring.extend) };
                    let (f_open, f_ext) = ($set1(f_open_s as $elem), $set1(f_ext_s as $elem));
                    let prof = &s.profile[s.slot[c as usize] * segs * L..][..segs * L];
                    let top_prev = clamp(if col == 1 { 0 } else { gap(col - 1) } - off[0]);

                    // Pass 1: each lane's own vertical-gap chain, from M and Iy only
                    // (F + open never beats F + extend, so H + open = max(M, Iy) + open).
                    let mut v_f = $insert::<0>($set1(NEG), clamp(gap(col) + f_open_s - off[0]));
                    let mut v_h = $add(shift_in(s.h_prev[segs - 1], top_prev), v_shift);
                    for t in 0..segs {
                        let vm = $add(v_h, $load_profile(&prof[t * L..]));
                        s.m_local[t] = vm;
                        let ht = $max(vm, s.e[t]);
                        s.f_local[t] = v_f;
                        v_f = $max($add(v_f, f_ext), $add(ht, f_open));
                        v_h = s.h_prev[t];
                    }
                    // Scan across lanes: the Ix value entering each lane from the one above.
                    let exits = to_lanes(v_f);
                    let lane_decay = f_ext_s * segs as i32;
                    let mut entering = [NEG; L];
                    // in the frame of the lane being entered; starts as minus infinity
                    let mut carried = i32::MIN / 4;
                    for lane in 1..L {
                        carried = (exits[lane - 1] as i32).max(carried.saturating_add(lane_decay)) + shift[lane];
                        entering[lane] = clamp(carried);
                    }

                    // Pass 2: exact states, H, the next column's Iy, and the trace.
                    let out = &mut trace[j * col_bytes..][..col_bytes];
                    let mut v_in = to_vec(entering);
                    let (mut up_m, mut up_f) = ($set1(NEG), $set1(NEG));
                    let (mut first_f, mut first_packed) = ($set1(NEG), v_zero);
                    for t in 0..segs {
                        let vm = s.m_local[t];
                        let ve = s.e[t];
                        let f = $max(s.f_local[t], v_in);
                        let h = $max($max(vm, ve), f);
                        s.h_cur[t] = h;
                        if TRACK {
                            v_min = $min(v_min, h);
                            v_max = $max(v_max, h);
                        }
                        let (eo, ee) = if t == last_seg { (last_open, last_ext) } else { (v_open, v_ext) };
                        let h_open = $add(h, eo);
                        let e_next = $max($add(ve, ee), h_open);
                        s.e[t] = e_next;
                        let best = flags(best_bits, $cmpeq(vm, h), $cmpeq(f, h));
                        // The next Iy opens from M (or Ix) exactly when opening wins and M (or
                        // Ix) is this cell's maximum: the best flags, moved up, masked.
                        let iy_next = _mm256_and_si256($slli::<4>(best), $cmpeq(h_open, e_next));
                        if scalar_ends && t == last_seg {
                            let at = |v: __m256i| to_lanes(v)[last_lane] as i32 + off[last_lane];
                            s.last_row.push((at(vm), at(f)));
                        }
                        if t == 0 {
                            // its Ix predecessors are only known after the last segment
                            first_f = f;
                            first_packed = _mm256_or_si256(best, iy_next);
                        } else {
                            let ix = flags(ix_bits, $cmpeq($add(up_m, f_open), f), $cmpeq($add(up_f, f_ext), f));
                            let packed = _mm256_or_si256(best, _mm256_or_si256(ix, iy_next));
                            $store_bytes(packed, &mut out[t * L..]);
                        }
                        (up_m, up_f) = (vm, f);
                        v_in = $add(v_in, f_ext);
                    }
                    // Ix predecessors of the first segment come from the last segment of the
                    // lane above (or the empty row 0 for the first lane).
                    let ix0 = flags(
                        ix_bits,
                        $cmpeq($add($add(shift_in(up_m, NEG), v_shift), f_open), first_f),
                        $cmpeq($add($add(shift_in(up_f, NEG), v_shift), f_ext), first_f),
                    );
                    $store_bytes(_mm256_or_si256(first_packed, ix0), out);
                    std::mem::swap(&mut s.h_prev, &mut s.h_cur);
                }
                // make the streamed trace visible before it is read back
                _mm_sfence();
                if scalar_ends {
                    let ends = ScalarEnds { m, n, segs, lanes: L, col_bytes, end_open, end_ext };
                    s.last_h.clear();
                    s.last_h.resize(m + 1, 0);
                    s.last_iy.clear();
                    s.last_iy.resize(m + 1, 0);
                    for t in 0..segs {
                        let (h, e) = (to_lanes(s.h_prev[t]), to_lanes(s.e[t]));
                        for lane in 0..L {
                            let row = lane * segs + t + 1;
                            if row <= m {
                                s.last_h[row] = h[lane] as i32 + off[lane];
                                s.last_iy[row] = e[lane] as i32 + off[lane];
                            }
                        }
                    }
                    ends.fill(enc, scoring, &s.last_row, &s.last_h, &mut s.last_iy, trace);
                }
                // Every stored score must stay exact after one lane shift.
                let limit = $limit - max_shift;
                let (lo, hi) = (to_lanes(v_min), to_lanes(v_max));
                !(lo.iter().any(|&v| (v as i32) < -limit) || hi.iter().any(|&v| (v as i32) > limit))
            }
        };
    }

    trace_kernel!(
        fill_i16_inner,
        i16,
        16,
        i16::MIN,
        30_000,
        _mm256_adds_epi16,
        _mm256_subs_epi16,
        _mm256_max_epi16,
        _mm256_min_epi16,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        _mm256_slli_epi16,
        store_bytes_i16,
        load_profile_i16,
        2,
        _mm256_insert_epi16
    );
    trace_kernel!(
        fill_i32_inner,
        i32,
        8,
        i32::MIN / 4,
        i32::MAX / 8,
        _mm256_add_epi32,
        _mm256_sub_epi32,
        _mm256_max_epi32,
        _mm256_min_epi32,
        _mm256_set1_epi32,
        _mm256_cmpeq_epi32,
        _mm256_slli_epi32,
        store_bytes_i32,
        load_profile_i32,
        4,
        _mm256_insert_epi32
    );

    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fill_i16(buf: &mut TraceBuffers, scoring: &Scoring, frame: Frame) -> bool {
        // SAFETY: the caller guarantees AVX2.
        unsafe {
            match frame {
                // `frame_i16` only centres scores whose whole range fits.
                Frame::Centred(_) => fill_i16_inner::<false>(buf, scoring, frame),
                Frame::PerLane => fill_i16_inner::<true>(buf, scoring, frame),
            }
        }
    }

    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fill_i32(buf: &mut TraceBuffers, scoring: &Scoring) {
        // SAFETY: the caller guarantees AVX2.
        let exact = unsafe { fill_i32_inner::<true>(buf, scoring, Frame::Centred(0)) };
        assert!(exact, "32-bit alignment scores overflowed");
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! The AVX2 kernel's algorithm, generic over the lane width rather than expanded from a
    //! macro. Trace bytes use ordinary stores (NEON has no streaming store intrinsic).

    use super::{Frame, REBASE_COLUMNS, ScalarEnds, Scoring, TraceBuffers};
    use crate::neon::{I16x16, I32x8, Lanes};

    /// The buffers of one lane width.
    #[derive(Debug)]
    pub(super) struct Vecs<V> {
        /// Substitution scores per letter of `b`, as bytes in striped order.
        profile: Vec<i8>,
        slot: Vec<usize>,
        h_prev: Vec<V>,
        h_cur: Vec<V>,
        e: Vec<V>,
        f_local: Vec<V>,
        /// M of each segment in the current column, from pass 1 for pass 2.
        m_local: Vec<V>,
        /// (M, Ix) of the last row in each column, for [`ScalarEnds`].
        last_row: Vec<(i32, i32)>,
        /// H and Iy of every row in the second-to-last column, likewise.
        last_h: Vec<i32>,
        last_iy: Vec<i32>,
    }

    impl<V> Default for Vecs<V> {
        fn default() -> Self {
            Self {
                profile: Vec::new(),
                slot: Vec::new(),
                h_prev: Vec::new(),
                h_cur: Vec::new(),
                e: Vec::new(),
                f_local: Vec::new(),
                m_local: Vec::new(),
                last_row: Vec::new(),
                last_h: Vec::new(),
                last_iy: Vec::new(),
            }
        }
    }

    #[derive(Debug, Default)]
    pub(super) struct Buffers {
        w16: Vecs<I16x16>,
        w32: Vecs<I32x8>,
    }

    /// A lane width, with its buffers.
    trait Width<const L: usize>: Lanes<L> {
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

    /// Fills the trace. Returns `false` if scores left the range where this lane width is
    /// exact (`limit`, less the largest lane shift).
    ///
    /// Scores are stored relative to per-lane offsets set by `frame` (comparisons and the
    /// trace do not depend on them). With `TRACK` false the range is not checked, for
    /// frames whose bounds already guarantee it.
    fn fill<const L: usize, V: Width<L>, const TRACK: bool>(
        buf: &mut TraceBuffers,
        scoring: &Scoring,
        frame: Frame,
        limit: i32,
    ) -> bool {
        let enc = &buf.enc;
        let (m, n) = (enc.codes_a.len(), enc.codes_b.len());
        let segs = m.div_ceil(L);
        let k = enc.alphabet;
        let clamp = |v: i32| V::cast(v.clamp(V::NEG.into(), V::MAX.into()));
        let splat = |v: i32| V::splat(V::cast(v));
        let gap = |len: usize| scoring.left_gap(len);
        // true score = stored + off[lane]
        let mut off: [i32; L] = match frame {
            Frame::Centred(bias) => [-bias; L],
            Frame::PerLane => std::array::from_fn(|lane| gap(lane * segs + 1)),
        };
        // Added after shifting a vector down one lane, to move each value into the frame
        // of the lane it lands in. Lane 0 receives the inserted value as is.
        let lane_shift = |off: &[i32; L]| -> ([i32; L], i32) {
            let d: [i32; L] = std::array::from_fn(|l| if l == 0 { 0 } else { off[l - 1] - off[l] });
            (d, d.iter().map(|x| x.abs()).max().unwrap_or(0))
        };
        let (mut shift, mut max_shift) = lane_shift(&off);
        if max_shift >= limit / 2 {
            return false;
        }
        let mut v_shift = V::from_array(shift.map(V::cast));
        // A trace field's two flags (M ties, Ix ties), already at the field's bits.
        let field_bits = |shift: u32| (splat(1 << shift), splat(2 << shift));
        let (best_bits, ix_bits) = (field_bits(0), field_bits(2));
        let flags = |(m_bit, ix_bit): (V, V), eq_m: V, eq_f: V| eq_m.and(m_bit).or(eq_f.and(ix_bit));
        // Free trailing gaps let the last row and column climb to their running maximum,
        // far from their lanes' offsets in the per-lane frame. There the fill uses ordinary
        // gaps (nothing else depends on those cells) and the last row and column are redone
        // exactly in scalar code.
        let scalar_ends = matches!(frame, Frame::PerLane) && scoring.free_right_gaps;
        let (end_open, end_ext) = if scoring.free_right_gaps { (0, 0) } else { (scoring.open, scoring.extend) };
        let (right_open, right_ext) = if scalar_ends { (scoring.open, scoring.extend) } else { (end_open, end_ext) };
        let (last_seg, last_lane) = ((m - 1) % segs, (m - 1) / segs);
        let (v_open, v_ext) = (splat(scoring.open), splat(scoring.extend));
        // Horizontal (Iy) gap scores for the segment holding the last row, whose trailing
        // gaps may be free.
        let (last_open, last_ext) = {
            let (mut o, mut e) = ([V::cast(scoring.open); L], [V::cast(scoring.extend); L]);
            o[last_lane] = V::cast(right_open);
            e[last_lane] = V::cast(right_ext);
            (V::from_array(o), V::from_array(e))
        };

        let s = V::vecs(&mut buf.simd);
        s.slot.clear();
        s.slot.resize(k, usize::MAX);
        s.profile.clear();
        for &c in &enc.codes_b {
            let c = c as usize;
            if s.slot[c] != usize::MAX {
                continue;
            }
            s.slot[c] = s.profile.len() / (segs * L);
            for t in 0..segs {
                for lane in 0..L {
                    let row = lane * segs + t;
                    let v = if row < m { enc.table[enc.codes_a[row] as usize * k + c] } else { 0 };
                    s.profile.push(i8::try_from(v).expect("substitution scores fit in a byte"));
                }
            }
        }
        // Column 0: H = Ix = leading vertical gap; the first Iy opens from it.
        let rows = |t: usize| -> [usize; L] { std::array::from_fn(|lane| lane * segs + t + 1) };
        s.h_prev.clear();
        s.e.clear();
        for t in 0..segs {
            let r = rows(t);
            s.h_prev.push(V::from_array(std::array::from_fn(|lane| clamp(gap(r[lane]) - off[lane]))));
            s.e.push(V::from_array(std::array::from_fn(|lane| {
                let open = if t == last_seg && lane == last_lane { right_open } else { scoring.open };
                clamp(gap(r[lane]) + open - off[lane])
            })));
        }
        let neg = V::splat(V::NEG);
        s.h_cur.clear();
        s.h_cur.resize(segs, neg);
        s.f_local.clear();
        s.f_local.resize(segs, neg);
        s.m_local.clear();
        s.m_local.resize(segs, neg);
        // every byte is overwritten below, so only grow the buffer
        let col_bytes = segs * L;
        if buf.trace.len() < n * col_bytes + 32 {
            buf.trace.resize(n * col_bytes + 32, 0);
        }
        buf.trace_start = buf.trace.as_ptr().align_offset(32);
        let trace = &mut buf.trace[buf.trace_start..];

        let mut v_min = splat(0);
        let mut v_max = splat(0);
        s.last_row.clear();
        for (j, &c) in enc.codes_b.iter().enumerate() {
            if scalar_ends && j == n - 1 {
                break;
            }
            if matches!(frame, Frame::PerLane) && j > 0 && j % REBASE_COLUMNS == 0 {
                // Each lane's first row becomes its new zero.
                let v_d = s.h_prev[0];
                for t in 0..segs {
                    s.h_prev[t] = s.h_prev[t].sub(v_d);
                    s.e[t] = s.e[t].sub(v_d);
                }
                for (o, x) in off.iter_mut().zip(v_d.to_array()) {
                    *o += x.into();
                }
                let (d, m) = lane_shift(&off);
                (shift, max_shift) = (d, max_shift.max(m));
                if max_shift >= limit / 2 {
                    return false;
                }
                v_shift = V::from_array(shift.map(V::cast));
            }
            let col = j + 1;
            let (f_open_s, f_ext_s) = if col == n { (right_open, right_ext) } else { (scoring.open, scoring.extend) };
            let (f_open, f_ext) = (splat(f_open_s), splat(f_ext_s));
            let prof = &s.profile[s.slot[c as usize] * segs * L..][..segs * L];
            let top_prev = clamp(if col == 1 { 0 } else { gap(col - 1) } - off[0]);

            // Pass 1: each lane's own vertical-gap chain, from M and Iy only (F + open
            // never beats F + extend, so H + open = max(M, Iy) + open).
            let mut v_f = neg.with_first(clamp(gap(col) + f_open_s - off[0]));
            let mut v_h = s.h_prev[segs - 1].shift_in(top_prev).add(v_shift);
            for t in 0..segs {
                let vm = v_h.add(V::widen(&prof[t * L..]));
                s.m_local[t] = vm;
                let ht = vm.max(s.e[t]);
                s.f_local[t] = v_f;
                v_f = v_f.add(f_ext).max(ht.add(f_open));
                v_h = s.h_prev[t];
            }
            // Scan across lanes: the Ix value entering each lane from the one above.
            let exits = v_f.to_array();
            let lane_decay = f_ext_s * segs as i32;
            let mut entering = [V::NEG; L];
            // in the frame of the lane being entered; starts as minus infinity
            let mut carried = i32::MIN / 4;
            for lane in 1..L {
                carried = exits[lane - 1].into().max(carried.saturating_add(lane_decay)) + shift[lane];
                entering[lane] = clamp(carried);
            }

            // Pass 2: exact states, H, the next column's Iy, and the trace.
            let out = &mut trace[j * col_bytes..][..col_bytes];
            let mut v_in = V::from_array(entering);
            let (mut up_m, mut up_f) = (neg, neg);
            let (mut first_f, mut first_packed) = (neg, splat(0));
            for t in 0..segs {
                let vm = s.m_local[t];
                let ve = s.e[t];
                let f = s.f_local[t].max(v_in);
                let h = vm.max(ve).max(f);
                s.h_cur[t] = h;
                if TRACK {
                    v_min = v_min.min(h);
                    v_max = v_max.max(h);
                }
                let (eo, ee) = if t == last_seg { (last_open, last_ext) } else { (v_open, v_ext) };
                let h_open = h.add(eo);
                let e_next = ve.add(ee).max(h_open);
                s.e[t] = e_next;
                let best = flags(best_bits, vm.eq(h), f.eq(h));
                // The next Iy opens from M (or Ix) exactly when opening wins and M (or Ix)
                // is this cell's maximum: the best flags, moved up, masked.
                let iy_next = best.shl4().and(h_open.eq(e_next));
                if scalar_ends && t == last_seg {
                    let at = |v: V| v.to_array()[last_lane].into() + off[last_lane];
                    s.last_row.push((at(vm), at(f)));
                }
                if t == 0 {
                    // its Ix predecessors are only known after the last segment
                    first_f = f;
                    first_packed = best.or(iy_next);
                } else {
                    let ix = flags(ix_bits, up_m.add(f_open).eq(f), up_f.add(f_ext).eq(f));
                    best.or(ix).or(iy_next).narrow(&mut out[t * L..]);
                }
                (up_m, up_f) = (vm, f);
                v_in = v_in.add(f_ext);
            }
            // Ix predecessors of the first segment come from the last segment of the lane
            // above (or the empty row 0 for the first lane).
            let ix0 = flags(
                ix_bits,
                up_m.shift_in(V::NEG).add(v_shift).add(f_open).eq(first_f),
                up_f.shift_in(V::NEG).add(v_shift).add(f_ext).eq(first_f),
            );
            first_packed.or(ix0).narrow(out);
            std::mem::swap(&mut s.h_prev, &mut s.h_cur);
        }
        if scalar_ends {
            let ends = ScalarEnds { m, n, segs, lanes: L, col_bytes, end_open, end_ext };
            s.last_h.clear();
            s.last_h.resize(m + 1, 0);
            s.last_iy.clear();
            s.last_iy.resize(m + 1, 0);
            for t in 0..segs {
                let (h, e) = (s.h_prev[t].to_array(), s.e[t].to_array());
                for lane in 0..L {
                    let row = lane * segs + t + 1;
                    if row <= m {
                        s.last_h[row] = h[lane].into() + off[lane];
                        s.last_iy[row] = e[lane].into() + off[lane];
                    }
                }
            }
            ends.fill(enc, scoring, &s.last_row, &s.last_h, &mut s.last_iy, trace);
        }
        // Every stored score must stay exact after one lane shift.
        let limit = limit - max_shift;
        let (lo, hi) = (v_min.to_array(), v_max.to_array());
        !(lo.iter().any(|&v| v.into() < -limit) || hi.iter().any(|&v| v.into() > limit))
    }

    /// 16-bit fill; `false` if the scores did not fit.
    pub(super) fn fill_i16(buf: &mut TraceBuffers, scoring: &Scoring, frame: Frame) -> bool {
        match frame {
            // `frame_i16` only centres scores whose whole range fits.
            Frame::Centred(_) => fill::<16, I16x16, false>(buf, scoring, frame, 30_000),
            Frame::PerLane => fill::<16, I16x16, true>(buf, scoring, frame, 30_000),
        }
    }

    pub(super) fn fill_i32(buf: &mut TraceBuffers, scoring: &Scoring) {
        let exact = fill::<8, I32x8, true>(buf, scoring, Frame::Centred(0), i32::MAX / 8);
        assert!(exact, "32-bit alignment scores overflowed");
    }
}

#[cfg(test)]
mod tests {
    use super::super::scoring::{GapKind, MatrixKind, Substitution};
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    fn random_seq(seed: &mut u64, len: usize, letters: &[u8]) -> Vec<u8> {
        (0..len).map(|_| letters[lcg(seed) as usize % letters.len()]).collect()
    }

    #[test]
    fn striped_trace_matches_biopython_order() {
        let schemes = [
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, true),
            Scoring::new(Substitution::matrix(b'C', MatrixKind::MsaRefinement), GapKind::Msa, true),
            Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false),
            Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, true),
        ];
        let mut seed = 7;
        let mut buf = TraceBuffers::default();
        let mut reference = gotoh::AlignBuffers::default();
        for trial in 0..400 {
            let letters: &[u8] = match trial % 3 {
                0 => b"CCCCCCCCCCCCCCDA-",
                1 => b"CCCCCCADEF",
                _ => b"CD",
            };
            let la = 1 + lcg(&mut seed) as usize % if trial % 25 == 0 { 3000 } else { 120 };
            let lb = 1 + lcg(&mut seed) as usize % if trial % 25 == 0 { 3000 } else { 120 };
            let a: Vec<u8> = (0..la).map(|_| letters[lcg(&mut seed) as usize % letters.len()]).collect();
            let b: Vec<u8> = (0..lb).map(|_| letters[lcg(&mut seed) as usize % letters.len()]).collect();
            for s in &schemes {
                let got = align(&a, &b, s, &mut buf);
                let want = gotoh::align(&a, &b, s, &mut reference);
                assert_eq!(got, want, "trial {trial} lens {la},{lb}");
            }
        }
    }

    #[test]
    fn long_sequences_match_the_reference() {
        // 9,000-letter rows push the first column far below a centred 16-bit range
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::MsaRefinement), GapKind::Msa, true);
        let a = vec![b'C'; 9000];
        let mut b = a.clone();
        b.truncate(6000);
        b[3000] = b'D';
        let got = align(&a, &b, &s, &mut TraceBuffers::default());
        assert_eq!(got, gotoh::align(&a, &b, &s, &mut gotoh::AlignBuffers::default()));
    }

    /// `a` with random substitutions, insertions and deletions, like two reads of one allele.
    fn mutate(seed: &mut u64, a: &[u8], letters: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(a.len());
        for &c in a {
            match lcg(seed) % 40 {
                0 => {}
                1 => {
                    let len = 1 + lcg(seed) as usize % 30;
                    out.extend(random_seq(seed, len, letters));
                }
                2 => out.push(letters[lcg(seed) as usize % letters.len()]),
                _ => out.push(c),
            }
        }
        out
    }

    /// This machine's 16-bit vector fill, or `None` without one.
    #[allow(unreachable_code, unused_variables, reason = "which paths exist depends on the target")]
    fn simd_fill_i16(buf: &mut TraceBuffers, s: &Scoring, frame: Frame) -> Option<bool> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was checked just above.
            return Some(unsafe { avx2::fill_i16(buf, s, frame) });
        }
        #[cfg(target_arch = "aarch64")]
        return Some(neon::fill_i16(buf, s, frame));
        None
    }

    #[test]
    fn per_lane_frames_keep_long_alignments_in_16_bit_lanes() {
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, true);
        let letters = b"CCCCCCADEFAC";
        let mut seed = 5;
        let mut per_lane = 0;
        for (la, lb) in [(9000, 9000), (14_000, 4000), (300, 20_000), (12_000, 11_000)] {
            let a = random_seq(&mut seed, la, letters);
            let similar = {
                let mut b = mutate(&mut seed, &a, letters);
                b.truncate(lb);
                b
            };
            for b in [similar, random_seq(&mut seed, lb, letters), vec![b'D'; lb]] {
                let mut buf = TraceBuffers::default();
                buf.enc.encode(&a, &b, &s.sub);
                if !matches!(frame_i16(a.len(), b.len(), &s, &buf.enc.table), Frame::PerLane) {
                    continue;
                }
                per_lane += 1;
                let Some(held) = simd_fill_i16(&mut buf, &s, Frame::PerLane) else { return };
                assert!(held, "lens {},{}: 16-bit lanes overflowed", a.len(), b.len());
                let got = traceback(&a, &b, &buf.trace[buf.trace_start..], a.len().div_ceil(16), 16);
                let want = gotoh::align(&a, &b, &s, &mut gotoh::AlignBuffers::default());
                assert!(got == want, "lens {},{}: alignment differs from the reference", a.len(), b.len());
            }
        }
        // only the short "similar" copy of the 300-letter sequence fits a centred frame
        assert_eq!(per_lane, 11);
    }

    #[test]
    fn per_lane_frames_fall_back_when_a_lane_spreads_too_far() {
        // 3,750 rows per lane falling 4 per row below the diagonal: lanes end up 15,000
        // apart, which a lane shift cannot carry in 16 bits.
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, true);
        let (a, b) = (vec![b'D'; 60_000], vec![b'D'; 300]);
        let mut buf = TraceBuffers::default();
        buf.enc.encode(&a, &b, &s.sub);
        let Some(held) = simd_fill_i16(&mut buf, &s, Frame::PerLane) else { return };
        assert!(!held);
        let want = gotoh::align(&a, &b, &s, &mut gotoh::AlignBuffers::default());
        assert!(align(&a, &b, &s, &mut buf) == want, "32-bit fallback differs from the reference");
    }

    #[test]
    fn biased_16_bit_lanes_near_the_range_limit() {
        // Lengths whose score span only fits 16-bit lanes once it is centred on zero.
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, false);
        let mut seed = 11;
        let mut buf = TraceBuffers::default();
        for (la, lb) in [(5200, 5200), (6000, 4100), (2000, 7000)] {
            let a = random_seq(&mut seed, la, b"CCCCCCADEFAC");
            let pairs = [
                (a.clone(), a[..lb.min(la)].to_vec()),
                (vec![b'C'; la], vec![b'D'; lb]),
                (a, random_seq(&mut seed, lb, b"CCCCCCADEFAC")),
            ];
            for (a, b) in &pairs {
                let want = gotoh::align(a, b, &s, &mut gotoh::AlignBuffers::default());
                assert_eq!(align(a, b, &s, &mut buf), want, "lens {},{}", a.len(), b.len());
            }
        }
    }

    #[test]
    fn falls_back_when_16_bit_scores_overflow() {
        // 7,000 mostly canonical letters: the score span is too wide for 16-bit lanes
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::MsaRefinement), GapKind::Msa, true);
        let a: Vec<u8> = (0..7000).map(|i| if i % 50 == 0 { b'D' } else { b'C' }).collect();
        let b = a.clone();
        let got = align(&a, &b, &s, &mut TraceBuffers::default());
        assert_eq!(got, gotoh::align(&a, &b, &s, &mut gotoh::AlignBuffers::default()));
    }

    #[test]
    #[ignore = "benchmark on dumped MSA inputs (TELOCROC_MSA_BENCH=dir)"]
    fn bench_dumped_msa() {
        let dir = std::env::var("TELOCROC_MSA_BENCH").expect("set TELOCROC_MSA_BENCH");
        let s = Scoring::new(Substitution::matrix(b'C', MatrixKind::Msa), GapKind::Msa, true);
        let mut buf = TraceBuffers::default();
        let (mut cells, mut secs) = (0u64, 0f64);
        for entry in std::fs::read_dir(dir).expect("readable dump directory") {
            let text = std::fs::read_to_string(entry.expect("dump entry").path()).expect("readable dump");
            let seqs: Vec<&[u8]> = text.lines().filter_map(|l| l.strip_prefix("seq\t")).map(str::as_bytes).collect();
            for w in seqs.windows(2) {
                let t = std::time::Instant::now();
                let _ = align(w[0], w[1], &s, &mut buf);
                secs += t.elapsed().as_secs_f64();
                cells += (w[0].len() * w[1].len()) as u64;
            }
        }
        eprintln!("{cells:.3e} cells in {secs:.2}s = {:.2} Gcells/s", cells as f64 / secs / 1e9);
        // the same work on every thread at once
        use rayon::prelude::*;
        let threads = rayon::current_num_threads();
        let t = std::time::Instant::now();
        let files: Vec<String> = std::fs::read_dir(std::env::var("TELOCROC_MSA_BENCH").expect("set"))
            .expect("dir")
            .map(|e| std::fs::read_to_string(e.expect("entry").path()).expect("dump"))
            .collect();
        let total: u64 = (0..threads)
            .into_par_iter()
            .map(|_| {
                let mut buf = TraceBuffers::default();
                let mut cells = 0u64;
                for text in &files {
                    let seqs: Vec<&[u8]> =
                        text.lines().filter_map(|l| l.strip_prefix("seq\t")).map(str::as_bytes).collect();
                    for w in seqs.windows(2) {
                        let _ = align(w[0], w[1], &s, &mut buf);
                        cells += (w[0].len() * w[1].len()) as u64;
                    }
                }
                cells
            })
            .sum();
        let secs = t.elapsed().as_secs_f64();
        eprintln!("{threads} threads: {total:.3e} cells in {secs:.2}s = {:.2} Gcells/s", total as f64 / secs / 1e9);
    }
}
