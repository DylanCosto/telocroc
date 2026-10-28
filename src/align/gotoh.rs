//! Biopython's three-state Gotoh global alignment (`GOTOH_GLOBAL_ALIGN`), with the same
//! traceback preference, so the first optimal alignment matches `aligner.align(a, b)[0]`.

use super::scoring::Scoring;

const NEG: i32 = i32::MIN / 4;

const FROM_M: u8 = 0;
const FROM_IX: u8 = 1;
const FROM_IY: u8 = 2;
const FROM_NONE: u8 = 3;

/// The first of (M, Ix, Iy) achieving the maximum, as Biopython's traceback prefers.
#[inline(always)]
fn best3(m: i32, ix: i32, iy: i32) -> (i32, u8) {
    let (mut best, mut from) = (m, FROM_M);
    if ix > best {
        (best, from) = (ix, FROM_IX);
    }
    if iy > best {
        (best, from) = (iy, FROM_IY);
    }
    (best, from)
}

/// Reusable buffers for [`align`].
#[derive(Debug, Default)]
pub struct AlignBuffers {
    m: Vec<i32>,
    ix: Vec<i32>,
    iy: Vec<i32>,
    prev_m: Vec<i32>,
    prev_ix: Vec<i32>,
    prev_iy: Vec<i32>,
    trace: Vec<u8>,
}

/// Gap scores for a vertical (Ix, consumes `a`) and horizontal (Iy, consumes `b`) step.
struct Gaps {
    open: i32,
    extend: i32,
    right_open: i32,
    right_extend: i32,
}

impl Gaps {
    fn new(s: &Scoring) -> Self {
        let (right_open, right_extend) = if s.free_right_gaps { (0, 0) } else { (s.open, s.extend) };
        Self { open: s.open, extend: s.extend, right_open, right_extend }
    }
}

/// Global alignment score by the plain three-state recurrence (reference implementation).
pub fn score(a: &[u8], b: &[u8], scoring: &Scoring) -> i32 {
    let mut buf = AlignBuffers::default();
    fill(a, b, scoring, &mut buf, false)
}

/// Fills the DP; with `keep_trace`, records each state's preferred predecessor per cell.
fn fill(a: &[u8], b: &[u8], scoring: &Scoring, buf: &mut AlignBuffers, keep_trace: bool) -> i32 {
    let (na, nb) = (a.len(), b.len());
    let g = Gaps::new(scoring);
    let width = nb + 1;
    for v in [&mut buf.m, &mut buf.ix, &mut buf.iy, &mut buf.prev_m, &mut buf.prev_ix, &mut buf.prev_iy] {
        v.clear();
        v.resize(width, NEG);
    }
    if keep_trace {
        buf.trace.clear();
        buf.trace.resize((na + 1) * width, 0);
    }
    // row 0: only leading horizontal gaps
    buf.prev_m[0] = 0;
    for j in 1..=nb {
        buf.prev_iy[j] = scoring.left_gap(j);
        if keep_trace {
            let from = if j == 1 { FROM_M } else { FROM_IY };
            buf.trace[j] = pack(FROM_NONE, FROM_NONE, from);
        }
    }
    if keep_trace {
        buf.trace[0] = pack(FROM_NONE, FROM_NONE, FROM_NONE);
    }
    for i in 1..=na {
        let ca = a[i - 1];
        let (h_open, h_extend) = if i == na { (g.right_open, g.right_extend) } else { (g.open, g.extend) };
        buf.m[0] = NEG;
        buf.ix[0] = scoring.left_gap(i);
        buf.iy[0] = NEG;
        let row = i * width;
        if keep_trace {
            let from = if i == 1 { FROM_M } else { FROM_IX };
            buf.trace[row] = pack(FROM_NONE, from, FROM_NONE);
        }
        for j in 1..=nb {
            let (v_open, v_extend) = if j == nb { (g.right_open, g.right_extend) } else { (g.open, g.extend) };
            let (dm, fm) = best3(buf.prev_m[j - 1], buf.prev_ix[j - 1], buf.prev_iy[j - 1]);
            buf.m[j] = dm + scoring.sub.score(ca, b[j - 1]);
            let (vx, fx) = best3(buf.prev_m[j] + v_open, buf.prev_ix[j] + v_extend, buf.prev_iy[j] + v_open);
            buf.ix[j] = vx;
            let (hy, fy) = best3(buf.m[j - 1] + h_open, buf.ix[j - 1] + h_open, buf.iy[j - 1] + h_extend);
            buf.iy[j] = hy;
            if keep_trace {
                buf.trace[row + j] = pack(fm, fx, fy);
            }
        }
        std::mem::swap(&mut buf.m, &mut buf.prev_m);
        std::mem::swap(&mut buf.ix, &mut buf.prev_ix);
        std::mem::swap(&mut buf.iy, &mut buf.prev_iy);
    }
    let (best, _) = best3(buf.prev_m[nb], buf.prev_ix[nb], buf.prev_iy[nb]);
    best
}

#[inline(always)]
fn pack(from_m: u8, from_ix: u8, from_iy: u8) -> u8 {
    from_m | (from_ix << 2) | (from_iy << 4)
}

/// `str(alignment[0]), str(alignment[1])` for the first alignment Biopython reports.
///
/// # Panics
///
/// Panics if either sequence is empty (Biopython rejects those too).
pub fn align(a: &[u8], b: &[u8], scoring: &Scoring, buf: &mut AlignBuffers) -> (Vec<u8>, Vec<u8>) {
    assert!(!a.is_empty() && !b.is_empty(), "cannot align an empty sequence");
    fill(a, b, scoring, buf, true);
    let (na, nb) = (a.len(), b.len());
    let width = nb + 1;
    // final state: M, then Ix, then Iy among those reaching the best score
    let (_, mut state) = best3(buf.prev_m[nb], buf.prev_ix[nb], buf.prev_iy[nb]);
    let (mut i, mut j) = (na, nb);
    let mut row_a = Vec::with_capacity(na + nb);
    let mut row_b = Vec::with_capacity(na + nb);
    while state != FROM_NONE {
        let cell = buf.trace[i * width + j];
        let from = (cell >> (2 * state)) & 3;
        match state {
            FROM_M => {
                row_a.push(a[i - 1]);
                row_b.push(b[j - 1]);
                i -= 1;
                j -= 1;
            }
            FROM_IX => {
                row_a.push(a[i - 1]);
                row_b.push(b'-');
                i -= 1;
            }
            _ => {
                row_a.push(b'-');
                row_b.push(b[j - 1]);
                j -= 1;
            }
        }
        state = from;
        if i == 0 && j == 0 {
            break;
        }
    }
    row_a.reverse();
    row_b.reverse();
    (row_a, row_b)
}
