//! Alignment-based distances between repeat color vectors (`tvr_distance`) and the
//! all-pairs distance matrices used for clustering (`get_dist_matrix_parallel`).

use std::cell::RefCell;

use rayon::prelude::*;

use super::batch::{self, BatchBuffers, BatchScorer};
use super::scoring::Scoring;
use super::striped::{self, ScoreBuffers};
use crate::numpy;
use crate::pyrandom::PyRandom;

/// Upper bound on distances (the log ratio can diverge).
pub const MAX_SEQ_DIST: f64 = 10.0;
/// Lower bound on distances, to keep rounding from producing zero or negative values.
const MIN_SEQ_DIST: f64 = 0.0001;
/// Below `-IDEN_SCORE_SCALAR_CHEAT * identity score` the distance is taken as maximal.
const IDEN_SCORE_SCALAR_CHEAT: f64 = 0.900;
/// When comparing sequences of different lengths, truncate both to the shorter one, but
/// not below this.
const MIN_VIABLE_SEQ_LEN: usize = 1000;

/// How `tvr_distance` trims the two sequences before aligning them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthAdjust {
    /// Compare the sequences as they are.
    None,
    /// Truncate both to the shorter length, but keep at least 1000 letters.
    TruncateMinViable,
}

impl LengthAdjust {
    fn apply<'s>(self, a: &'s [u8], b: &'s [u8]) -> (&'s [u8], &'s [u8]) {
        match self {
            Self::None => (a, b),
            Self::TruncateMinViable => {
                let min_len = a.len().min(b.len()).max(MIN_VIABLE_SEQ_LEN);
                (&a[..min_len.min(a.len())], &b[..min_len.min(b.len())])
            }
        }
    }
}

#[derive(Debug, Default)]
struct DistBuffers {
    score: ScoreBuffers,
    batch: BatchBuffers,
    shuffled_a: Vec<u8>,
    shuffled_b: Vec<u8>,
    pool: Vec<u8>,
}

thread_local! {
    static BUFFERS: RefCell<DistBuffers> = RefCell::new(DistBuffers::default());
}

/// Python's `min(a, b)` and `max(a, b)`, which keep the first argument on NaN.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// `tvr_distance`: how much better than shuffled sequences the two align, as a distance in
/// `[MIN_SEQ_DIST, MAX_SEQ_DIST]`. The shuffles draw from `rng`.
pub fn tvr_distance(
    tvr_i: &[u8],
    tvr_j: &[u8],
    scoring: &Scoring,
    adjust: LengthAdjust,
    shuffles: usize,
    rng: &mut PyRandom,
) -> f64 {
    BUFFERS.with_borrow_mut(|buf| {
        let (seq_i, seq_j) = adjust.apply(tvr_i, tvr_j);
        let aln_score = striped::score(seq_i, seq_j, scoring, &mut buf.score);
        let Some(iden_score) = worth_shuffling(seq_i, seq_j, scoring, aln_score) else {
            return MAX_SEQ_DIST;
        };
        let mut rand_scores = Vec::with_capacity(shuffles);
        for _ in 0..shuffles {
            rng.sample_all_into(seq_i, &mut buf.shuffled_a, &mut buf.pool);
            rng.sample_all_into(seq_j, &mut buf.shuffled_b, &mut buf.pool);
            rand_scores.push(striped::score(&buf.shuffled_a, &buf.shuffled_b, scoring, &mut buf.score));
        }
        shuffled_distance(aln_score, iden_score, &rand_scores)
    })
}

/// The pair's identity score, or `None` when the alignment is so poor that the distance is
/// `MAX_SEQ_DIST` without shuffling.
fn worth_shuffling(a: &[u8], b: &[u8], scoring: &Scoring, aln_score: i32) -> Option<f64> {
    let iden_score = (scoring.self_score(a) + scoring.self_score(b)) / 2.0;
    (f64::from(aln_score) >= -(IDEN_SCORE_SCALAR_CHEAT * iden_score)).then_some(iden_score)
}

/// The distance from the alignment, identity and shuffled-alignment scores.
fn shuffled_distance(aln_score: i32, iden_score: f64, rand_scores: &[i32]) -> f64 {
    let aln_score = f64::from(aln_score);
    let rand_scores: Vec<f64> = rand_scores.iter().map(|&s| f64::from(s)).collect();
    let rand_score = numpy::mean(&rand_scores);
    let dist = if rand_score >= aln_score {
        MAX_SEQ_DIST
    } else {
        py_min(-((aln_score - rand_score) / (iden_score - rand_score)).ln(), MAX_SEQ_DIST)
    };
    py_max(dist, MIN_SEQ_DIST)
}

/// A symmetric `float32` distance matrix (`np.zeros((n, n), dtype='<f4')`).
#[derive(Clone, Debug)]
pub struct DistMatrix {
    n: usize,
    values: Vec<f32>,
}

impl DistMatrix {
    /// Raw little-endian `float32` values, row-major (for debugging dumps).
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.values[i * self.n + j]
    }

    /// `dist_matrix /= MAX_SEQ_DIST`, in single precision as NumPy does.
    pub fn normalize(&mut self) {
        for v in &mut self.values {
            *v /= MAX_SEQ_DIST as f32;
        }
    }

    /// `squareform(dist_matrix)`: the upper triangle, row by row, widened to f64.
    pub fn condensed(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * self.n.saturating_sub(1) / 2);
        for i in 0..self.n {
            for j in i + 1..self.n {
                out.push(f64::from(self.get(i, j)));
            }
        }
        out
    }
}

/// `get_dist_matrix_parallel`: all pairwise distances. Pair `k` of
/// `itertools.combinations(range(n), 2)` uses `random.seed(rng_seed + k)`, so the
/// result does not depend on how work is split across threads.
pub fn dist_matrix(
    seqs: &[Vec<u8>],
    scoring: &Scoring,
    adjust: LengthAdjust,
    shuffles: usize,
    rng_seed: u64,
) -> DistMatrix {
    let n = seqs.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let dists: Vec<f64> = match BatchScorer::new(scoring) {
        Some(scorer) => {
            let pairs: Vec<(&[u8], &[u8])> = pairs.iter().map(|&(i, j)| adjust.apply(&seqs[i], &seqs[j])).collect();
            batched_distances(&pairs, scoring, &scorer, shuffles, rng_seed)
        }
        None => pairs
            .par_iter()
            .enumerate()
            .with_min_len(4)
            .map(|(k, &(i, j))| {
                let mut rng = PyRandom::new(rng_seed + k as u64);
                tvr_distance(&seqs[i], &seqs[j], scoring, adjust, shuffles, &mut rng)
            })
            .collect(),
    };
    let mut values = vec![0f32; n * n];
    for (&(i, j), &d) in pairs.iter().zip(&dists) {
        values[i * n + j] = d as f32;
        values[j * n + i] = d as f32;
    }
    DistMatrix { n, values }
}

/// Batches smaller than this go through the striped kernel instead, which is faster for
/// a few pairs than a batch with most lanes idle.
const MIN_BATCH: usize = 14;

/// The distances `tvr_distance` gives for already trimmed pairs (pair `k` shuffles with
/// `random.seed(rng_seed + k)`), aligning [`batch::LANES`] pairs at a time.
///
/// Pairs are sorted by size so each batch's pairs are about the same size; every batch
/// first aligns the real pairs, then shuffles only those that need it.
fn batched_distances(
    pairs: &[(&[u8], &[u8])],
    scoring: &Scoring,
    scorer: &BatchScorer,
    shuffles: usize,
    rng_seed: u64,
) -> Vec<f64> {
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&k| (pairs[k].0.len(), pairs[k].1.len()));
    // Several chunks per thread, so threads stay busy when sizes vary across chunks.
    let per_chunk = (pairs.len() / (4 * rayon::current_num_threads())).clamp(batch::LANES, 8 * batch::LANES);
    let chunks: Vec<Vec<f64>> = order
        .par_chunks(per_chunk)
        .map(|chunk| {
            BUFFERS.with_borrow_mut(|buf| {
                let real: Vec<(&[u8], &[u8])> = chunk.iter().map(|&k| pairs[k]).collect();
                let aln_scores = score_all(&real, scoring, scorer, buf);
                let iden: Vec<Option<f64>> =
                    real.iter().zip(&aln_scores).map(|(&(a, b), &s)| worth_shuffling(a, b, scoring, s)).collect();
                // All shuffled copies, in pair order, then the pairs that align them.
                let mut letters = Vec::new();
                for ((&k, &(a, b)), _) in chunk.iter().zip(&real).zip(&iden).filter(|(_, i)| i.is_some()) {
                    let mut rng = PyRandom::new(rng_seed + k as u64);
                    for _ in 0..shuffles {
                        rng.sample_all_into(a, &mut buf.shuffled_a, &mut buf.pool);
                        letters.extend_from_slice(&buf.shuffled_a);
                        rng.sample_all_into(b, &mut buf.shuffled_b, &mut buf.pool);
                        letters.extend_from_slice(&buf.shuffled_b);
                    }
                }
                let mut rest = letters.as_slice();
                let mut shuffled = Vec::new();
                for (&(a, b), _) in real.iter().zip(&iden).filter(|(_, i)| i.is_some()) {
                    for _ in 0..shuffles {
                        let (sa, tail) = rest.split_at(a.len());
                        let (sb, tail) = tail.split_at(b.len());
                        shuffled.push((sa, sb));
                        rest = tail;
                    }
                }
                let rand_scores = score_all(&shuffled, scoring, scorer, buf);
                let mut rand_scores = rand_scores.chunks(shuffles.max(1));
                aln_scores
                    .iter()
                    .zip(&iden)
                    .map(|(&aln, iden)| match iden {
                        Some(iden) => shuffled_distance(aln, *iden, rand_scores.next().unwrap_or(&[])),
                        None => MAX_SEQ_DIST,
                    })
                    .collect()
            })
        })
        .collect();
    let mut dists = vec![0.0; pairs.len()];
    for (&k, d) in order.iter().zip(chunks.into_iter().flatten()) {
        dists[k] = d;
    }
    dists
}

/// Alignment scores of `pairs`, in order.
fn score_all(pairs: &[(&[u8], &[u8])], scoring: &Scoring, scorer: &BatchScorer, buf: &mut DistBuffers) -> Vec<i32> {
    let mut out = vec![0; pairs.len()];
    for (group, scores) in pairs.chunks(batch::LANES).zip(out.chunks_mut(batch::LANES)) {
        if group.len() >= MIN_BATCH && group.iter().all(|(a, b)| !a.is_empty() && !b.is_empty()) {
            scorer.score(group, &mut buf.batch, scores);
        } else {
            for (&(a, b), s) in group.iter().zip(scores) {
                *s = striped::score(a, b, scoring, &mut buf.score);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::scoring::{GapKind, MatrixKind, Substitution};
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    /// The batched matrix is bit for bit the one `tvr_distance` gives pair by pair.
    #[test]
    fn batched_matrix_matches_pairwise_distances() {
        let scoring = Scoring::new(Substitution::matrix(b'C', MatrixKind::Tvr), GapKind::Tvr, false);
        assert!(BatchScorer::new(&scoring).is_some(), "the TVR scheme takes the batched path");
        let mut seed = 3;
        let base: Vec<u8> = (0..1500).map(|_| b"CCCCCCCCCDEFA"[lcg(&mut seed) as usize % 13]).collect();
        // Related sequences (a shared base with mutations) and unrelated ones, of mixed
        // lengths on both sides of MIN_VIABLE_SEQ_LEN.
        let seqs: Vec<Vec<u8>> = (0..45)
            .map(|i| {
                let len = [1500, 1200, 999, 400, 1000, 60][i % 6];
                let mut s = base[..len].to_vec();
                for c in s.iter_mut() {
                    let rate = if i % 4 == 0 { 2 } else { 20 };
                    if lcg(&mut seed).is_multiple_of(rate) {
                        *c = b"CDEFGA"[lcg(&mut seed) as usize % 6];
                    }
                }
                s
            })
            .collect();
        for (adjust, shuffles) in
            [(LengthAdjust::TruncateMinViable, 2), (LengthAdjust::None, 3), (LengthAdjust::None, 1)]
        {
            let dist = dist_matrix(&seqs, &scoring, adjust, shuffles, 17);
            let mut k = 0;
            for i in 0..seqs.len() {
                for j in i + 1..seqs.len() {
                    let mut rng = PyRandom::new(17 + k);
                    let want = tvr_distance(&seqs[i], &seqs[j], &scoring, adjust, shuffles, &mut rng) as f32;
                    assert_eq!(dist.get(i, j).to_bits(), want.to_bits(), "pair {i},{j} ({adjust:?})");
                    k += 1;
                }
            }
        }
    }
}
