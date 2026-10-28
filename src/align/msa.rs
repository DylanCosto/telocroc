//! Consensus TVR sequences: Telogator2's progressive alignment, iterative refinement and
//! column-vote consensus (`progressive_alignment`, `iterative_refinement`,
//! `get_final_tvr_consensus`).

use std::cell::RefCell;

use rayon::prelude::*;

use super::scoring::{GAP, Scoring};
use super::striped_trace::{self, TraceBuffers};
use crate::numpy;

thread_local! {
    static BUFFERS: RefCell<TraceBuffers> = RefCell::new(TraceBuffers::default());
}

fn pairwise_align(a: &[u8], b: &[u8], scoring: &Scoring) -> (Vec<u8>, Vec<u8>) {
    BUFFERS.with_borrow_mut(|buf| striped_trace::align(a, b, scoring, buf))
}

/// Drops alignment columns that are gaps in every row.
fn remove_gap_columns(rows: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let keep: Vec<bool> = (0..width).map(|c| rows.iter().any(|r| r.get(c).is_some_and(|&x| x != GAP))).collect();
    rows.iter().map(|r| r.iter().zip(&keep).filter(|&(_, &k)| k).map(|(&x, _)| x).collect()).collect()
}

fn pad_to(row: &mut Vec<u8>, len: usize) {
    if row.len() < len {
        row.resize(len, GAP);
    }
}

/// `progressive_alignment`: start from the closest pair, then repeatedly add the sequence
/// closest (on average) to the rows aligned so far, aligning it to every row.
///
/// `dists[i][j]` are the normalized single-precision distances between `seqs`.
pub fn progressive_alignment(seqs: &[Vec<u8>], dists: &[Vec<f32>], scoring: &Scoring) -> Vec<Vec<u8>> {
    let n = seqs.len();
    if n < 3 {
        return seqs.to_vec();
    }
    let mut closest = (0, 1);
    let mut min_distance = f32::INFINITY;
    for (i, row) in dists.iter().enumerate().take(n) {
        for (j, &d) in row.iter().enumerate().take(n).skip(i + 1) {
            if d < min_distance {
                min_distance = d;
                closest = (i, j);
            }
        }
    }
    let (row1, row2) = pairwise_align(&seqs[closest.0], &seqs[closest.1], scoring);
    // (aligned row, index into `dists`)
    let mut alignment: Vec<(Vec<u8>, usize)> = vec![(row1, closest.0), (row2, closest.1)];
    let mut remaining: Vec<usize> = (0..n).filter(|&i| i != closest.0 && i != closest.1).collect();
    while !remaining.is_empty() {
        let mean_dists: Vec<f32> = remaining
            .iter()
            .map(|&i| numpy::mean_f32(&alignment.iter().map(|&(_, j)| dists[i][j]).collect::<Vec<_>>()))
            .collect();
        let mut pick = 0;
        for (k, &d) in mean_dists.iter().enumerate() {
            if d.is_nan() {
                pick = k;
                break;
            }
            if d < mean_dists[pick] {
                pick = k;
            }
        }
        let new_seq = &seqs[remaining.remove(pick)];
        let aligned: Vec<(Vec<u8>, Vec<u8>)> =
            alignment.par_iter().map(|(row, _)| pairwise_align(row, new_seq, scoring)).collect();
        let max_len = aligned.iter().map(|(r, _)| r.len()).max().unwrap_or(0);
        let mut new_row = aligned[0].1.clone();
        pad_to(&mut new_row, max_len);
        for ((row, _), (realigned, _)) in alignment.iter_mut().zip(aligned) {
            *row = realigned;
            pad_to(row, max_len);
        }
        // Telogator2 labels every added row with the last sequence's index; keep that,
        // since it steers which distances later rows are averaged over.
        alignment.push((new_row, n - 1));
    }
    alignment.sort_by_key(|&(_, idx)| idx);
    let rows: Vec<Vec<u8>> = alignment.into_iter().map(|(r, _)| r).collect();
    remove_gap_columns(&rows)
}

/// One alignment column's letter frequencies, letters in first-appearance order.
type Column = Vec<(u8, f64)>;

fn column_profile(rows: &[&[u8]]) -> Vec<Column> {
    let width = rows.first().map_or(0, |r| r.len());
    let n = rows.len() as f64;
    (0..width)
        .map(|c| {
            let mut counts: Vec<(u8, usize)> = Vec::new();
            for r in rows {
                let x = r[c];
                match counts.iter_mut().find(|(l, _)| *l == x) {
                    Some((_, k)) => *k += 1,
                    None => counts.push((x, 1)),
                }
            }
            counts.into_iter().map(|(l, k)| (l, k as f64 / n)).collect()
        })
        .collect()
}

fn score_against_profile(seq: &[u8], profile: &[Column], scoring: &Scoring) -> f64 {
    let mut score = 0.0;
    for (&c, col) in seq.iter().zip(profile) {
        if c == GAP {
            continue;
        }
        for &(p, freq) in col {
            if p != GAP {
                score += freq * f64::from(scoring.sub.score(c, p));
            }
        }
    }
    score
}

fn alignment_score(rows: &[Vec<u8>], scoring: &Scoring) -> f64 {
    let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
    let profile = column_profile(&refs);
    rows.iter().map(|r| score_against_profile(r, &profile, scoring)).sum()
}

/// `iterative_refinement`: realign each row to the consensus of the others until the
/// sum-of-profile score improves by less than 1%.
///
/// Python breaks ties between equally frequent column letters in hash order, which
/// varies between runs; this takes the letter that appears first in the column.
pub fn iterative_refinement(initial: Vec<Vec<u8>>, scoring: &Scoring) -> Vec<Vec<u8>> {
    const MAX_ITERATIONS: usize = 10;
    const IMPROVEMENT_THRESHOLD: f64 = 0.01;
    if initial.len() < 3 {
        return initial;
    }
    let mut current = initial;
    let mut current_score = alignment_score(&current, scoring);
    for _ in 0..MAX_ITERATIONS {
        let mut new_alignment: Vec<Vec<u8>> = (0..current.len())
            .into_par_iter()
            .map(|i| {
                let others: Vec<&[u8]> =
                    current.iter().enumerate().filter(|&(k, _)| k != i).map(|(_, r)| r.as_slice()).collect();
                let consensus: Vec<u8> = column_profile(&others)
                    .iter()
                    .map(|col| {
                        let mut best = col[0];
                        for &cand in &col[1..] {
                            if cand.1 > best.1 {
                                best = cand;
                            }
                        }
                        best.0
                    })
                    .collect();
                let ungapped: Vec<u8> = current[i].iter().copied().filter(|&c| c != GAP).collect();
                pairwise_align(&consensus, &ungapped, scoring).1
            })
            .collect();
        let max_len = new_alignment.iter().map(Vec::len).max().unwrap_or(0);
        for row in &mut new_alignment {
            pad_to(row, max_len);
        }
        let new_score = alignment_score(&new_alignment, scoring);
        if current_score.abs() < 1e-6 {
            if new_score < current_score {
                break;
            }
        } else if (new_score - current_score) / current_score.abs() < IMPROVEMENT_THRESHOLD {
            break;
        }
        current = new_alignment;
        current_score = new_score;
    }
    remove_gap_columns(&current)
}

/// Parameters for [`final_tvr_consensus`].
#[derive(Clone, Debug)]
pub struct ConsensusRules {
    pub default_char: u8,
    pub untrustworthy: Vec<u8>,
    /// Bonus per letter when breaking ties (`tiebreak_adj`).
    pub tiebreak: Vec<(u8, i32)>,
}

/// `get_final_tvr_consensus`: column-by-column vote over an alignment.
pub fn final_tvr_consensus(aligned: &[Vec<u8>], rules: &ConsensusRules) -> Vec<u8> {
    const MIN_COVERAGE: usize = 3;
    const MAX_GAP_FRAC: f64 = 0.60;
    const FILL: u8 = b' ';
    let rows: Vec<&[u8]> = aligned
        .iter()
        .map(|r| {
            let end = r.iter().rposition(|&c| c != GAP).map_or(0, |p| p + 1);
            &r[..end]
        })
        .collect();
    let n = rows.len();
    assert!(n > 0, "final_tvr_consensus received an empty alignment");
    if n < MIN_COVERAGE {
        return rows[0].to_vec();
    }
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut consensus = Vec::with_capacity(width);
    for c in 0..width {
        // Counter(col).most_common(): by count, ties in first-appearance order
        let mut counts: Vec<(u8, usize)> = Vec::new();
        for r in &rows {
            let x = r.get(c).copied().unwrap_or(FILL);
            match counts.iter_mut().find(|(l, _)| *l == x) {
                Some((_, k)) => *k += 1,
                None => counts.push((x, 1)),
            }
        }
        let count_of = |x: u8| counts.iter().find(|(l, _)| *l == x).map_or(0, |&(_, k)| k);
        let coverage = n - count_of(FILL);
        if coverage < MIN_COVERAGE {
            consensus.push(rules.default_char);
            continue;
        }
        let gap_count = count_of(GAP);
        counts.sort_by_key(|&(_, k)| std::cmp::Reverse(k));
        counts.retain(|&(l, _)| l != FILL && l != GAP);
        let trusted = |(l, k): (u8, usize)| !rules.untrustworthy.contains(&l) && k >= MIN_COVERAGE;
        if let Some(&first) = counts.first().filter(|&&x| trusted(x)) {
            consensus.push(first.0);
            continue;
        }
        if let Some(&second) = counts.get(1).filter(|&&x| trusted(x)) {
            consensus.push(second.0);
            continue;
        }
        if gap_count as f64 / n as f64 > MAX_GAP_FRAC {
            continue;
        }
        match counts.as_slice() {
            [] => continue,
            [only] => consensus.push(only.0),
            [first, second, ..] if first.1 > second.1 => consensus.push(first.0),
            _ => {
                let bonus = |l: u8| rules.tiebreak.iter().find(|&&(t, _)| t == l).map_or(0, |&(_, b)| b);
                let best = counts.iter().map(|&(l, _)| (bonus(l), l)).max().expect("counts is not empty");
                consensus.push(best.1);
            }
        }
    }
    consensus
}
