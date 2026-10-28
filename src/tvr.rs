//! TVR clustering (`tg_tvr.py`): paint each read's repeats into a color vector, find its
//! subtelomere / TVR / telomere boundaries, cluster reads by TVR similarity, and build a
//! consensus TVR per cluster.

use rayon::prelude::*;

use crate::align::distance::{self, LengthAdjust};
use crate::align::msa::{self, ConsensusRules};
use crate::align::scoring::{GapKind, MatrixKind, Scoring, Substitution, UNKNOWN};
use crate::kmers::KmerTable;
use crate::linkage::{self, Method};
use crate::telrep::RepeatRead;

/// Ignore variant repeats past this point when finding the TVR boundary.
const MAX_TVR_LEN: usize = 8000;
/// ...or past this point when there are very few variant repeats.
const MAX_TVR_LEN_SHORT: usize = 3000;
/// Bases added to a detected TVR boundary.
const TVR_BOUNDARY_BUFF: usize = 10;
/// How far to look for more variant repeats when extending the TVR boundary.
const TVR_BOUNDARY_ADJ_STEP: usize = 50;
/// Density thresholds for the subtelomere / TVR boundary in individual reads.
const UNKNOWN_WIN_SIZE: usize = 100;
const UNKNOWN_END_DENS: f64 = 0.125;
const UNKNOWN_WIN_SIZE_FINE: usize = 50;
const UNKNOWN_END_DENS_FINE: f64 = 0.500;
/// Density threshold separating canonical repeats from end-of-read sequencing artifacts.
const CANON_WIN_SIZE: usize = 100;
const CANON_END_DENS: f64 = 0.700;
/// TVR / canonical boundary parameters: (denoise block size, cumulative fraction, min hits).
const TVR_CANON_FILT_STRICT: (usize, f64, usize) = (10, 0.125, 100);
const TVR_CANON_FILT_LENIENT: (usize, f64, usize) = (5, 0.250, 50);
/// Reads per cluster used to build a consensus (the longest ones).
const MAX_MSA_READCOUNT: usize = 10;

/// Letter classes from the kmer flags.
#[derive(Clone, Debug)]
pub struct Letters {
    pub letters: Vec<u8>,
    pub canonical: u8,
    denoise: Vec<u8>,
    tvr: Vec<u8>,
    nofilt: Vec<u8>,
    dubious: Vec<u8>,
    subfilt: Vec<u8>,
}

impl Letters {
    pub fn new(kmers: &KmerTable) -> Self {
        let canonical =
            kmers.kmers.iter().rev().find(|k| k.has_flag("canonical")).map_or(kmers.canonical_letter, |k| k.letter);
        let mut dubious = kmers.letters_with_flag("dubious");
        if !dubious.contains(&UNKNOWN) {
            dubious.push(UNKNOWN);
        }
        Self {
            letters: kmers.letters().collect(),
            canonical,
            denoise: kmers.letters_with_flag("denoise"),
            tvr: kmers.letters_with_flag("tvr"),
            nofilt: kmers.letters_with_flag("nofilt"),
            dubious,
            subfilt: kmers.letters_with_flag("subtel-filt"),
        }
    }

    fn worth_chasing(&self, c: u8) -> bool {
        self.tvr.contains(&c) && !self.dubious.contains(&c)
    }

    fn unknown_like(&self) -> Vec<u8> {
        let mut v = vec![UNKNOWN];
        v.extend(&self.subfilt);
        v
    }

    fn consensus_rules(&self) -> ConsensusRules {
        let mut untrustworthy = vec![self.canonical];
        untrustworthy.extend(&self.dubious);
        ConsensusRules {
            default_char: self.canonical,
            untrustworthy,
            tiebreak: vec![(self.canonical, 1), (UNKNOWN, -1)],
        }
    }
}

/// A read's color vector split into subtelomere, TVR + telomere, and trimmed read end.
#[derive(Clone, Debug)]
pub struct ReadColors {
    /// Length of the subtelomere part of the color vector (`all_subtel_lens`).
    pub subtel_len: usize,
    /// Length of the TVR + telomere part (`all_tvrtel_lens`).
    pub tvrtel_len: usize,
    /// Length trimmed off the read end as likely sequencing artifact (`err_end_lens`).
    pub err_end_len: usize,
    /// Denoised TVR + telomere colors used for alignment (`colorvecs_for_msa`).
    pub tvrtel: Vec<u8>,
}

/// Paints each kmer hit's letter onto an all-unknown vector, later kmers winning.
fn color_vector(read: &RepeatRead, letters: &[u8]) -> Vec<u8> {
    let mut v = vec![UNKNOWN; read.tvrtel_len];
    for (hits, &letter) in read.hits.iter().zip(letters) {
        for &(s, e) in hits {
            v[s..e].fill(letter);
        }
    }
    v
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Below,
    Above,
}

/// `find_density_boundary`: first window start whose density of `which` crosses
/// `thresh`, else the most extreme window. `None` when there is no window at all, which
/// Python's `seq[:None], seq[None:]` turns into two copies of the whole sequence.
fn density_boundary(seq: &[u8], which: &[u8], win: usize, thresh: f64, dir: Direction) -> Option<usize> {
    let mut cum = Vec::with_capacity(seq.len());
    let mut total = 0u32;
    for c in seq {
        total += u32::from(which.contains(c));
        cum.push(total);
    }
    let mut first = None;
    let mut extreme_pos = None;
    let mut extreme = if dir == Direction::Below { 1.0 } else { 0.0 };
    for j in 0..seq.len().saturating_sub(win) {
        let dens = f64::from(cum[j + win] - cum[j]) / win as f64;
        match dir {
            Direction::Below => {
                if first.is_none() && dens <= thresh {
                    first = Some(j);
                }
                if dens < extreme {
                    extreme = dens;
                    extreme_pos = Some(j);
                }
            }
            Direction::Above => {
                if first.is_none() && dens >= thresh {
                    first = Some(j);
                }
                if dens > extreme {
                    extreme = dens;
                    extreme_pos = Some(j);
                }
            }
        }
    }
    first.or(extreme_pos)
}

fn split_at_boundary(seq: &[u8], pos: Option<usize>) -> (Vec<u8>, Vec<u8>) {
    match pos {
        Some(p) => (seq[..p].to_vec(), seq[p..].to_vec()),
        None => (seq.to_vec(), seq.to_vec()),
    }
}

fn reversed(s: &[u8]) -> Vec<u8> {
    s.iter().rev().copied().collect()
}

/// `denoise_colorvec`: drop short runs of noisy letters and bridge small gaps between
/// runs of the same letter. Like the original, the result is one letter longer than `v`.
fn denoise(v: &[u8], replace: u8, min_size: usize, delete: &[u8], merge: &[u8]) -> Vec<u8> {
    const MAX_GAP_FILL: usize = 50;
    if v.is_empty() {
        return Vec::new();
    }
    let len = v.len() + 1;
    let at = |i: usize| if i < v.len() { v[i] } else { replace };
    let mut blocks: Vec<(usize, usize, u8)> = Vec::new();
    let (mut current, mut start) = (v[0], 0);
    for i in 1..len {
        if at(i) != current {
            blocks.push((start, i, current));
            (current, start) = (at(i), i);
        }
    }
    blocks.retain(|&(s, e, c)| !(e - s < min_size && delete.contains(&c)));
    for i in (1..blocks.len()).rev() {
        if i >= blocks.len() {
            continue;
        }
        let (prev, cur) = (blocks[i - 1], blocks[i]);
        if cur.2 == prev.2 && merge.contains(&cur.2) && cur.0 - prev.1 <= MAX_GAP_FILL {
            blocks[i - 1] = (prev.0, cur.1, cur.2);
            blocks.remove(i);
        }
    }
    let mut out = vec![replace; len];
    for (s, e, c) in blocks {
        out[s..e].fill(c);
    }
    out
}

/// `find_cumulative_boundary`: first position where the running fraction of `which`
/// letters reaches `thresh`; `len(seq)` if there are fewer than `min_hits` of them.
fn cumulative_boundary(seq: &[u8], which: &[u8], thresh: f64, min_hits: usize) -> usize {
    let total = seq.iter().filter(|c| which.contains(c)).count();
    if total < min_hits || total == 0 {
        return seq.len();
    }
    let mut cum = 0usize;
    for (i, c) in seq.iter().enumerate() {
        cum += usize::from(which.contains(c));
        if cum as f64 / total as f64 >= thresh {
            return i;
        }
    }
    seq.len()
}

/// Splits one read's color vector (the per-read part of `cluster_tvrs`).
pub fn read_colors(read: &RepeatRead, letters: &Letters) -> ReadColors {
    let cv = color_vector(read, &letters.letters);
    let unknown = letters.unknown_like();
    let (left, right) =
        split_at_boundary(&cv, density_boundary(&cv, &unknown, UNKNOWN_WIN_SIZE, UNKNOWN_END_DENS, Direction::Below));
    // walk backwards with a smaller window to see if there are some tvrs we missed
    let left_rev = reversed(&left);
    let (recovered, rest_is_subtel) = split_at_boundary(
        &left_rev,
        density_boundary(&left_rev, &unknown, UNKNOWN_WIN_SIZE_FINE, UNKNOWN_END_DENS_FINE, Direction::Above),
    );
    let mut seq_right = reversed(&recovered);
    seq_right.extend_from_slice(&right);
    let mut seq_left = reversed(&rest_is_subtel);
    // the tvr should not begin with an unknown character
    let mut adj = 0;
    while adj + 1 < seq_right.len() && seq_right[adj] == UNKNOWN {
        adj += 1;
    }
    seq_left.extend_from_slice(&seq_right[..adj]);
    let seq_right = seq_right.split_off(adj);
    let mut merge = vec![letters.canonical];
    merge.extend(&letters.tvr);
    let denoised = denoise(&seq_right, letters.canonical, 10, &letters.denoise, &merge);
    // trim read ends that look like sequencing artifacts (low canonical density)
    let denoised_rev = reversed(&denoised);
    let (err_left, err_right) = split_at_boundary(
        &denoised_rev,
        density_boundary(&denoised_rev, &[letters.canonical], CANON_WIN_SIZE, CANON_END_DENS, Direction::Above),
    );
    ReadColors {
        subtel_len: seq_left.len(),
        tvrtel_len: seq_right.len(),
        err_end_len: err_left.len(),
        tvrtel: reversed(&err_right),
    }
}

/// Settings for one `cluster_tvrs` call.
#[derive(Clone, Copy, Debug)]
pub struct TvrClusterParams {
    pub tvr_truncate: usize,
    pub shuffles: usize,
    pub tree_cut: f64,
    pub clustering_only: bool,
    pub rng_seed: u64,
}

/// Result of `cluster_tvrs` (indices are into the reads passed in).
#[derive(Clone, Debug, Default)]
pub struct TvrClusters {
    pub clusters: Vec<Vec<usize>>,
    /// Consensus TVR + telomere colors per cluster (empty with `clustering_only`).
    pub consensus: Vec<Vec<u8>>,
    /// TVR length of each consensus (empty with `clustering_only`).
    pub tvr_lens: Vec<i64>,
    /// The normalized distance matrix the clusters came from.
    pub dist: Option<distance::DistMatrix>,
}

/// `cluster_tvrs` for reads already split by [`read_colors`].
pub fn cluster_tvrs(
    reads: &[&RepeatRead],
    colors: &[&ReadColors],
    letters: &Letters,
    p: TvrClusterParams,
) -> TvrClusters {
    let n = reads.len();
    let for_clustering: Vec<Vec<u8>> =
        colors.iter().map(|c| c.tvrtel[..c.tvrtel.len().min(p.tvr_truncate)].to_vec()).collect();
    let scoring = Scoring::new(Substitution::matrix(letters.canonical, MatrixKind::Tvr), GapKind::Tvr, false);
    let mut dist =
        distance::dist_matrix(&for_clustering, &scoring, LengthAdjust::TruncateMinViable, p.shuffles, p.rng_seed);
    dist.normalize();

    let mut clusters = vec![(0..n).collect::<Vec<_>>()];
    if n > 1 {
        clusters = sorted_clusters(linkage::cluster(&dist.condensed(), n, Method::Ward, p.tree_cut));
        for c in &mut clusters {
            // longest reads first
            c.sort_by(|&a, &b| (reads[b].tvrtel_len, b).cmp(&(reads[a].tvrtel_len, a)));
        }
    }
    if p.clustering_only {
        return TvrClusters { clusters, dist: Some(dist), ..Default::default() };
    }

    let msa_scoring = Scoring::new(Substitution::matrix(letters.canonical, MatrixKind::Msa), GapKind::Msa, true);
    let refine_scoring =
        Scoring::new(Substitution::matrix(letters.canonical, MatrixKind::MsaRefinement), GapKind::Msa, true);
    let rules = letters.consensus_rules();
    let consensus: Vec<Vec<u8>> = clusters
        .par_iter()
        .map(|members| {
            let mut by_len: Vec<(usize, usize)> = members.iter().map(|&ci| (colors[ci].tvrtel.len(), ci)).collect();
            by_len.sort_unstable_by(|a, b| b.cmp(a));
            let picked: Vec<usize> = by_len.iter().take(MAX_MSA_READCOUNT).map(|&(_, ci)| ci).collect();
            let sub_dists: Vec<Vec<f32>> =
                picked.iter().map(|&a| picked.iter().map(|&b| dist.get(a, b)).collect()).collect();
            let seqs: Vec<Vec<u8>> = picked.iter().map(|&ci| colors[ci].tvrtel.clone()).collect();
            let initial = msa::progressive_alignment(&seqs, &sub_dists, &msa_scoring);
            let refined = msa::iterative_refinement(initial, &refine_scoring);
            msa::final_tvr_consensus(&refined, &rules)
        })
        .collect();
    let tvr_lens = consensus.iter().map(|c| tvr_tel_boundary(c, letters)).collect();
    TvrClusters { clusters, consensus, tvr_lens, dist: Some(dist) }
}

/// `sorted([(len(c), sorted(c)) for c in clusters], reverse=True)`: largest first, ties
/// broken by the (sorted) member lists, larger first.
fn sorted_clusters(mut clusters: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for c in &mut clusters {
        c.sort_unstable();
    }
    clusters.sort_by(|a, b| (b.len(), b).cmp(&(a.len(), a)));
    clusters
}

/// Python list indexing, where a negative index counts from the end.
fn py_index(v: &[u8], i: i64) -> Option<u8> {
    let idx = if i < 0 { i + v.len() as i64 } else { i };
    usize::try_from(idx).ok().and_then(|k| v.get(k).copied())
}

/// Length of the TVR at the start of a consensus (the TVR / canonical boundary).
fn tvr_tel_boundary(consensus: &[u8], letters: &Letters) -> i64 {
    let len = consensus.len();
    let delete: Vec<u8> = letters.tvr.iter().copied().filter(|c| !letters.nofilt.contains(c)).collect();
    let mut merge = vec![letters.canonical];
    merge.extend(&letters.tvr);
    let attempt = |max_len: usize, (min_size, cum_thresh, min_hits): (usize, f64, usize)| {
        let mut current = consensus.to_vec();
        if len > max_len {
            current.truncate(max_len);
            current.resize(len, letters.canonical);
        }
        let denoised = reversed(&denoise(&current, letters.canonical, min_size, &delete, &merge));
        let boundary = cumulative_boundary(&denoised, &letters.tvr, cum_thresh, min_hits);
        (denoised, boundary)
    };
    let (mut denoised, mut boundary) = attempt(MAX_TVR_LEN, TVR_CANON_FILT_STRICT);
    let failed = len + 1;
    if boundary == failed {
        (denoised, boundary) = attempt(MAX_TVR_LEN_SHORT, TVR_CANON_FILT_LENIENT);
    }
    if boundary == failed {
        return len as i64 - boundary as i64 + 1;
    }
    let mut tb = boundary.saturating_sub(TVR_BOUNDARY_BUFF) as i64;
    let step = TVR_BOUNDARY_ADJ_STEP as i64;
    if tb > step + 1 {
        let chase = |i: i64| py_index(&denoised, i).is_some_and(|c| letters.worth_chasing(c));
        let found_ahead = (0..=step).rev().find(|&j| chase(tb - j));
        if let Some(found) = found_ahead {
            tb -= found;
            while chase(tb) {
                if tb <= 0 {
                    tb = 0;
                    break;
                }
                match (1..=step).find(|&j| chase(tb - j)) {
                    Some(j) => tb -= j,
                    None => break,
                }
            }
        } else {
            while py_index(&denoised, tb) == Some(letters.canonical) {
                tb += 1;
            }
        }
    }
    len as i64 - tb
}

/// `cluster_consensus_tvrs`: cluster whole sequences (consensus TVRs or subtelomeres)
/// with match/mismatch scoring and free trailing gaps.
pub fn cluster_sequences(
    seqs: &[Vec<u8>],
    truncate: Option<usize>,
    tree_cut: f64,
    method: Method,
    rng_seed: u64,
) -> (Vec<Vec<usize>>, distance::DistMatrix) {
    const SHUFFLES: usize = 3;
    let n = seqs.len();
    let trimmed: Vec<Vec<u8>> = match truncate {
        Some(t) => seqs.iter().map(|s| s[..s.len().min(t)].to_vec()).collect(),
        None => seqs.to_vec(),
    };
    let scoring = Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, true);
    let mut dist = distance::dist_matrix(&trimmed, &scoring, LengthAdjust::None, SHUFFLES, rng_seed);
    dist.normalize();
    if n <= 1 {
        return (vec![(0..n).collect()], dist);
    }
    (sorted_clusters(linkage::cluster(&dist.condensed(), n, method, tree_cut)), dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_is_one_longer_and_merges() {
        // tg_tvr.denoise_colorvec('CCCCCADDCCCCC', replace_char='C', chars_to_delete=['D'], chars_to_merge=['C'])
        let out = denoise(b"CCCCCADDCCCCC", b'C', 10, b"D", b"C");
        assert_eq!(out, b"CCCCCACCCCCCCC");
    }
}
