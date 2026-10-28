//! Stage 0b: telomere repeat composition of each read (`get_tel_repeat_comp_parallel`).
//!
//! Each read is turned to the q orientation, its terminal telomere is measured from the
//! p-vs-q repeat density signal, and reads that do not end in telomere are dropped. Kept
//! reads carry the positions of every repeat kmer in their TVR + telomere region.

use memchr::memmem;
use rayon::prelude::*;

use crate::kmers::KmerTable;
use crate::screen::TelRead;
use crate::util::revcomp;
use crate::wavelet;

pub type Span = (usize, usize);

const TEL_WINDOW_SIZE: usize = 100;
const P_VS_Q_AMP_THRESH: f64 = 0.5;
const MIN_TEL_SCORE: i64 = 100;

/// A read that ends in telomere (one `kmer_hit_dat` entry).
#[derive(Clone, Debug)]
pub struct RepeatRead {
    /// Non-overlapping repeat hits in the TVR + telomere region, per kmer in table order.
    pub hits: Vec<Vec<Span>>,
    /// Length of the TVR + telomere region (plus the subtelomere buffer).
    pub tvrtel_len: usize,
    /// Read name up to the first space.
    pub name: String,
    /// Whole read, in q orientation.
    pub seq: Vec<u8>,
}

/// Filtering thresholds for terminal telomeres (`--filt-*`).
#[derive(Clone, Copy, Debug)]
pub struct EndFilters {
    pub min_term_tel: i64,
    pub max_term_nontel: i64,
    pub min_term_subtel: i64,
}

/// Reads that passed, plus how many failed each filter (a read can fail several).
#[derive(Debug, Default)]
pub struct RepeatComposition {
    pub reads: Vec<RepeatRead>,
    pub removed_term_tel: usize,
    pub removed_term_nontel: usize,
    pub removed_term_subtel: usize,
}

struct ReadOutcome {
    kept: Option<RepeatRead>,
    failed: [bool; 3],
}

/// Runs the repeat-composition step on all reads, in parallel, preserving read order.
pub fn repeat_composition(reads: &[TelRead], kmers: &KmerTable, filters: EndFilters) -> RepeatComposition {
    let outcomes: Vec<ReadOutcome> = reads.par_iter().map(|r| analyze_read(r, kmers, filters)).collect();
    let mut comp = RepeatComposition::default();
    for o in outcomes {
        comp.removed_term_tel += usize::from(o.failed[0]);
        comp.removed_term_nontel += usize::from(o.failed[1]);
        comp.removed_term_subtel += usize::from(o.failed[2]);
        comp.reads.extend(o.kept);
    }
    comp
}

fn analyze_read(read: &TelRead, kmers: &KmerTable, filters: EndFilters) -> ReadOutcome {
    let min_tel_score = if filters.min_term_tel > 0 { filters.min_term_tel.min(MIN_TEL_SCORE) } else { MIN_TEL_SCORE };
    let fwd = covered_bases(&read.seq, &kmers.canonical);
    let rev = covered_bases(&read.seq, &kmers.canonical_rev);
    // put everything into q orientation
    let seq = if fwd > rev { revcomp(&read.seq) } else { read.seq.clone() };
    let len = seq.len() as i64;
    let all_kmers: Vec<&[u8]> = kmers.kmers.iter().map(|k| k.seq.as_slice()).collect();
    let all_kmers_rev: Vec<&[u8]> = kmers.kmers_rev.iter().map(Vec::as_slice).collect();
    let (term_tel, nontel_end) = terminating_tel_q(&seq, &all_kmers, &all_kmers_rev, min_tel_score);
    let term_tel = term_tel.min(len);
    let nontel_end = nontel_end.min(len);

    let failed = [
        filters.min_term_tel > 0 && term_tel < filters.min_term_tel,
        filters.max_term_nontel > 0 && nontel_end > filters.max_term_nontel,
        filters.min_term_subtel > 0 && len < term_tel + filters.min_term_subtel,
    ];
    if failed.iter().any(|&f| f) {
        return ReadOutcome { kept: None, failed };
    }
    let subtel_end = (len - term_tel - filters.min_term_subtel).max(0) as usize;
    let mut tvrtel = &seq[subtel_end.min(seq.len())..];
    // no terminating tel at all: pretend the entire read is tvr + tel
    if tvrtel.is_empty() {
        tvrtel = &seq;
    }
    let hits = nonoverlapping_kmer_hits(tvrtel, &all_kmers_rev, &kmers.is_substring);
    let kept = RepeatRead {
        hits,
        tvrtel_len: tvrtel.len(),
        name: read.name.split(' ').next().unwrap_or_default().to_string(),
        seq,
    };
    ReadOutcome { kept: Some(kept), failed }
}

/// Marks every base covered by a (non-overlapping, leftmost) hit of any of `kmers`.
fn mark_hits(seq: &[u8], kmers: &[impl AsRef<[u8]>]) -> Vec<u8> {
    let mut marks = vec![0u8; seq.len()];
    for k in kmers {
        let k = k.as_ref();
        for start in memmem::find_iter(seq, k) {
            marks[start..start + k.len()].fill(1);
        }
    }
    marks
}

/// `get_telomere_base_count`: bases covered by hits of any of `kmers`.
fn covered_bases(seq: &[u8], kmers: &[Vec<u8>]) -> usize {
    mark_hits(seq, kmers).iter().map(|&m| m as usize).sum()
}

/// `get_telomere_kmer_density`: fraction of covered bases in each sliding window.
fn kmer_density(seq: &[u8], kmers: &[&[u8]]) -> Vec<f64> {
    let marks = mark_hits(seq, kmers);
    let mut cum = Vec::with_capacity(marks.len());
    let mut total = 0u32;
    for m in marks {
        total += m as u32;
        cum.push(total);
    }
    let w = TEL_WINDOW_SIZE;
    (0..seq.len().saturating_sub(w)).map(|i| f64::from(cum[i + w] - cum[i]) / w as f64).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    P,
    Q,
}

#[derive(Clone, Copy, Debug)]
struct Region {
    start: usize,
    end: Option<usize>,
    arm: Option<Arm>,
}

/// `get_telomere_regions`: segment the smoothed p-minus-q density into p, q and
/// non-telomere runs.
fn telomere_regions(p: &[f64], q: &[f64]) -> Vec<Region> {
    let w = TEL_WINDOW_SIZE;
    let n = p.len().max(q.len());
    if n < w {
        return vec![Region { start: 0, end: Some(0), arm: None }];
    }
    let mut power = vec![0.0; n];
    for i in 0..p.len().min(q.len()) {
        power[i] = p[i] - q[i];
    }
    let mut power = wavelet::smooth(&power);
    power.truncate(power.len().saturating_sub(w));
    if power.is_empty() {
        return vec![Region { start: 0, end: Some(0), arm: None }];
    }
    let classify = |v: f64| {
        if v >= P_VS_Q_AMP_THRESH {
            Some(Arm::P)
        } else if v <= -P_VS_Q_AMP_THRESH {
            Some(Arm::Q)
        } else {
            None
        }
    };
    let mut regions = vec![Region { start: 0, end: None, arm: classify(power[0]) }];
    for (i, &v) in power.iter().enumerate().skip(1) {
        let last = regions.last_mut().expect("regions is never empty");
        let continues = match last.arm {
            Some(Arm::P) => v >= P_VS_Q_AMP_THRESH,
            Some(Arm::Q) => v <= -P_VS_Q_AMP_THRESH,
            None => v.abs() < P_VS_Q_AMP_THRESH,
        };
        if continues {
            last.end = Some(i + 1);
        } else {
            if last.end.is_none() {
                // previous region was only 1bp wide
                last.end = Some(last.start + 1);
            }
            regions.push(Region { start: i, end: None, arm: classify(v) });
        }
    }
    let last = regions.last_mut().expect("regions is never empty");
    if last.end.is_none() {
        last.end = Some(power.len() - 1);
    }
    regions
}

/// `get_terminating_tl` for a read in q orientation: (telomere length, non-telomere bases
/// at the read end), or (0, 0) if the read does not end in telomere.
fn terminating_tel_q(seq: &[u8], kmers: &[&[u8]], kmers_rev: &[&[u8]], min_tel_score: i64) -> (i64, i64) {
    let window_adj = (TEL_WINDOW_SIZE / 2) as i64;
    let p = kmer_density(seq, kmers);
    let q = kmer_density(seq, kmers_rev);
    let regions = telomere_regions(&p, &q);
    let span = |r: &Region| r.end.expect("all region ends are set") as i64 - r.start as i64;
    let last = regions.last().expect("regions is never empty");
    let edge_nontel = if last.arm.is_none() { span(last) } else { 0 };
    let scores: Vec<i64> = regions.iter().map(|r| if r.arm.is_some() { span(r) } else { -span(r) }).collect();
    // reverse cumulative sum, then the first maximum
    let mut cum = vec![0i64; scores.len()];
    let mut acc = 0;
    for i in (0..scores.len()).rev() {
        acc += scores[i];
        cum[i] = acc;
    }
    let mut max_i = 0;
    for (i, &c) in cum.iter().enumerate() {
        if c > cum[max_i] {
            max_i = i;
        }
    }
    if cum[max_i] < min_tel_score {
        return (0, 0);
    }
    let tel_len = last.end.expect("all region ends are set") as i64 - regions[max_i].start as i64 + window_adj;
    if tel_len <= 0 { (0, 0) } else { (tel_len, edge_nontel) }
}

/// `get_nonoverlapping_kmer_hits`: repeat hits per kmer, with hits inside a longer kmer's
/// hit removed, adjacent hits merged into blocks, and hits truncated where another kmer's
/// hit begins.
pub fn nonoverlapping_kmer_hits(seq: &[u8], kmers: &[&[u8]], is_substring: &[Vec<usize>]) -> Vec<Vec<Span>> {
    let raw: Vec<Vec<Span>> =
        kmers.iter().map(|k| memmem::find_iter(seq, k).map(|s| (s, s + k.len())).collect()).collect();
    // A span overlaps one of `hits` (sorted, disjoint) if some hit starts before it ends
    // and ends after it starts.
    let overlaps = |hits: &[Span], (s, e): Span| {
        let first_ending_after = hits.partition_point(|&(_, he)| he <= s);
        hits.get(first_ending_after).is_some_and(|&(hs, _)| hs < e)
    };
    let mut out: Vec<Vec<Span>> = raw
        .iter()
        .enumerate()
        .map(|(i, hits)| {
            hits.iter().copied().filter(|&span| !is_substring[i].iter().any(|&j| overlaps(&raw[j], span))).collect()
        })
        .collect();
    for spans in &mut out {
        let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
        for &(s, e) in spans.iter() {
            match merged.last_mut() {
                Some(last) if last.1 == s => last.1 = e,
                _ => merged.push((s, e)),
            }
        }
        *spans = merged;
    }
    // Truncate hits overlapping the start of another kmer's hit. Starts never change, so
    // for each hit only the first other-kmer start after its own start can cut it.
    let starts: Vec<Vec<usize>> = out.iter().map(|v| v.iter().map(|&(s, _)| s).collect()).collect();
    for (i, spans) in out.iter_mut().enumerate() {
        for (j, starts_j) in starts.iter().enumerate() {
            if j == i {
                continue;
            }
            for span in spans.iter_mut() {
                let next = starts_j.partition_point(|&s| s <= span.0);
                if let Some(&s) = starts_j.get(next).filter(|&&s| s < span.1) {
                    span.1 = s;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_matches_python_order() {
        // tg_kmer.get_nonoverlapping_kmer_hits('CCCTAACCCTGACCCTAA', ['CCCTGA', 'CCCTAA', 'CCTA'], [[], [], [1]])
        let kmers: Vec<&[u8]> = vec![b"CCCTGA", b"CCCTAA", b"CCTA"];
        let hits = nonoverlapping_kmer_hits(b"CCCTAACCCTGACCCTAA", &kmers, &[vec![], vec![], vec![1]]);
        assert_eq!(hits, vec![vec![(6, 12)], vec![(0, 6), (12, 18)], vec![]]);
    }
}
