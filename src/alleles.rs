//! Allele rows of `tlens_by_allele.tsv` (`get_allele_tsv_dat`, `merge_allele_tsv_dat`),
//! telomere-length summaries, and the QC metrics in `qc/stats.tsv`.

use std::fmt::Write as _;

use crate::cli::TlMethod;
use crate::numpy;
use crate::util;

/// Mapping quality given to every read before anchoring.
pub const DUMMY_TEL_MAPQ: u8 = 60;

/// One output allele. Fields are kept as the strings Telogator2 writes, since several
/// steps edit them as comma-separated text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlleleRow {
    pub chr: String,
    pub position: String,
    pub ref_samp: String,
    pub allele_id: String,
    pub tl: String,
    pub read_tls: String,
    pub read_lengths: String,
    pub read_mapq: String,
    pub tvr_len: String,
    pub tvr_consensus: String,
    pub supporting_reads: String,
}

impl AlleleRow {
    pub fn tsv_header(method: &str) -> String {
        format!(
            "#chr\tposition\tref_samp\tallele_id\tTL_{method}\tread_TLs\tread_lengths\tread_mapq\ttvr_len\ttvr_consensus\tsupporting_reads\n"
        )
    }

    pub fn tsv_line(&self) -> String {
        let f = [
            &self.chr,
            &self.position,
            &self.ref_samp,
            &self.allele_id,
            &self.tl,
            &self.read_tls,
            &self.read_lengths,
            &self.read_mapq,
            &self.tvr_len,
            &self.tvr_consensus,
            &self.supporting_reads,
        ];
        let mut line = f.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\t");
        line.push('\n');
        line
    }

    /// First entry of a comma-separated field.
    pub fn first(field: &str) -> &str {
        field.split(',').next().unwrap_or_default()
    }
}

/// `choose_tl_from_observations` on a non-empty, ascending list of read TLs.
pub fn choose_tl(tls: &[i64], method: TlMethod) -> i64 {
    let v: Vec<f64> = tls.iter().map(|&x| x as f64).collect();
    let value = match method {
        TlMethod::Mean => numpy::mean(&v),
        TlMethod::Median => numpy::median(&v),
        TlMethod::Max => v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        TlMethod::Percentile(p) => numpy::percentile(&v, f64::from(p)),
        TlMethod::FromTop(k) => {
            let k = k.min(tls.len());
            // atls[-k], where atls[-0] is atls[0]
            v[if k == 0 { 0 } else { tls.len() - k }]
        }
    };
    value as i64
}

/// One read of an allele: (TL, read length, name).
#[derive(Clone, Debug)]
pub struct AlleleRead {
    pub tl: i64,
    pub read_len: usize,
    pub name: String,
}

/// `get_allele_tsv_dat` for one cluster: TLs are the TVR + telomere region minus the
/// consensus TVR. Returns `None` if the cluster has fewer than `min_reads` reads.
pub fn allele_row(mut reads: Vec<AlleleRead>, tvr: &[u8], method: TlMethod, min_reads: usize) -> Option<AlleleRow> {
    if reads.len() < min_reads {
        return None;
    }
    // sorted((tl, read_len, name, mapq)); mapq is the same for every read here
    reads.sort_by(|a, b| (a.tl, a.read_len, &a.name).cmp(&(b.tl, b.read_len, &b.name)));
    let tls: Vec<i64> = reads.iter().map(|r| r.tl - tvr.len() as i64).collect();
    let join = |it: &mut dyn Iterator<Item = String>| it.collect::<Vec<_>>().join(",");
    Some(AlleleRow {
        chr: util::UNCLUST_CHR.to_string(),
        position: "0".to_string(),
        ref_samp: "-".to_string(),
        allele_id: "0".to_string(),
        tl: choose_tl(&tls, method).to_string(),
        read_tls: join(&mut tls.iter().map(i64::to_string)),
        read_lengths: join(&mut reads.iter().map(|r| r.read_len.to_string())),
        read_mapq: join(&mut reads.iter().map(|_| DUMMY_TEL_MAPQ.to_string())),
        tvr_len: tvr.len().to_string(),
        tvr_consensus: String::from_utf8_lossy(tvr).into_owned(),
        supporting_reads: join(&mut reads.iter().map(|r| r.name.clone())),
    })
}

/// One read when merging alleles: (TL, read length, mapq, name), the last three as text.
type MergedRead = (i64, String, String, String);

/// `merge_allele_tsv_dat`: pool two alleles' reads, keeping the identity of the one with
/// more reads. Returns which input (0 or 1) the merged row replaces.
pub fn merge_rows(a: &AlleleRow, b: &AlleleRow, method: TlMethod) -> (usize, AlleleRow) {
    let split = |s: &str| s.split(',').map(str::to_string).collect::<Vec<_>>();
    let tls = |r: &AlleleRow| -> Vec<i64> {
        r.read_tls.split(',').map(|x| x.parse().expect("read TLs are integers")).collect()
    };
    let (tl_a, tl_b) = (tls(a), tls(b));
    let (keep, other, which) = if tl_b.len() > tl_a.len() { (b, a, 1) } else { (a, b, 0) };
    // sorted(zip(tls, read_lengths, mapqs, names)); the string columns compare as strings
    let mut reads: Vec<MergedRead> = Vec::new();
    for (r, t) in [(a, &tl_a), (b, &tl_b)] {
        let (lens, mapqs, names) = (split(&r.read_lengths), split(&r.read_mapq), split(&r.supporting_reads));
        for k in 0..t.len() {
            reads.push((t[k], lens[k].clone(), mapqs[k].clone(), names[k].clone()));
        }
    }
    reads.sort();
    let sorted_tls: Vec<i64> = reads.iter().map(|r| r.0).collect();
    let join = |f: &dyn Fn(&MergedRead) -> String| reads.iter().map(f).collect::<Vec<_>>().join(",");
    let merged = AlleleRow {
        chr: keep.chr.clone(),
        position: keep.position.clone(),
        ref_samp: keep.ref_samp.clone(),
        allele_id: format!("{},{}", keep.allele_id, other.allele_id),
        tl: choose_tl(&sorted_tls, method).to_string(),
        read_tls: join(&|r| r.0.to_string()),
        read_lengths: join(&|r| r.1.clone()),
        read_mapq: join(&|r| r.2.clone()),
        tvr_len: keep.tvr_len.clone(),
        tvr_consensus: keep.tvr_consensus.clone(),
        supporting_reads: join(&|r| r.3.clone()),
    };
    (which, merged)
}

/// Read counts of alleles excluded from the final set, by reason.
#[derive(Clone, Copy, Debug, Default)]
pub struct FinalFilterCounts {
    pub pass_alleles: usize,
    pub pass_reads: usize,
    pub unmapped_alleles: usize,
    pub unmapped_reads: usize,
    pub short_alleles: usize,
    pub short_reads: usize,
    pub interstitial_alleles: usize,
    pub interstitial_reads: usize,
}

/// The final-allele tally Telogator2 computes while plotting `all_final_alleles.png`.
pub fn final_filter_counts(rows: &[AlleleRow], min_atl: i64) -> FinalFilterCounts {
    let mut c = FinalFilterCounts::default();
    for r in rows {
        let n_reads = r.read_lengths.split(',').count();
        let rep_atl: i64 = r.tl.parse().expect("TL is an integer");
        let tvr_len: i64 = r.tvr_len.parse().expect("tvr_len is an integer");
        if r.chr == util::UNCLUST_CHR || r.chr == util::BLANK_CHR {
            c.unmapped_alleles += 1;
            c.unmapped_reads += n_reads;
        } else if tvr_len + rep_atl < min_atl {
            c.short_alleles += 1;
            c.short_reads += n_reads;
        } else if r.allele_id.contains('i') {
            c.interstitial_alleles += 1;
            c.interstitial_reads += n_reads;
        } else {
            c.pass_alleles += 1;
            c.pass_reads += n_reads;
        }
    }
    c
}

/// `qc/stats.tsv` contents (`parse_tsv` over the final table plus the counts above).
pub fn stats_tsv(rows: &[AlleleRow], min_reads: usize, min_maxatl: i64, counts: &FinalFilterCounts) -> String {
    const MIN_ATL: i64 = -2000;
    const MAX_ATL: i64 = 20000;
    let mut allele_tls: Vec<i64> = Vec::new();
    let mut readlens: Vec<i64> = Vec::new();
    let mut fails = 0;
    for r in rows {
        if r.allele_id.split(',').any(|a| a.ends_with('i')) {
            fails += 1;
            continue;
        }
        let tls: Vec<i64> = r
            .read_tls
            .split(',')
            .map(|x| x.parse().expect("read TLs are integers"))
            .filter(|&x| x < MAX_ATL && x > MIN_ATL)
            .collect();
        if tls.is_empty() || tls.iter().max().is_some_and(|&m| m < min_maxatl) || tls.len() < min_reads {
            fails += 1;
            continue;
        }
        allele_tls.push(r.tl.parse().expect("TL is an integer"));
        readlens.extend(r.read_lengths.split(',').map(|x| x.parse::<i64>().expect("read lengths are integers")));
    }
    let as_f64 = |v: &[i64]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let mean_or_zero = |v: &[i64]| if v.is_empty() { 0 } else { numpy::mean(&as_f64(v)) as i64 };
    let metrics: [(&str, i64); 11] = [
        ("num_alleles", allele_tls.len() as i64),
        ("num_alleles_fail", fails),
        ("tl_mean", mean_or_zero(&allele_tls)),
        ("tl_median", if allele_tls.is_empty() { 0 } else { numpy::median(&as_f64(&allele_tls)) as i64 }),
        ("tl_short", allele_tls.iter().filter(|&&t| t < 1000).count() as i64),
        ("num_telreads", readlens.len() as i64),
        ("telread_mean_length", mean_or_zero(&readlens)),
        ("telread_n50_length", if readlens.is_empty() { 0 } else { util::n50(&readlens) }),
        ("fail_reads_unmapped", counts.unmapped_reads as i64),
        ("fail_reads_maxatl", counts.short_reads as i64),
        ("fail_reads_interstitial", counts.interstitial_reads as i64),
    ];
    let mut out = String::new();
    let names: Vec<&str> = metrics.iter().map(|m| m.0).collect();
    let values: Vec<String> = metrics.iter().map(|m| m.1.to_string()).collect();
    let _ = writeln!(out, "{}", names.join("\t"));
    let _ = writeln!(out, "{}", values.join("\t"));
    out
}

/// `write_blank_stats_tsv`.
pub fn blank_stats_tsv() -> String {
    stats_tsv(&[], 1, 0, &FinalFilterCounts::default())
}
