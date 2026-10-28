//! Stage 6: align each allele's subtelomeres to the Telogator reference and assign the
//! allele to the chromosome arm(s) its reads map to.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use rust_htslib::bam::{self, Read as _, record::Cigar};

use crate::alleles::AlleleRow;
use crate::cli::ReadType;
use crate::numpy;
use crate::util;

/// Share of the anchoring score an arm needs to be reported.
const ANCHORING_ASSIGNMENT_FRAC: f64 = 0.150;
const MAX_QUAL_SCORE: f64 = 60.0;

/// The external aligner used for anchoring.
#[derive(Clone, Debug)]
pub enum Aligner {
    Minimap2(PathBuf),
    Winnowmap { exe: PathBuf, k15: PathBuf },
    Pbmm2(PathBuf),
}

impl Aligner {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Minimap2(_) => "minimap2",
            Self::Winnowmap { .. } => "winnowmap",
            Self::Pbmm2(_) => "pbmm2",
        }
    }

    pub fn exe(&self) -> &Path {
        match self {
            Self::Minimap2(e) | Self::Pbmm2(e) | Self::Winnowmap { exe: e, .. } => e,
        }
    }
}

/// Paths for the anchoring step, under `temp/06_anchoring/`.
#[derive(Clone, Debug)]
pub struct AnchorPaths {
    pub dir: PathBuf,
    pub unanchored: PathBuf,
    pub sam: PathBuf,
    pub bam: PathBuf,
    pub log: PathBuf,
}

impl AnchorPaths {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            unanchored: dir.join("unanchored_subtels.fa.gz"),
            sam: dir.join("subtel_aln.sam"),
            bam: dir.join("subtel_aln.bam"),
            log: dir.join("subtel_aln.log"),
            dir,
        }
    }
}

/// minimap2 or winnowmap mapping `queries` (a path, or `-` for stdin) to `reference`;
/// `None` for pbmm2.
fn mapping_command(
    aligner: &Aligner,
    read_type: ReadType,
    reference: &Path,
    threads: usize,
    sam: &Path,
    queries: &Path,
) -> Option<Command> {
    let threads = threads.max(1).to_string();
    let (exe, preset, k15) = match aligner {
        Aligner::Minimap2(exe) => (exe, if read_type == ReadType::Ont { "map-ont" } else { "map-hifi" }, None),
        Aligner::Winnowmap { exe, k15 } => {
            (exe, if read_type == ReadType::Ont { "map-ont" } else { "map-pb" }, Some(k15))
        }
        Aligner::Pbmm2(_) => return None,
    };
    let mut c = Command::new(exe);
    if let Some(k15) = k15 {
        c.arg("-W").arg(k15);
    }
    c.args(["-ax", preset, "-Y", "-t", &threads, "-o"]).arg(sam).arg(reference).arg(queries);
    Some(c)
}

/// minimap2 or winnowmap started ahead of the anchoring step, reading its queries from
/// stdin: it builds the reference index straight away, while the pipeline is still
/// clustering, and then waits. Dropping it unused stops the process.
#[derive(Debug)]
pub struct StartedAligner {
    child: Option<Child>,
}

impl StartedAligner {
    /// Starts the aligner, or returns `None` for pbmm2 (which runs at the anchoring step).
    ///
    /// # Errors
    ///
    /// Fails if the log file cannot be created or the aligner cannot be started.
    pub fn start(
        aligner: &Aligner,
        read_type: ReadType,
        reference: &Path,
        threads: usize,
        paths: &AnchorPaths,
    ) -> Result<Option<Self>> {
        let Some(mut cmd) = mapping_command(aligner, read_type, reference, threads, &paths.sam, Path::new("-")) else {
            return Ok(None);
        };
        let log = File::create(&paths.log).with_context(|| format!("creating {}", paths.log.display()))?;
        let child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .with_context(|| format!("running {}", aligner.exe().display()))?;
        Ok(Some(Self { child: Some(child) }))
    }

    /// Streams the queries to the aligner and waits for it to finish.
    fn finish(mut self, subtels: &[(String, Vec<u8>)]) -> Result<std::process::ExitStatus> {
        let mut child = self.child.take().expect("aligner is only finished once");
        {
            let stdin = child.stdin.take().expect("aligner stdin is piped");
            let mut w = BufWriter::new(stdin);
            // An aligner that already failed closes its stdin; its exit status says why.
            let _ = write_fasta(&mut w, subtels).and_then(|()| Ok(w.flush()?));
        }
        Ok(child.wait()?)
    }
}

impl Drop for StartedAligner {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// `>name\nseq\n` for each non-empty subtelomere.
fn write_fasta(w: &mut impl Write, subtels: &[(String, Vec<u8>)]) -> Result<()> {
    for (name, seq) in subtels.iter().filter(|(_, s)| !s.is_empty()) {
        writeln!(w, ">{name}")?;
        w.write_all(seq)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Writes the subtelomere FASTA, runs the aligner (or feeds the one already `started`)
/// and returns its alignments in coordinate-sorted order (also saved as `subtel_aln.bam`).
///
/// # Errors
///
/// Fails if the aligner cannot be run or produces no alignments file.
#[allow(clippy::too_many_arguments)]
pub fn align_subtels(
    subtels: &[(String, Vec<u8>)],
    started: Option<StartedAligner>,
    aligner: &Aligner,
    read_type: ReadType,
    reference: &Path,
    threads: usize,
    paths: &AnchorPaths,
    temp_dir: &Path,
) -> Result<Vec<bam::Record>> {
    {
        let file =
            File::create(&paths.unanchored).with_context(|| format!("creating {}", paths.unanchored.display()))?;
        let mut enc = GzEncoder::new(BufWriter::new(file), Compression::new(6));
        write_fasta(&mut enc, subtels)?;
        enc.finish()?.flush()?;
    }
    let status = match (started, aligner) {
        (Some(started), _) => started.finish(subtels)?,
        (None, Aligner::Pbmm2(exe)) => {
            // pbmm2 wants an uncompressed, indexed reference
            let plain = temp_dir.join("telogator-ref.fa");
            decompress_reference(reference, &plain)?;
            let mut c = Command::new(exe);
            c.arg("align").arg(&plain).arg(&paths.unanchored).arg(&paths.bam);
            c.args(["--preset", "HiFi", "--sort", "-j", &threads.max(1).to_string()]);
            let status = run_logged(c, aligner, paths);
            let _ = fs::remove_file(&plain);
            let _ = fs::remove_file(plain.with_extension("fa.fai"));
            status?
        }
        (None, _) => {
            let c = mapping_command(aligner, read_type, reference, threads, &paths.sam, &paths.unanchored)
                .expect("minimap2 and winnowmap have a mapping command");
            run_logged(c, aligner, paths)?
        }
    };
    if !status.success() {
        bail!("alignment command returned an error ({status}); see {}", paths.log.display());
    }
    let records = match aligner {
        Aligner::Pbmm2(_) => read_records(&paths.bam)?,
        _ => {
            if !paths.sam.exists() {
                bail!("subtel alignment failed. Check the log: {}", paths.log.display());
            }
            let (header, mut records) = read_sam(&paths.sam)?;
            sort_like_samtools(&mut records);
            write_sorted_bam(&paths.bam, &header, &records)?;
            let _ = fs::remove_file(&paths.sam);
            records
        }
    };
    Ok(records)
}

fn run_logged(mut cmd: Command, aligner: &Aligner, paths: &AnchorPaths) -> Result<std::process::ExitStatus> {
    let log = File::create(&paths.log).with_context(|| format!("creating {}", paths.log.display()))?;
    cmd.stdout(Stdio::null()).stderr(log).status().with_context(|| format!("running {}", aligner.exe().display()))
}

fn decompress_reference(gz: &Path, out: &Path) -> Result<()> {
    let mut text = Vec::new();
    MultiGzDecoder::new(File::open(gz).with_context(|| format!("opening {}", gz.display()))?).read_to_end(&mut text)?;
    fs::write(out, &text)?;
    rust_htslib::faidx::Reader::from_path(out).with_context(|| format!("indexing {}", out.display()))?;
    Ok(())
}

fn read_records(path: &Path) -> Result<Vec<bam::Record>> {
    let mut reader = bam::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for rec in reader.records() {
        out.push(rec?);
    }
    Ok(out)
}

fn read_sam(path: &Path) -> Result<(bam::Header, Vec<bam::Record>)> {
    let mut reader = bam::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let header = bam::Header::from_template(reader.header());
    let mut out = Vec::new();
    for rec in reader.records() {
        out.push(rec?);
    }
    Ok((header, out))
}

/// `samtools sort` coordinate order: reference, position, then strand, keeping input
/// order for ties; unmapped reads (no reference) go last.
fn sort_like_samtools(records: &mut [bam::Record]) {
    records.sort_by_key(|r| {
        let tid = if r.tid() < 0 { u32::MAX } else { r.tid() as u32 };
        (tid, r.pos() + 1, r.is_reverse())
    });
}

fn write_sorted_bam(path: &Path, header: &bam::Header, records: &[bam::Record]) -> Result<()> {
    {
        let mut writer = bam::Writer::from_path(path, header, bam::Format::Bam)
            .with_context(|| format!("creating {}", path.display()))?;
        for r in records {
            writer.write(r)?;
        }
    }
    bam::index::build(path, None, bam::index::Type::Bai, 1).with_context(|| format!("indexing {}", path.display()))?;
    Ok(())
}

/// The fields of one alignment Telogator2 keeps (`parse_read`).
#[derive(Clone, Debug)]
struct Alignment {
    read_pos: (i64, i64),
    reference: String,
    ref_pos: (i64, i64),
    mapq: i64,
    /// Length of `str(query_sequence)` (`'None'` when SEQ is `*`).
    seq_len: i64,
}

fn parse_alignment(rec: &bam::Record, ref_names: &[String]) -> Alignment {
    let seq_len = if rec.seq_len() == 0 { 4 } else { rec.seq_len() as i64 };
    let reference = if rec.tid() >= 0 { ref_names[rec.tid() as usize].clone() } else { "*".to_string() };
    if reference == "*" || rec.is_unmapped() {
        return Alignment {
            read_pos: (0, seq_len),
            reference: "*".to_string(),
            ref_pos: (0, seq_len),
            mapq: 0,
            seq_len,
        };
    }
    let cigar = rec.cigar();
    let ops: Vec<Cigar> = cigar.iter().copied().collect();
    let is_clip = |c: &Cigar| matches!(c, Cigar::SoftClip(_) | Cigar::HardClip(_));
    let start_clip = ops.first().filter(|c| is_clip(c)).map_or(0, |c| i64::from(c.len()));
    let end_clip = if ops.len() > 1 { ops.last().filter(|c| is_clip(c)).map_or(0, |c| i64::from(c.len())) } else { 0 };
    let (mut ref_adj, mut read_adj) = (0i64, 0i64);
    for c in &ops {
        let len = i64::from(c.len());
        match c {
            Cigar::Match(_) | Cigar::Diff(_) | Cigar::Equal(_) => {
                ref_adj += len;
                read_adj += len;
            }
            Cigar::Del(_) => ref_adj += len,
            Cigar::Ins(_) => read_adj += len,
            _ => {}
        }
    }
    let pos = rec.pos();
    let mut read_pos = (start_clip, start_clip + read_adj);
    let read_len = start_clip + read_adj + end_clip;
    let ref_pos = if rec.is_reverse() {
        read_pos = (read_len - read_pos.1, read_len - read_pos.0);
        (pos + ref_adj, pos)
    } else {
        (pos, pos + ref_adj)
    };
    Alignment { read_pos, reference, ref_pos, mapq: i64::from(rec.mapq()), seq_len }
}

/// Fills chromosome, position, reference build and mapping qualities of anchored alleles
/// (`alleles[k]` is allele `k`, as numbered in the subtelomere read names).
pub fn assign_anchors(records: &[bam::Record], ref_names: &[String], alleles: &mut [AlleleRow]) {
    // alignments grouped by read name, names in order of first appearance
    let mut order: Vec<String> = Vec::new();
    let mut by_name: HashMap<String, Vec<Alignment>> = HashMap::new();
    for rec in records {
        let name = String::from_utf8_lossy(rec.qname()).into_owned();
        let aln = parse_alignment(rec, ref_names);
        by_name
            .entry(name.clone())
            .or_insert_with(|| {
                order.push(name);
                Vec::new()
            })
            .push(aln);
    }
    let mut best_mapq: HashMap<String, i64> = HashMap::new();
    let mut top_by_cluster: HashMap<usize, Vec<Alignment>> = HashMap::new();
    for name in &order {
        if !name.starts_with("cluster") {
            continue;
        }
        let original = name.split('_').skip(2).collect::<Vec<_>>().join("_");
        let alns = &by_name[name];
        // sorted((read_span, mapq, -dist_to_end, i), reverse=True)[0]
        type Choice = (i64, i64, i64, usize);
        let mut best: Option<(Choice, &Alignment)> = None;
        for (i, a) in alns.iter().enumerate() {
            let dist_to_end = a.seq_len - a.read_pos.0.max(a.read_pos.1);
            let key = ((a.read_pos.1 - a.read_pos.0).abs(), a.mapq, -dist_to_end, i);
            if best.as_ref().is_none_or(|(k, _)| key > *k) {
                best = Some((key, a));
            }
            let e = best_mapq.entry(original.clone()).or_insert(a.mapq);
            *e = (*e).max(a.mapq);
        }
        let Some(cluster) = name.split('_').next().and_then(|s| s.get(8..)).and_then(|s| s.parse::<usize>().ok())
        else {
            continue;
        };
        let (_, top) = best.expect("every listed read has an alignment");
        top_by_cluster.entry(cluster).or_default().push(top.clone());
    }

    let mut clusters: Vec<usize> = top_by_cluster.keys().copied().collect();
    clusters.sort_unstable();
    for k in clusters {
        let mut arms: Vec<(String, f64, Vec<i64>)> = Vec::new();
        for a in &top_by_cluster[&k] {
            if a.reference == "*" {
                continue;
            }
            let span = (a.ref_pos.0 - a.ref_pos.1).abs() as f64;
            let weight = span * ((a.mapq as f64 + 1.0) / (MAX_QUAL_SCORE + 1.0));
            match arms.iter_mut().find(|(r, _, _)| *r == a.reference) {
                Some((_, score, anchors)) => {
                    *score += weight;
                    anchors.push(a.ref_pos.1);
                }
                None => arms.push((a.reference.clone(), weight, vec![a.ref_pos.1])),
            }
        }
        let total: f64 = arms.iter().map(|a| a.1).sum();
        if total <= 0.0 {
            continue;
        }
        let mut ranked: Vec<(f64, &str)> = arms.iter().map(|(r, s, _)| (s / total, r.as_str())).collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.cmp(a.1)));
        let (mut chrs, mut positions, mut builds) = (Vec::new(), Vec::new(), Vec::new());
        for (frac, arm) in ranked {
            if frac < ANCHORING_ASSIGNMENT_FRAC {
                continue;
            }
            let mut parts = arm.split('_');
            let samp = parts.next().unwrap_or_default().to_string();
            let chr = parts.next().unwrap_or_default().to_string();
            if chrs.contains(&chr) {
                continue;
            }
            let anchors = &arms.iter().find(|(r, _, _)| r == arm).expect("ranked arms come from arms").2;
            let v: Vec<f64> = anchors.iter().map(|&x| x as f64).collect();
            positions.push((numpy::median(&v) as i64).to_string());
            chrs.push(chr);
            builds.push(samp);
        }
        if chrs.is_empty() {
            continue;
        }
        let row = &mut alleles[k];
        let interstitial = (0..chrs.len()).any(|i| {
            util::is_interstitial_tel(&format!("{}_{}", builds[i], chrs[i]), positions[i].parse().unwrap_or(0))
        });
        row.chr = chrs.join(",");
        row.position = positions.join(",");
        row.ref_samp = builds.join(",");
        if interstitial {
            row.allele_id.push('i');
        }
        row.read_mapq = row
            .supporting_reads
            .split(',')
            .map(|n| best_mapq.get(n).copied().unwrap_or(0).to_string())
            .collect::<Vec<_>>()
            .join(",");
    }
}

/// Sort key for the final table: chromosome, arm, longest TL first, position.
pub fn sort_alleles(alleles: &mut Vec<AlleleRow>) {
    type Key = (u32, String, i64, i64, usize);
    let mut keyed: Vec<(Key, AlleleRow)> = alleles
        .drain(..)
        .enumerate()
        .map(|(i, r)| {
            let chr = AlleleRow::first(&r.chr).to_string();
            let (name, arm) = chr.split_at(chr.len().saturating_sub(1));
            let tl: i64 = r.tl.parse().expect("TL is an integer");
            let pos: i64 = AlleleRow::first(&r.position).parse().unwrap_or(0);
            ((util::chrom_rank(name), arm.to_string(), -tl, pos, i), r)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    alleles.extend(keyed.into_iter().map(|(_, r)| r));
}
