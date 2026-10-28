//! The Telogator2 workflow (`telogator2.py:main`), stage by stage.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::align::distance::{self, LengthAdjust, MAX_SEQ_DIST};
use crate::align::scoring::{GapKind, Scoring, Substitution};
use crate::alleles::{self, AlleleRead, AlleleRow};
use crate::anchor::{self, Aligner, AnchorPaths};
use crate::cli::{Args, TlMethod};
use crate::input::{Decoding, Gunzip, InputKind};
use crate::linkage::Method;
use crate::pyrandom::PyRandom;
use crate::resources;
use crate::screen;
use crate::telrep::{self, EndFilters, RepeatRead};
use crate::tvr::{self, Letters, ReadColors, TvrClusterParams};
use crate::util::{self, BLANK_CHR, UNCLUST_CHR};

const HUMAN_GENOME_BP: f64 = 3_100_000_000.0;

/// Prints a stage label and returns a timer; [`done`] prints the elapsed time.
fn stage(label: &str) -> Instant {
    print!("{label}...");
    let _ = std::io::stdout().flush();
    Instant::now()
}

fn done(t: Instant) {
    println!(" ({:.1} sec)", t.elapsed().as_secs_f64());
}

/// A group of reads (indices into the repeat-composition reads) on one chromosome label,
/// with each read's reversed subtelomere.
#[derive(Clone, Debug)]
struct ReadGroup {
    chr: &'static str,
    reads: Vec<usize>,
    subtels_rev: Vec<Vec<u8>>,
}

/// Writes distance matrices and consensus sequences under `$TELOCROC_DUMP`, named
/// like Telogator2's `temp/` files, for comparing runs.
struct Dumper(Option<PathBuf>);

impl Dumper {
    fn from_env() -> Self {
        Self(std::env::var_os("TELOCROC_DUMP").map(PathBuf::from))
    }

    fn write(&self, name: &str, bytes: &[u8]) {
        if let Some(dir) = &self.0 {
            let path = dir.join(name);
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(path, bytes);
        }
    }

    fn dist(&self, name: &str, dist: Option<&distance::DistMatrix>) {
        if let (Some(_), Some(d)) = (&self.0, dist) {
            self.write(&format!("{name}.f32"), &d.to_le_bytes());
        }
    }

    fn consensus(&self, name: &str, seqs: &[Vec<u8>]) {
        if self.0.is_some() {
            let mut text = Vec::new();
            for (i, s) in seqs.iter().enumerate() {
                text.extend_from_slice(format!(">tvr-{:03}\n", i + 1).as_bytes());
                text.extend_from_slice(s);
                text.push(b'\n');
            }
            self.write(name, &text);
        }
    }
}

/// Everything the clustering stages share.
struct Clusterer<'a> {
    reads: &'a [RepeatRead],
    colors: &'a [ReadColors],
    letters: &'a Letters,
    min_reads: usize,
    dump: Dumper,
}

impl Clusterer<'_> {
    fn cluster(&self, members: &[usize], params: TvrClusterParams) -> tvr::TvrClusters {
        let reads: Vec<&RepeatRead> = members.iter().map(|&i| &self.reads[i]).collect();
        let colors: Vec<&ReadColors> = members.iter().map(|&i| &self.colors[i]).collect();
        tvr::cluster_tvrs(&reads, &colors, self.letters, params)
    }

    /// The read's subtelomere (everything before its TVR + telomere), reversed.
    fn subtel_rev(&self, read: usize) -> Vec<u8> {
        let seq = &self.reads[read].seq;
        let len = seq.len().saturating_sub(self.colors[read].tvrtel_len);
        seq[..len].iter().rev().copied().collect()
    }
}

/// Output of [`tvr_refinement`].
#[derive(Default)]
struct Refined {
    unclustered: Vec<usize>,
    blank: Vec<usize>,
    groups: Vec<ReadGroup>,
    consensus_tvrs: Vec<Vec<u8>>,
}

/// `tvr_refinement`: re-cluster each cluster at finer TVR resolution; on the last
/// iteration emit clusters with their consensus TVR and subtelomeres.
fn tvr_refinement(
    ctx: &Clusterer<'_>,
    init: Vec<Vec<usize>>,
    iterations: &[(usize, TvrClusterParams)],
    message: &str,
    dump_dir: &str,
    print_stats: bool,
) -> Refined {
    let mut out = Refined::default();
    let mut current = init;
    for (k, &(iteration, params)) in iterations.iter().enumerate() {
        let t = stage(&format!("{message} [iteration {iteration}]"));
        let last = k + 1 == iterations.len();
        let params = TvrClusterParams { clustering_only: !last, ..params };
        let results: Vec<tvr::TvrClusters> = current.par_iter().map(|members| ctx.cluster(members, params)).collect();
        let mut next = Vec::new();
        for (ci, (members, result)) in current.iter().zip(&results).enumerate() {
            ctx.dump.dist(&format!("{dump_dir}/i{iteration}_dist-matrix_{ci:03}"), result.dist.as_ref());
            if last {
                ctx.dump.consensus(&format!("{dump_dir}/i{iteration}_consensus_{ci:03}.fa"), &result.consensus);
            }
            for sub in &result.clusters {
                let global: Vec<usize> = sub.iter().map(|&n| members[n]).collect();
                if global.len() >= ctx.min_reads {
                    next.push(global);
                } else {
                    out.unclustered.extend(global);
                }
            }
            if last {
                for (sci, sub) in result.clusters.iter().enumerate() {
                    let global: Vec<usize> = sub.iter().map(|&n| members[n]).collect();
                    let tvr_len = result.tvr_lens[sci];
                    if tvr_len <= 0 {
                        out.blank.extend(global);
                        continue;
                    }
                    if global.len() < ctx.min_reads {
                        continue;
                    }
                    let subtels_rev = global.iter().map(|&r| ctx.subtel_rev(r)).collect();
                    out.groups.push(ReadGroup { chr: UNCLUST_CHR, reads: global, subtels_rev });
                    let cons = &result.consensus[sci];
                    out.consensus_tvrs.push(cons[..(tvr_len as usize).min(cons.len())].to_vec());
                }
            }
        }
        current = next;
        done(t);
        if print_stats {
            println!(" - {} clusters ({} reads)", current.len(), current.iter().map(Vec::len).sum::<usize>());
            println!(" - {} reads with blank TVR", out.blank.len());
            println!(" - {} unclustered reads", out.unclustered.len());
        }
    }
    out
}

fn check_inputs(args: &Args) -> Result<()> {
    let mut any_cram = false;
    for p in &args.input {
        if !fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() > 0) {
            bail!("input not found: {}", p.display());
        }
        any_cram |= InputKind::of(p)? == InputKind::Cram;
    }
    let mut unique = args.input.clone();
    unique.sort();
    unique.dedup();
    if unique.len() < args.input.len() {
        bail!("an input file was provided multiple times");
    }
    if any_cram && args.cram_ref.is_none() {
        bail!("cram input requires reference fasta via --ref");
    }
    if args.min_reads < 1 {
        bail!("-n must be >= 1");
    }
    Ok(())
}

fn choose_aligner(args: &Args, k15: Option<PathBuf>) -> Result<Aligner> {
    let given = [&args.minimap2, &args.winnowmap, &args.pbmm2].iter().filter(|a| a.is_some()).count();
    if given > 1 {
        bail!("multiple aligners specified, use only one of --minimap2, --winnowmap, --pbmm2");
    }
    let aligner = if let Some(exe) = &args.winnowmap {
        let Some(k15) = args.winnowmap_k15.clone().or(k15) else {
            bail!("winnowmap was chosen as aligner but no --winnowmap-k15 was provided.");
        };
        Aligner::Winnowmap { exe: exe.clone(), k15 }
    } else if let Some(exe) = &args.pbmm2 {
        Aligner::Pbmm2(exe.clone())
    } else if let Some(exe) = &args.minimap2 {
        Aligner::Minimap2(exe.clone())
    } else {
        println!("Warning: no aligner was specified, trying \"minimap2\"...");
        Aligner::Minimap2(PathBuf::from("minimap2"))
    };
    if !resources::executable_exists(aligner.exe()) {
        bail!("{} executable not found.", aligner.name());
    }
    Ok(aligner)
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// Runs the whole pipeline.
///
/// # Errors
///
/// Fails on invalid options, unreadable inputs, or a failed anchoring alignment.
pub fn run(args: &Args, argv: &[String]) -> Result<()> {
    check_inputs(args)?;
    let method = args.tl_method()?;
    let out_dir = args.output.clone();
    let temp_dir = out_dir.join("temp");
    let qc_dir = out_dir.join("qc");
    let anchor_paths = AnchorPaths::new(temp_dir.join("06_anchoring"));
    for d in [&out_dir, &temp_dir, &qc_dir, &anchor_paths.dir] {
        fs::create_dir_all(d).with_context(|| format!("could not create output directory {}", d.display()))?;
    }
    let stats_path = qc_dir.join("stats.tsv");

    let rng_seed = if args.rng <= -1 { resources::random_seed() } else { args.rng as u64 };
    write_text(&qc_dir.join("rng.txt"), &format!("{rng_seed}\n"))?;
    let mut global_rng = PyRandom::new(rng_seed);

    let (tvr_shuffles, refine_iterations): ([usize; 3], Vec<usize>) =
        if args.fast_aln { ([1, 1, 1], vec![2]) } else { ([2, 3, 3], vec![1, 2]) };
    let tvr_params = [
        TvrClusterParams {
            tvr_truncate: 1000,
            shuffles: tvr_shuffles[0],
            tree_cut: args.treecut_iter0,
            clustering_only: true,
            rng_seed,
        },
        TvrClusterParams {
            tvr_truncate: 2000,
            shuffles: tvr_shuffles[1],
            tree_cut: args.treecut_iter1,
            clustering_only: true,
            rng_seed,
        },
        TvrClusterParams {
            tvr_truncate: 3000,
            shuffles: tvr_shuffles[2],
            tree_cut: args.treecut_iter2,
            clustering_only: false,
            rng_seed,
        },
    ];
    let final_params =
        TvrClusterParams { tvr_truncate: 1000, shuffles: 1, tree_cut: 100.0, clustering_only: false, rng_seed };
    let refine_schedule: Vec<(usize, TvrClusterParams)> =
        refine_iterations.iter().map(|&i| (i, tvr_params[i])).collect();

    let (telogator_ref, default_k15) = resources::telogator_ref(args.telogator_ref.as_deref())?;
    let aligner = choose_aligner(args, default_k15)?;
    let decoding = Decoding { cram_ref: args.cram_ref.clone(), gunzip: Gunzip::new(args.igzip.as_deref())? };
    let kmer_table = resources::kmer_table(args.kmers.as_deref(), args.read_type)?;
    let letters = Letters::new(&kmer_table);
    let initial_kmer = [kmer_table.canonical[0].as_slice(), kmer_table.canonical[0].as_slice()].concat();
    let initial_kmer_rc = [kmer_table.canonical_rev[0].as_slice(), kmer_table.canonical_rev[0].as_slice()].concat();

    let cmd: Vec<String> = argv.iter().map(|a| util::strip_path(a)).collect();
    write_text(&qc_dir.join("cmd.txt"), &format!("{}\n", cmd.join(" ")))?;
    // Index the reference in the background while reads are screened and clustered.
    let anchoring = !args.debug_noanchor && !args.debug_telreads;
    let mut started_aligner = if anchoring {
        anchor::StartedAligner::start(&aligner, args.read_type, &telogator_ref, args.threads, &anchor_paths)?
    } else {
        None
    };

    // [0a] telomere reads
    let tel_reads_path = temp_dir.join("tel_reads.fa.gz");
    let t = stage(&format!(
        "getting reads with at least {} matches to {}",
        args.min_canon_hits,
        String::from_utf8_lossy(&initial_kmer)
    ));
    let tmp_path = temp_dir.join("tel_reads.fa.gz.temp");
    let screened = screen::extract_telomere_reads(
        &args.input,
        &tmp_path,
        &initial_kmer,
        &initial_kmer_rc,
        args.min_canon_hits,
        args.threads,
        &decoding,
    )?;
    fs::rename(&tmp_path, &tel_reads_path)?;
    done(t);
    if screened.supplementary > 0 {
        println!(" - skipped {} supplementary alignments", screened.supplementary);
    }
    println!(" - {} --> {} reads", screened.all_reads, screened.reads.len());
    println!(
        " - ({} bp) [{}x] --> ({} bp)",
        screened.total_bp_all,
        (screened.total_bp_all as f64 / HUMAN_GENOME_BP + 0.5) as i64,
        screened.total_bp_tel
    );
    if screened.reads.is_empty() {
        println!("Warning: No telomere reads found, stopping here...");
        return write_text(&stats_path, &alleles::blank_stats_tsv());
    }
    if args.debug_telreads {
        println!("--debug-telreads was specified, stopping now.");
        return write_text(&stats_path, &alleles::blank_stats_tsv());
    }

    let t = stage(&format!("filtering by read length (>{}bp)", args.min_read_len));
    let methylated = screened.methylated;
    let (all_reads, filtered) = screen::dedup_by_name(screened.reads, args.min_read_len);
    done(t);
    println!(" - {} --> {} reads", all_reads.len() + filtered, all_reads.len());
    if all_reads.is_empty() {
        println!("Warning: No telomere reads remaining, stopping here...");
        return write_text(&stats_path, &alleles::blank_stats_tsv());
    }
    println!(" - {} reads with methylation data", all_reads.iter().filter(|r| methylated.contains(&r.name)).count());

    // [0b] telomere repeat composition
    let t = stage("getting telomere repeat composition");
    let filters =
        EndFilters { min_term_tel: args.filt_tel, max_term_nontel: args.filt_nontel, min_term_subtel: args.filt_sub };
    let composition = telrep::repeat_composition(&all_reads, &kmer_table, filters);
    let num_starting = all_reads.len();
    drop(all_reads);
    done(t);
    let mut reads = composition.reads;
    println!(" - {num_starting} --> {} reads", reads.len());
    if args.filt_tel > 0 {
        println!(
            " - ({} reads removed for ending in < {}bp tel sequence)",
            composition.removed_term_tel, args.filt_tel
        );
    }
    if args.filt_nontel > 0 {
        println!(
            " - ({} reads removed for ending in > {}bp non-tel sequence)",
            composition.removed_term_nontel, args.filt_nontel
        );
    }
    if args.filt_sub > 0 {
        println!(
            " - ({} reads removed for ending in < {}bp subtel sequence)",
            composition.removed_term_subtel, args.filt_sub
        );
    }
    if reads.is_empty() {
        println!("Warning: No telomere reads remaining, stopping here...");
        return write_text(&stats_path, &alleles::blank_stats_tsv());
    }
    if args.downsample > 0 && reads.len() > args.downsample as usize {
        let t = stage("downsampling reads");
        let before = reads.len();
        let mut inds: Vec<usize> = (0..before).collect();
        global_rng.shuffle(&mut inds);
        let mut drop_set = vec![false; before];
        for &i in &inds[..before - args.downsample as usize] {
            drop_set[i] = true;
        }
        let mut k = 0;
        reads.retain(|_| {
            k += 1;
            !drop_set[k - 1]
        });
        done(t);
        println!(" - {before} --> {} reads", reads.len());
    }
    let colors: Vec<ReadColors> = reads.par_iter().map(|r| tvr::read_colors(r, &letters)).collect();
    let ctx = Clusterer {
        reads: &reads,
        colors: &colors,
        letters: &letters,
        min_reads: args.min_reads,
        dump: Dumper::from_env(),
    };

    // [1a] initial clustering
    let t = stage("initial clustering of all reads [iteration 0]");
    let all: Vec<usize> = (0..reads.len()).collect();
    let init = ctx.cluster(&all, tvr_params[0]);
    ctx.dump.dist("01_initial/i0_dist-matrix", init.dist.as_ref());
    let (mut init_clusters, mut unclustered) = (Vec::new(), Vec::new());
    for c in init.clusters {
        if c.len() >= args.min_reads {
            init_clusters.push(c);
        } else {
            unclustered.extend(c);
        }
    }
    done(t);
    println!(" - {} clusters ({} reads)", init_clusters.len(), init_clusters.iter().map(Vec::len).sum::<usize>());
    println!(" - {} unclustered reads", unclustered.len());

    // [1b] TVR cluster refinement
    let refined =
        tvr_refinement(&ctx, init_clusters, &refine_schedule, "initial clustering of all reads", "01_initial", true);
    unclustered.extend(refined.unclustered);
    let (mut blank, mut groups, mut consensus_tvrs) = (refined.blank, refined.groups, refined.consensus_tvrs);

    // [2] salvage unclustered reads
    let salvaged =
        tvr_refinement(&ctx, vec![unclustered], &refine_schedule, "reanalyzing unclustered reads", "02_unclust", false);
    println!(" - {} new candidate clusters found", salvaged.groups.len());
    blank.extend(salvaged.blank);
    groups.extend(salvaged.groups);
    consensus_tvrs.extend(salvaged.consensus_tvrs);

    // [3] collapse clusters with similar TVRs
    let t = stage("collapsing duplicates");
    let (collapsed, collapse_dist) =
        tvr::cluster_sequences(&consensus_tvrs, None, args.treecut_collapse, Method::Single, rng_seed);
    ctx.dump.dist("03_collapse/tvrs_dist-matrix", Some(&collapse_dist));
    let before = groups.len();
    groups = collapsed
        .iter()
        .map(|members| ReadGroup {
            chr: UNCLUST_CHR,
            reads: members.iter().flat_map(|&g| groups[g].reads.iter().copied()).collect(),
            subtels_rev: members.iter().flat_map(|&g| groups[g].subtels_rev.iter().cloned()).collect(),
        })
        .collect();
    done(t);
    if before == groups.len() {
        println!(" - no duplicates found");
    } else {
        println!(" - {before} clusters --> {} clusters", groups.len());
    }
    if blank.len() >= args.min_reads {
        let subtels_rev = blank.iter().map(|&r| ctx.subtel_rev(r)).collect();
        groups.push(ReadGroup { chr: BLANK_CHR, reads: blank, subtels_rev });
        println!("appending cluster of reads with blank TVR...");
        println!(" - {} clusters", groups.len());
    }

    // [4] subtel cluster refinement
    let final_groups: Vec<ReadGroup> = if args.debug_nosubtel {
        println!("skipping subtel cluster refinement...");
        groups
    } else {
        let t = stage("refining clusters [SUBTEL]");
        let split: Vec<(Vec<Vec<usize>>, distance::DistMatrix)> = groups
            .par_iter()
            .map(|g| tvr::cluster_sequences(&g.subtels_rev, Some(3000), args.treecut_subtel, Method::Ward, rng_seed))
            .collect();
        let mut out = Vec::new();
        for (gi, (g, (subs, dist))) in groups.iter().zip(split).enumerate() {
            ctx.dump.dist(&format!("04_subtel/dist-matrix_{gi:03}"), Some(&dist));
            for sub in subs {
                if sub.len() >= args.min_reads {
                    out.push(ReadGroup {
                        chr: g.chr,
                        reads: sub.iter().map(|&n| g.reads[n]).collect(),
                        subtels_rev: sub.iter().map(|&n| g.subtels_rev[n].clone()).collect(),
                    });
                }
            }
        }
        done(t);
        println!(" - {} clusters ({} reads)", out.len(), out.iter().map(|g| g.reads.len()).sum::<usize>());
        out
    };

    // [5] final reclustering
    let t = stage("final reclustering to get TVRs and TLs for each allele");
    let finals: Vec<(Vec<usize>, tvr::TvrClusters)> = final_groups
        .par_iter()
        .map(|g| {
            let mut sorted = g.reads.clone();
            sorted.sort_unstable();
            let result = ctx.cluster(&sorted, final_params);
            (sorted, result)
        })
        .collect();
    let mut subtels_out: Vec<(String, Vec<u8>)> = Vec::new();
    let mut allele_rows: Vec<AlleleRow> = Vec::new();
    for (gi, (g, (sorted, result))) in final_groups.iter().zip(&finals).enumerate() {
        ctx.dump.dist(&format!("05_final/dist-matrix_{gi:03}"), result.dist.as_ref());
        ctx.dump.consensus(&format!("05_final/consensus_{gi:03}.fa"), &result.consensus);
        for (sci, sub) in result.clusters.iter().enumerate() {
            let tvr_len = result.tvr_lens[sci];
            let cons = &result.consensus[sci];
            let tvr: &[u8] = if tvr_len > 0 { &cons[..(tvr_len as usize).min(cons.len())] } else { &[] };
            let allele_reads: Vec<AlleleRead> = sub
                .iter()
                .map(|&n| {
                    let r = sorted[n];
                    let c = &colors[r];
                    AlleleRead {
                        tl: reads[r].tvrtel_len as i64 - c.subtel_len as i64 - c.err_end_len as i64,
                        read_len: reads[r].seq.len(),
                        name: reads[r].name.clone(),
                    }
                })
                .collect();
            let Some(row) = alleles::allele_row(allele_reads, tvr, method, args.min_reads) else {
                continue;
            };
            for (i, &n) in sub.iter().enumerate() {
                // Telogator2 pairs subtelomeres (in the group's order) with reads by their
                // position in the *sorted* read list; kept as is.
                let subtel: Vec<u8> =
                    g.subtels_rev.get(n).map(|s| s.iter().rev().copied().collect()).unwrap_or_default();
                subtels_out.push((format!("cluster-{}_read-{i}_{}", allele_rows.len(), reads[sorted[n]].name), subtel));
            }
            allele_rows.push(row);
        }
    }
    done(t);
    for (i, row) in allele_rows.iter_mut().enumerate() {
        row.allele_id = i.to_string();
    }

    // [6] subtel anchoring
    if !args.debug_noanchor {
        let t = stage("aligning subtels to reference");
        let records = anchor::align_subtels(
            &subtels_out,
            started_aligner.take(),
            &aligner,
            args.read_type,
            &telogator_ref,
            args.threads,
            &anchor_paths,
            &temp_dir,
        )?;
        done(t);
        println!("getting anchors from alignment...");
        let ref_names = resources::reference_names(&anchor_paths.bam)?;
        anchor::assign_anchors(&records, &ref_names, &mut allele_rows);
        anchor::sort_alleles(&mut allele_rows);
    }

    // [7] collapse similar alleles mapped near each other
    if args.collapse_hom > 0 {
        let t = stage(&format!("merging alleles mapped within {}bp of each other", args.collapse_hom));
        let before = allele_rows.len();
        collapse_homozygous(&mut allele_rows, args, method, &mut global_rng);
        done(t);
        println!(" - {before} --> {} alleles", allele_rows.len());
    }

    // [8] results
    for row in &mut allele_rows {
        row.chr = row.chr.replace(BLANK_CHR, UNCLUST_CHR);
    }
    println!("writing allele TL results to \"tlens_by_allele.tsv\"...");
    let mut tsv = AlleleRow::tsv_header(&args.tl_method);
    for row in &allele_rows {
        tsv.push_str(&row.tsv_line());
    }
    write_text(&out_dir.join("tlens_by_allele.tsv"), &tsv)?;
    let counts = alleles::final_filter_counts(&allele_rows, args.min_atl_final);
    if counts.pass_alleles == 0 {
        println!("no alleles to plot.");
    }
    println!(" - {} final telomere alleles ({} reads)", counts.pass_alleles, counts.pass_reads);
    println!(" - {} alleles unmapped ({} reads)", counts.unmapped_alleles, counts.unmapped_reads);
    println!(" - {} alleles with atl < {} bp ({} reads)", counts.short_alleles, args.min_atl_final, counts.short_reads);
    println!(" - {} interstitial alelles ({} reads)", counts.interstitial_alleles, counts.interstitial_reads);
    write_text(&stats_path, &alleles::stats_tsv(&allele_rows, args.min_reads, args.min_atl_final, &counts))
}

/// `quick_compare_tvrs`, drawing its shuffles from the run's global generator.
fn quick_compare_tvrs(a: &[u8], b: &[u8], rng: &mut PyRandom) -> f64 {
    let scoring = Scoring::new(Substitution::MatchMismatch, GapKind::Tvr, true);
    distance::tvr_distance(a, b, &scoring, LengthAdjust::None, 3, rng) / MAX_SEQ_DIST
}

/// Stage 7: merge alleles on the same arm and reference within `--collapse-hom` bp whose
/// TVRs are within `-th`, one pair at a time, restarting after each merge.
fn collapse_homozygous(rows: &mut Vec<AlleleRow>, args: &Args, method: TlMethod, rng: &mut PyRandom) {
    loop {
        let mut delete = None;
        'search: for i in 0..rows.len() {
            for j in i + 1..rows.len() {
                let (a, b) = (&rows[i], &rows[j]);
                let pos = |r: &AlleleRow| AlleleRow::first(&r.position).parse::<i64>().unwrap_or(0);
                if AlleleRow::first(&a.chr) != AlleleRow::first(&b.chr)
                    || AlleleRow::first(&a.ref_samp) != AlleleRow::first(&b.ref_samp)
                    || (pos(a) - pos(b)).abs() > args.collapse_hom
                {
                    continue;
                }
                let mut dist = args.treecut_hom * 2.0;
                if !a.tvr_consensus.is_empty() && !b.tvr_consensus.is_empty() {
                    dist = quick_compare_tvrs(a.tvr_consensus.as_bytes(), b.tvr_consensus.as_bytes(), rng);
                }
                if args.debug_collapse {
                    println!("{}\n{}\n", a.tvr_consensus, b.tvr_consensus);
                    println!(
                        "DEBUG: ({}) vs. ({}): {dist} <= {} : {}\n",
                        a.allele_id,
                        b.allele_id,
                        args.treecut_hom,
                        dist <= args.treecut_hom
                    );
                }
                if dist <= args.treecut_hom {
                    let (which, merged) = alleles::merge_rows(a, b, method);
                    if which == 0 {
                        rows[i] = merged;
                        delete = Some(j);
                    } else {
                        rows[j] = merged;
                        delete = Some(i);
                    }
                    break 'search;
                }
            }
        }
        match delete {
            Some(d) => {
                rows.remove(d);
            }
            None => break,
        }
    }
}
