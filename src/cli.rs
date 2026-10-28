//! Command-line options, mirroring Telogator2's `argparse` interface.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};

/// Sequencing platform, which picks the canonical repeats and aligner preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ReadType {
    Hifi,
    Ont,
}

/// How to summarise an allele's read telomere lengths (`-m`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TlMethod {
    Mean,
    Median,
    Max,
    /// `pNN`: the NNth percentile.
    Percentile(u32),
    /// `NNfromtop`: the NNth-largest read.
    FromTop(usize),
}

impl TlMethod {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "mean" => Self::Mean,
            "median" => Self::Median,
            "max" => Self::Max,
            _ if s.starts_with('p') && s[1..].parse::<u32>().is_ok() => Self::Percentile(s[1..].parse()?),
            _ if s.ends_with("fromtop") && s[..s.len() - 7].parse::<usize>().is_ok() => {
                Self::FromTop(s[..s.len() - 7].parse()?)
            }
            _ => bail!("-m must be mean, median, max, pNN or NNfromtop (got {s:?})"),
        })
    }
}

#[derive(Debug, Parser)]
#[command(name = "telocroc", version, about = "Fast allele-specific telomere length from long reads")]
pub struct Args {
    /// Input reads (fa / fa.gz / fq / fq.gz / bam / cram)
    #[arg(short = 'i', required = true, num_args = 1.., value_name = "INPUT")]
    pub input: Vec<PathBuf>,
    /// Path to output directory
    #[arg(short = 'o', value_name = "output/")]
    pub output: PathBuf,
    /// Telomere kmers file
    #[arg(short = 'k', value_name = "kmers.tsv")]
    pub kmers: Option<PathBuf>,
    /// Telogator reference fasta
    #[arg(short = 't', value_name = "telogator.fa")]
    pub telogator_ref: Option<PathBuf>,
    /// Read type
    #[arg(short = 'r', value_enum, default_value = "ont")]
    pub read_type: ReadType,
    /// Minimum read length
    #[arg(short = 'l', default_value_t = 4000)]
    pub min_read_len: usize,
    /// Minimum hits to tandem canonical kmer
    #[arg(short = 'c', default_value_t = 8)]
    pub min_canon_hits: usize,
    /// Minimum number of reads per cluster
    #[arg(short = 'n', default_value_t = 3)]
    pub min_reads: usize,
    /// Method for choosing ATL: mean / median / max / pNN / NNfromtop
    #[arg(short = 'm', default_value = "p75")]
    pub tl_method: String,
    /// Downsample to this many telomere reads
    #[arg(short = 'd', default_value_t = -1, allow_negative_numbers = true)]
    pub downsample: i64,
    /// Number of threads to use
    #[arg(short = 'p', default_value_t = 4)]
    pub threads: usize,

    /// [FILTERING] Remove reads that end in < this much tel
    #[arg(long = "filt-tel", default_value_t = 400)]
    pub filt_tel: i64,
    /// [FILTERING] Remove reads that end in > this much non-tel
    #[arg(long = "filt-nontel", default_value_t = 100)]
    pub filt_nontel: i64,
    /// [FILTERING] Remove reads that end in < this much subtel
    #[arg(long = "filt-sub", default_value_t = 1000)]
    pub filt_sub: i64,

    /// [TREECUT] TVR clustering (iteration 0)
    #[arg(long = "t0", default_value_t = 0.200)]
    pub treecut_iter0: f64,
    /// [TREECUT] TVR clustering (iteration 1)
    #[arg(long = "t1", default_value_t = 0.150)]
    pub treecut_iter1: f64,
    /// [TREECUT] TVR clustering (iteration 2)
    #[arg(long = "t2", default_value_t = 0.100)]
    pub treecut_iter2: f64,
    /// [TREECUT] TVR clustering (collapse)
    #[arg(long = "tc", default_value_t = 0.050)]
    pub treecut_collapse: f64,
    /// [TREECUT] Subtel cluster refinement
    #[arg(long = "ts", default_value_t = 0.200)]
    pub treecut_subtel: f64,
    /// [TREECUT] Collapsing aligned alleles
    #[arg(long = "th", default_value_t = 0.050)]
    pub treecut_hom: f64,

    /// Minimum (TVR + TL) for an allele to count as a final allele
    #[arg(long = "afa-a", default_value_t = 100)]
    pub min_atl_final: i64,
    /// Accepted for compatibility; plots are not drawn
    #[arg(long = "afa-x", hide = true)]
    pub afa_x: Option<i64>,
    /// Accepted for compatibility; plots are not drawn
    #[arg(long = "afa-t", hide = true)]
    pub afa_t: Option<i64>,
    /// Accepted for compatibility; plots are not drawn
    #[arg(long = "va-y", hide = true)]
    pub va_y: Option<i64>,
    /// Accepted for compatibility; plots are not drawn
    #[arg(long = "va-t", hide = true)]
    pub va_t: Option<i64>,
    /// Accepted for compatibility; plots are not drawn
    #[arg(long = "va-p", hide = true)]
    pub va_p: Option<i64>,

    /// [DEBUG] Skip subtel cluster refinement
    #[arg(long = "debug-nosubtel")]
    pub debug_nosubtel: bool,
    /// [DEBUG] Do not align reads or do any anchoring
    #[arg(long = "debug-noanchor")]
    pub debug_noanchor: bool,
    /// [DEBUG] Stop immediately after extracting tel reads
    #[arg(long = "debug-telreads")]
    pub debug_telreads: bool,
    /// [DEBUG] Print each allele comparison during the final collapse
    #[arg(long = "debug-collapse")]
    pub debug_collapse: bool,
    /// [PERFORMANCE] Use faster but less accurate pairwise alignment
    #[arg(long = "fast-aln")]
    pub fast_aln: bool,
    /// Plotting and caching flags, accepted for compatibility and ignored
    #[arg(
        long = "ignored-flag",
        hide = true,
        num_args = 0,
        action = clap::ArgAction::Count,
        aliases = ["plot-filt-tvr", "plot-t1clust", "plot-t2clust", "plot-finclust", "plot-signals",
                   "plot-fusions", "dont-reprocess", "debug-replot", "debug-progress", "debug-dendro"]
    )]
    pub ignored_flags: u8,

    /// Merge similar alleles mapped <= this bp from each other
    #[arg(long = "collapse-hom", default_value_t = 1000)]
    pub collapse_hom: i64,
    /// /path/to/minimap2
    #[arg(long)]
    pub minimap2: Option<PathBuf>,
    /// /path/to/pbmm2
    #[arg(long)]
    pub pbmm2: Option<PathBuf>,
    /// /path/to/winnowmap
    #[arg(long)]
    pub winnowmap: Option<PathBuf>,
    /// High frequency kmers file (only needed for winnowmap)
    #[arg(long = "winnowmap-k15")]
    pub winnowmap_k15: Option<PathBuf>,
    /// /path/to/igzip for gzipped FASTA/FASTQ input (default: igzip on PATH, if any)
    #[arg(long)]
    pub igzip: Option<PathBuf>,
    /// Reference filename (only needed if input is cram)
    #[arg(long = "ref")]
    pub cram_ref: Option<PathBuf>,
    /// RNG seed value
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    pub rng: i64,
}

impl Args {
    pub fn tl_method(&self) -> Result<TlMethod> {
        TlMethod::parse(&self.tl_method)
    }
}

/// Telogator2 options that are long names behind a single dash (`-t0`, `-afa-x`, ...).
const SINGLE_DASH_LONG: &[&str] =
    &["t0", "t1", "t2", "tc", "ts", "th", "afa-x", "afa-t", "afa-a", "va-y", "va-t", "va-p"];

/// Rewrites `-t0`-style options to `--t0` so clap can parse Telogator2 command lines.
pub fn normalize_args(args: impl IntoIterator<Item = String>) -> Vec<String> {
    args.into_iter()
        .map(|a| match a.strip_prefix('-') {
            Some(rest) if !rest.starts_with('-') && SINGLE_DASH_LONG.contains(&rest) => format!("-{a}"),
            _ => a,
        })
        .collect()
}
