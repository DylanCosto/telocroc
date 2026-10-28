# telocroc

Fast allele-specific telomere length from long reads (PacBio HiFi and ONT).

## Speed

Against Telogator2 2.2.4 on a 10-core desktop (`-p 16`):

| input | Telogator2 | telocroc | speedup |
|---|---:|---:|---:|
| HG002 HiFi telomere reads (3,212) | 50 min 54 s | 55 s | 55x |
| HG002 ONT telomere reads | 22 min 10 s | 67 s | 20x |
| HG002 Revio HiFi whole-genome BAM (40 GB) | ~5 min | 32 s | ~9x |

The ONT run is mostly minimap2, and the whole-genome run is mostly reading the BAM from disk.

## Installing

Download a binary for Linux (x86-64, aarch64) or macOS (Apple Silicon) from
[Releases](https://github.com/DylanCosto/telocroc/releases) and unpack it; the subtelomere
reference is included. You also need an aligner: minimap2, winnowmap or pbmm2.

To build from source instead:

```bash
cargo build --release
scripts/fetch-resources.sh   # subtelomere reference (~70 MB)
```

Needs Rust 1.87+ and libclang (for `rust-htslib`). Builds on x86-64 and aarch64 (Apple Silicon,
Graviton): AVX2 is used when available on x86-64, and NEON always on aarch64.

## Usage

```bash
telocroc -i reads.bam -o out/ -r hifi -p 16 --minimap2 /path/to/minimap2
```

Results are in `out/tlens_by_allele.tsv`; see `telocroc --help` for all options.
`--igzip /path/to/igzip` (or `igzip` on `PATH`) speeds up gzipped FASTA/FASTQ input.

## Output

`tlens_by_allele.tsv` has one row per allele, in the same format as Telogator2's:

| column | |
|---|---|
| `chr`, `position`, `ref_samp` | anchor arm, coordinate and T2T reference contig (`chrUq`: not anchored) |
| `allele_id` | allele ID; a trailing `i` marks an interstitial telomere, usually best excluded |
| `TL_p75` | allele telomere length (the column is named after `-m`, p75 by default) |
| `read_TLs`, `read_lengths`, `read_mapq` | per supporting read |
| `tvr_len`, `tvr_consensus` | telomere variant repeat region length and consensus |
| `supporting_reads` | read names |

`qc/stats.tsv` has read and allele counts. `temp/` keeps the telomere reads and the
subtelomere alignments.

## Differences from Telogator2

- On the HG002 test reads, telocroc finds the same alleles as Telogator2 2.2.4 (HiFi: 96 of
  98, where Telogator2 splits two alleles that telocroc keeps as one; ONT: 79 of 79). The
  median telomere-length difference is 2-3 bp.
- Positions are 0-based, as in Telogator2 2.2.5 (2.2.4 reports them 1 bp higher).
- No plots (`violin_atl.png`, `all_final_alleles.png`) or `qc/*.npz` files yet. The plotting
  flags and `--dont-reprocess` are accepted and ignored.

## Citing

telocroc implements the Telogator2 method; please cite:

> Stephens Z, Kocher JP. Characterization of telomere variant repeats using long reads enables
> allele-specific telomere length estimation. *BMC Bioinformatics* 25, 194 (2024).
> https://doi.org/10.1186/s12859-024-05807-5

## License

MIT (`LICENSE`).

Implements the [Telogator2](https://github.com/zstephens/telogator2) method by Zachary
Stephens et al.; its license is in `LICENSE-TELOGATOR2`.
