# telocroc

Fast allele-specific telomere length from long reads (PacBio HiFi and ONT).

## Speed

Against Telogator2 2.2.4:

| input | Telogator2 | telocroc | speedup |
|---|---:|---:|---:|
| HG002 HiFi telomere reads (3,212) | 50 min 54 s | 55 s | 55x |
| HG002 ONT telomere reads | 22 min 10 s | 67 s | 20x |
| HG002 Revio HiFi whole-genome BAM (40 GB) | ~5 min | 32 s | ~9x |

## Installing

Download a binary for Linux (x86-64, aarch64) or macOS (Apple Silicon) from
[Releases](https://github.com/DylanCosto/telocroc/releases) and unpack it; the subtelomere
reference is included. You also need an aligner: minimap2, winnowmap, pbmm2, or
[bonefish](https://github.com/DylanCosto/bonefish), a faster minimap2-compatible aligner
(`--minimap2 /path/to/bonefish`).

To build from source instead:

```bash
cargo build --release
scripts/fetch-resources.sh   # subtelomere reference (~70 MB)
```

Needs Rust 1.91+. Builds on x86-64 and aarch64 (Apple Silicon,
Graviton): AVX2 is used when available on x86-64, and NEON always on aarch64.

## Usage

```bash
telocroc -i reads.bam -o out/ -r hifi -p 16 --minimap2 /path/to/minimap2
```

Input can be FASTA or FASTQ (gzipped or not), BAM, or CRAM (with `--ref`), aligned or unaligned.
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

## Citing

telocroc implements the Telogator2 method; please cite:

> Stephens Z, Kocher JP. Characterization of telomere variant repeats using long reads enables
> allele-specific telomere length estimation. *BMC Bioinformatics* 25, 194 (2024).
> https://doi.org/10.1186/s12859-024-05807-5

## License

MIT (`LICENSE`).

Implements the [Telogator2](https://github.com/zstephens/telogator2) method by Zachary
Stephens et al.; its license is in `LICENSE-TELOGATOR2`.
