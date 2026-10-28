# telocroc

Fast allele-specific telomere length from long reads (PacBio HiFi and ONT).

## Speed

Against Telogator2 2.2.4 on a 10-core desktop (`-p 16`), with the same alleles found:

| input | Telogator2 | telocroc | speedup |
|---|---:|---:|---:|
| HG002 HiFi telomere reads (3,212) | 50 min 54 s | 55 s | 55x |
| HG002 ONT telomere reads | 22 min 10 s | 67 s | 20x |
| HG002 Revio HiFi whole-genome BAM (40 GB) | ~5 min | 32 s | ~9x |

The ONT run is mostly minimap2, and the whole-genome run is mostly reading the BAM from disk.

## Building

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

## License

MIT (`LICENSE`).

Implements the [Telogator2](https://github.com/zstephens/telogator2) method by Zachary
Stephens et al.; its license is in `LICENSE-TELOGATOR2`.
