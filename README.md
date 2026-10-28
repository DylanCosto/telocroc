# telocroc

Fast allele-specific telomere length from long reads (PacBio HiFi and ONT).

## Speed

On a 10-core desktop with `-p 16`: 3,212 HG002 HiFi telomere reads in 55 s, and a whole
HG002 Revio HiFi BAM (40 GB) in 32 s.

## Building

```bash
cargo build --release
scripts/fetch-resources.sh   # subtelomere reference (~70 MB)
```

Needs Rust 1.87+ and libclang (for `rust-htslib`). AVX2 is used when available.

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
