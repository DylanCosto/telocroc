# TODO

## Platforms

- [x] **ARM (aarch64, Apple Silicon):** NEON versions of all four kernels (`src/neon.rs`
      pairs 128-bit registers so lane counts match AVX2). On an M4 Pro, single thread:
      batch 28.6, striped 10.1, striped trace 3.3 Gcells/s (scalar: 0.29).
- [ ] **NEON tuning:** try 8-bit striped lanes for short alignments; profile on Graviton.
- [ ] **x86 without AVX2:** add a switch that forces the scalar paths, and run the tests with it.
- [ ] **AVX-512 (AVX512BW):** 64 lanes in the batch kernel, 32 x 16-bit in the striped
      kernels. Needs a Zen 4/5 or Xeon machine to test on (this i5 has no AVX-512).

## Features

- [ ] **Short reads:** not covered by the long-read method (reads don't span the telomere
      and subtelomere, so alleles can't be anchored). Scope it first: total TL by k-mer
      counting (TelSeq-style) is feasible; allele-specific TL probably isn't.
- [ ] **Plots:** `violin_atl.png`, `all_final_alleles.png`.
- [ ] Write `qc/readlens.npz` / `qc/methylation.npz`; support `--dont-reprocess`.
- [ ] Send interstitial-telomere ("fusion") subtelomeres to the aligner.

## Speed

- [ ] **BGZF FASTQ:** decompress blocks in parallel when the input is bgzipped (plain
      gzip stays single-core; igzip is the fallback).
- [ ] **ONT anchoring:** minimap2 takes ~47 s of 67 s. Try lighter settings, or offer
      bonefish (halves the run time; 4 of 86 allele positions changed in one test).

## Testing and validation

- [x] **CI:** fmt, clippy and tests on x86_64 Linux and aarch64 Linux/macOS
      (`.github/workflows/ci.yml`).
- [ ] **CI:** a forced-scalar job (needs the x86 scalar switch above).
- [ ] End-to-end regression test on the bundled HiFi/ONT telomere reads (fetch the test
      data in CI, compare `tlens_by_allele.tsv`).
- [ ] More ONT samples (R10, ultra-long with telomere coverage); the whole-genome R9
      FASTQ gives no alleles.
- [ ] Follow upstream releases after 2.2.4.

## Release

- [ ] Ask the Telogator2 maintainers before going public.
- [x] Paper citation (`CITATION.cff`, README), `Cargo.toml` metadata.
- [x] Prebuilt binaries: pushing a `v*` tag builds Linux x86-64/aarch64 (glibc 2.28) and
      macOS arm64 tarballs into a draft release (`.github/workflows/release.yml`).
- [ ] Tag v0.1.0 once CI runs; a bioconda recipe.
