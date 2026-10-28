//! Exact early rejection of reads straight from BAM's 4-bit packed bases.
//!
//! Almost every read in a whole-genome BAM is not telomeric, and decoding all of them to
//! letters before searching costs more than the search. Each kmer is found in the packed
//! bytes by a 3-byte anchor at both nibble phases, then checked nibble by nibble. The
//! counts are overlapping occurrences, which bound the non-overlapping counts the exact
//! test uses from above, so a read this rejects can never pass the exact test.

/// BAM's 4-bit base codes (`=ACMGRSVTWYHKDBN`) for the bases a kmer may contain.
fn nibble_code(base: u8) -> Option<u8> {
    match base {
        b'A' => Some(1),
        b'C' => Some(2),
        b'G' => Some(4),
        b'T' => Some(8),
        _ => None,
    }
}

/// Where a kmer can start relative to an anchor byte.
#[derive(Clone, Copy, Debug)]
enum Phase {
    /// The kmer starts at the anchor byte's high nibble.
    Even,
    /// The kmer starts at the low nibble of the byte before the anchor.
    Odd,
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    bytes: [u8; 3],
    kmer: usize,
    phase: Phase,
}

/// A forward and a reverse-complement kmer, prepared for packed-base search.
#[derive(Clone, Debug)]
pub struct PackedPrefilter {
    codes: [Vec<u8>; 2],
    anchors: [Anchor; 4],
}

impl PackedPrefilter {
    /// Returns `None` when a kmer cannot be searched in packed form (shorter than 7 bases
    /// or holding a letter other than ACGT); callers then decode every read.
    pub fn new(kmer: &[u8], kmer_rc: &[u8]) -> Option<Self> {
        let encode = |k: &[u8]| k.iter().map(|&b| nibble_code(b)).collect::<Option<Vec<u8>>>();
        let codes = [encode(kmer)?, encode(kmer_rc)?];
        if codes.iter().any(|c| c.len() < 7) {
            return None;
        }
        let pair = |c: &[u8], i: usize| (c[i] << 4) | c[i + 1];
        let anchor = |kmer: usize, phase: Phase| {
            let (c, first) = (&codes[kmer], matches!(phase, Phase::Odd) as usize);
            Anchor { bytes: [pair(c, first), pair(c, first + 2), pair(c, first + 4)], kmer, phase }
        };
        let anchors = [anchor(0, Phase::Even), anchor(0, Phase::Odd), anchor(1, Phase::Even), anchor(1, Phase::Odd)];
        Some(Self { codes, anchors })
    }

    /// Whether either kmer might occur `min_hits` times in the `len` bases packed in
    /// `packed`. `false` is exact: the read cannot pass the decoded test.
    pub fn may_match(&self, packed: &[u8], len: usize, min_hits: usize) -> bool {
        min_hits == 0 || self.count_hits(packed, len, min_hits).iter().any(|&n| n >= min_hits)
    }

    /// Overlapping occurrences of each kmer, stopping once either reaches `limit`.
    fn count_hits(&self, packed: &[u8], len: usize, limit: usize) -> [usize; 2] {
        let packed = &packed[..len.div_ceil(2)];
        let mut counts = [0; 2];
        #[cfg(target_arch = "x86_64")]
        let start = if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was checked at runtime just above.
            unsafe { self.scan_avx2(packed, len, limit, &mut counts) }
        } else {
            0
        };
        #[cfg(target_arch = "aarch64")]
        let start = self.scan_neon(packed, len, limit, &mut counts);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let start = 0;
        for j in start..packed.len().saturating_sub(2) {
            if counts.iter().any(|&n| n >= limit) {
                break;
            }
            self.check_byte(packed, len, j, &mut counts);
        }
        counts
    }

    /// Counts every kmer occurrence anchored at byte `j`.
    fn check_byte(&self, packed: &[u8], len: usize, j: usize, counts: &mut [usize; 2]) {
        for a in &self.anchors {
            if packed[j..j + 3] != a.bytes {
                continue;
            }
            let start = match a.phase {
                Phase::Even => 2 * j,
                Phase::Odd if j > 0 => 2 * j - 1,
                Phase::Odd => continue,
            };
            if matches_at(packed, len, start, &self.codes[a.kmer]) {
                counts[a.kmer] += 1;
            }
        }
    }

    /// Checks anchors 32 bytes at a time, from byte 0 while 34 bytes remain. Returns the
    /// first byte not yet checked.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_avx2(&self, packed: &[u8], len: usize, limit: usize, counts: &mut [usize; 2]) -> usize {
        use std::arch::x86_64::{
            __m256i, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
            _mm256_set1_epi8, _mm256_setzero_si256,
        };
        let splat = |b: u8| _mm256_set1_epi8(b as i8);
        let want: [[__m256i; 3]; 4] = self.anchors.map(|a| a.bytes.map(splat));
        let mut j = 0;
        while j + 34 <= packed.len() {
            // SAFETY: j + 2 + 32 <= packed.len(), so all three unaligned loads are in bounds.
            let v = unsafe { [0, 1, 2].map(|o| _mm256_loadu_si256(packed.as_ptr().add(j + o).cast())) };
            let mut hit = _mm256_setzero_si256();
            for w in &want {
                let m = _mm256_and_si256(
                    _mm256_and_si256(_mm256_cmpeq_epi8(v[0], w[0]), _mm256_cmpeq_epi8(v[1], w[1])),
                    _mm256_cmpeq_epi8(v[2], w[2]),
                );
                hit = _mm256_or_si256(hit, m);
            }
            let mut mask = _mm256_movemask_epi8(hit) as u32;
            while mask != 0 {
                self.check_byte(packed, len, j + mask.trailing_zeros() as usize, counts);
                mask &= mask - 1;
            }
            if counts.iter().any(|&n| n >= limit) {
                return packed.len();
            }
            j += 32;
        }
        j
    }

    /// Checks anchors 32 bytes at a time (two 16-byte halves), from byte 0 while 34 bytes
    /// remain. Returns the first byte not yet checked.
    #[cfg(target_arch = "aarch64")]
    fn scan_neon(&self, packed: &[u8], len: usize, limit: usize, counts: &mut [usize; 2]) -> usize {
        use std::arch::aarch64::{
            vandq_u8, vceqq_u8, vdupq_n_u8, vget_lane_u64, vld1q_u8, vorrq_u8, vreinterpret_u64_u8,
            vreinterpretq_u16_u8, vshrn_n_u16,
        };

        use crate::neon::neon;
        let want = self.anchors.map(|a| a.bytes.map(|b| neon!(vdupq_n_u8(b))));
        let mut j = 0;
        while j + 34 <= packed.len() {
            for at in [j, j + 16] {
                // SAFETY: at + 2 + 16 <= j + 34 <= packed.len(), so all three unaligned loads
                // are in bounds.
                let v = unsafe { [0, 1, 2].map(|o| vld1q_u8(packed.as_ptr().add(at + o))) };
                let hit = want.iter().fold(neon!(vdupq_n_u8(0)), |hit, w| {
                    neon!(vorrq_u8(
                        hit,
                        vandq_u8(vandq_u8(vceqq_u8(v[0], w[0]), vceqq_u8(v[1], w[1])), vceqq_u8(v[2], w[2]))
                    ))
                });
                // No movemask on NEON: narrowing each 16-bit pair by 4 bits leaves 4 bits per
                // byte, byte k at bits 4k..4k + 4; keep one of them.
                let nibbles =
                    neon!(vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(hit)))));
                let mut mask = nibbles & 0x8888_8888_8888_8888;
                while mask != 0 {
                    self.check_byte(packed, len, at + mask.trailing_zeros() as usize / 4, counts);
                    mask &= mask - 1;
                }
            }
            if counts.iter().any(|&n| n >= limit) {
                return packed.len();
            }
            j += 32;
        }
        j
    }
}

/// Whether the nibbles from nibble `start` on spell `codes`, within the first `len`.
fn matches_at(packed: &[u8], len: usize, start: usize, codes: &[u8]) -> bool {
    start + codes.len() <= len
        && codes.iter().enumerate().all(|(t, &c)| {
            let n = start + t;
            let byte = packed[n / 2];
            (if n.is_multiple_of(2) { byte >> 4 } else { byte & 15 }) == c
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(seq: &[u8]) -> Vec<u8> {
        const NT16: &[u8; 16] = b"=ACMGRSVTWYHKDBN";
        let code = |b: u8| NT16.iter().position(|&c| c == b).expect("IUPAC letter") as u8;
        seq.chunks(2).map(|p| (code(p[0]) << 4) | p.get(1).map_or(0, |&b| code(b))).collect()
    }

    fn overlapping(seq: &[u8], kmer: &[u8]) -> usize {
        seq.windows(kmer.len()).filter(|w| *w == kmer).count()
    }

    /// Small xorshift generator so the tests need no dependency.
    fn next(state: &mut u64) -> usize {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as usize
    }

    fn random_seq(state: &mut u64, len: usize, alphabet: &[u8]) -> Vec<u8> {
        (0..len).map(|_| alphabet[next(state) % alphabet.len()]).collect()
    }

    #[test]
    fn counts_equal_overlapping_occurrences() {
        let (kmer, rc) = (b"CCCTAACCCTAA".as_slice(), b"TTAGGGTTAGGG".as_slice());
        let filter = PackedPrefilter::new(kmer, rc).expect("ACGT kmers");
        let mut state = 0x9e37_79b9_7f4a_7c15;
        for case in 0..400 {
            let len = case % 7 + 2 * case;
            let mut seq = random_seq(&mut state, len, b"ACGTN");
            // Plant repeats of both kmers at random (often odd) offsets, some overlapping.
            for k in 0..case % 9 {
                let pos = next(&mut state) % (seq.len() + 1);
                let unit = if k % 2 == 0 { &kmer[..6] } else { &rc[..6] };
                let tract: Vec<u8> = unit.iter().cycle().take(6 * (k + 2)).copied().collect();
                seq.splice(pos..pos, tract);
            }
            let packed = pack(&seq);
            let got = filter.count_hits(&packed, seq.len(), usize::MAX);
            assert_eq!(got, [overlapping(&seq, kmer), overlapping(&seq, rc)], "case {case}, len {}", seq.len());
        }
    }

    #[test]
    fn never_rejects_a_read_the_exact_test_keeps() {
        let (kmer, rc) = (b"CCCTAACCCTAA".as_slice(), b"TTAGGGTTAGGG".as_slice());
        let filter = PackedPrefilter::new(kmer, rc).expect("ACGT kmers");
        for offset in 0..3 {
            for repeats in 0..12 {
                let mut seq = vec![b'G'; offset];
                seq.extend(b"CCCTAA".iter().cycle().take(6 * repeats));
                seq.extend(b"ACGTTGCA");
                let packed = pack(&seq);
                let exact = memchr::memmem::find_iter(&seq, kmer).count() >= 8;
                assert!(!exact || filter.may_match(&packed, seq.len(), 8), "offset {offset}, repeats {repeats}");
            }
        }
    }

    #[test]
    fn falls_back_for_kmers_it_cannot_pack() {
        assert!(PackedPrefilter::new(b"CCCTAA", b"TTAGGG").is_none());
        assert!(PackedPrefilter::new(b"CCCTNACCCTAA", b"TTAGGGTTAGGG").is_none());
    }
}
