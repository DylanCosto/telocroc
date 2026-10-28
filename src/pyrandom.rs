//! CPython's `random` module (MT19937), reproduced bit for bit.
//!
//! Telogator2 seeds Python's generator once per pairwise distance and shuffles the two
//! sequences with `random.sample`, so matching CPython's stream makes every distance in
//! the clustering matrices identical to the Python run with the same `--rng` seed.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A Mersenne Twister seeded the way `random.seed(int)` seeds it in CPython 3.
#[derive(Clone, Debug)]
pub struct PyRandom {
    mt: Box<[u32; N]>,
    index: usize,
}

impl PyRandom {
    /// Equivalent to `random.seed(seed)` for an integer seed.
    pub fn new(seed: u64) -> Self {
        // CPython splits abs(seed) into little-endian 32-bit words (one word for zero).
        let key: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        let mut rng = Self { mt: Box::new([0; N]), index: N };
        rng.init_by_array(&key);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        let mt = &mut self.mt;
        mt[0] = seed;
        for i in 1..N {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mt = &mut self.mt;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
    }

    fn generate(&mut self) {
        let mt = &mut self.mt;
        for kk in 0..N {
            let y = (mt[kk] & UPPER_MASK) | (mt[(kk + 1) % N] & LOWER_MASK);
            let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
            mt[kk] = mt[(kk + M) % N] ^ (y >> 1) ^ mag;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.getrandbits(k)` for `1 <= k <= 32`.
    fn getrandbits(&mut self, k: u32) -> u32 {
        debug_assert!((1..=32).contains(&k), "getrandbits({k}) is only implemented up to 32 bits");
        self.next_u32() >> (32 - k)
    }

    /// `random._randbelow(n)` (the getrandbits rejection sampler) for `n >= 1`.
    pub fn randbelow(&mut self, n: usize) -> usize {
        assert!(n >= 1 && n <= u32::MAX as usize, "randbelow({n}) out of range");
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `''.join(random.sample(s, len(s)))`: a random permutation, as Telogator2's
    /// `shuffle_seq` builds it. Writes into `out` to avoid allocating per call.
    pub fn sample_all_into(&mut self, s: &[u8], out: &mut Vec<u8>, pool: &mut Vec<u8>) {
        // For k == n CPython always takes the pool branch of random.sample.
        let n = s.len();
        out.clear();
        pool.clear();
        pool.extend_from_slice(s);
        for i in 0..n {
            let j = self.randbelow(n - i);
            out.push(pool[j]);
            pool[j] = pool[n - i - 1];
        }
    }

    /// `random.shuffle(x)` in place.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython_stream() {
        // random.seed(12345); [random.getrandbits(32) for _ in range(3)]
        let mut rng = PyRandom::new(12345);
        let got: Vec<u32> = (0..3).map(|_| rng.getrandbits(32)).collect();
        assert_eq!(got, vec![1_789_368_711, 3_146_859_322, 43_676_229]);
    }

    #[test]
    fn sample_matches_cpython() {
        let (mut out, mut pool) = (Vec::new(), Vec::new());
        // random.seed(7); ''.join(random.sample('ABCDEFGHIJ', 10))
        PyRandom::new(7).sample_all_into(b"ABCDEFGHIJ", &mut out, &mut pool);
        assert_eq!(out, b"FCGJAHEBDI");
        let s = b"AACCCCDDEEFFAAAAAAAAACCCCCCCCCCCCCCCCCCCCCCCCDDDDDDDDDDDDDDD";
        PyRandom::new(99_999_999 + 5).sample_all_into(s, &mut out, &mut pool);
        assert_eq!(out, b"CADCCADCDDCCDDACCCECCDDCCDCDCCCAAACACCCCCDDDADFADCDCCAADECFC");
    }

    #[test]
    fn shuffle_matches_cpython() {
        let mut x: Vec<u32> = (0..20).collect();
        PyRandom::new(3).shuffle(&mut x);
        assert_eq!(x, vec![8, 3, 6, 5, 15, 16, 2, 12, 0, 1, 13, 10, 19, 9, 14, 11, 4, 17, 18, 7]);
    }
}
