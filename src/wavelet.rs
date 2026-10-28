//! Daubechies-4 wavelet denoising, ported from PyWavelets' periodization-mode C kernels
//! so that the smoothed p-vs-q telomere signal matches `pywt` to the last bit.

use crate::numpy;

const F: usize = 8;

const DEC_LO: [u64; F] = [
    0xbf85b41730b72e29,
    0x3fa0d60ac768117b,
    0x3f9f94e2196383a9,
    0xbfc7f0c1b7c604d4,
    0xbf9ca7c6f9db5bfb,
    0x3fe4302cdd3de43a,
    0x3fe6e005ea45d748,
    0x3fcd7d052af15ec0,
];
const DEC_HI: [u64; F] = [
    0xbfcd7d052af15ec0,
    0x3fe6e005ea45d748,
    0xbfe4302cdd3de43a,
    0xbf9ca7c6f9db5bfb,
    0x3fc7f0c1b7c604d4,
    0x3f9f94e2196383a9,
    0xbfa0d60ac768117b,
    0xbf85b41730b72e29,
];
const REC_LO: [u64; F] = [
    0x3fcd7d052af15ec0,
    0x3fe6e005ea45d748,
    0x3fe4302cdd3de43a,
    0xbf9ca7c6f9db5bfb,
    0xbfc7f0c1b7c604d4,
    0x3f9f94e2196383a9,
    0x3fa0d60ac768117b,
    0xbf85b41730b72e29,
];
const REC_HI: [u64; F] = [
    0xbf85b41730b72e29,
    0xbfa0d60ac768117b,
    0x3f9f94e2196383a9,
    0x3fc7f0c1b7c604d4,
    0xbf9ca7c6f9db5bfb,
    0xbfe4302cdd3de43a,
    0x3fe6e005ea45d748,
    0xbfcd7d052af15ec0,
];

fn filter(bits: &[u64; F]) -> [f64; F] {
    bits.map(f64::from_bits)
}

/// `downsampling_convolution_periodization` with step 2 and fstep 1.
fn downsample_per(input: &[f64], filt: &[f64; F]) -> Vec<f64> {
    let n = input.len();
    let step = 2;
    let padding = (step - (n % step)) % step;
    let mut out = Vec::with_capacity(n.div_ceil(2));
    let mut i = F / 2;
    while i < F && i < n {
        let mut sum = 0.0;
        let mut j = 0;
        while j <= i {
            sum += filt[j] * input[i - j];
            j += 1;
        }
        while j < F {
            let mut k = 0;
            while k < padding && j < F {
                sum += filt[j] * input[n - 1];
                k += 1;
                j += 1;
            }
            k = 0;
            while k < n && j < F {
                sum += filt[j] * input[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        out.push(sum);
        i += step;
    }
    while i < n {
        let mut sum = 0.0;
        for j in 0..F {
            sum += filt[j] * input[i - j];
        }
        out.push(sum);
        i += step;
    }
    while i < F && i < n + F / 2 {
        let mut sum = 0.0;
        let mut j = 0;
        wrap_front(input, filt, i, &mut j, padding, &mut sum);
        while j <= i {
            sum += filt[j] * input[i - j];
            j += 1;
        }
        while j < F {
            let mut k = 0;
            while k < padding && j < F {
                sum += filt[j] * input[n - 1];
                k += 1;
                j += 1;
            }
            k = 0;
            while k < n && j < F {
                sum += filt[j] * input[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        out.push(sum);
        i += step;
    }
    while i < n + F / 2 {
        let mut sum = 0.0;
        let mut j = 0;
        wrap_front(input, filt, i, &mut j, padding, &mut sum);
        while j < F {
            sum += filt[j] * input[i - j];
            j += 1;
        }
        out.push(sum);
        i += step;
    }
    out
}

/// The `while (i-j >= N)` wrap-around block shared by the last two downsampling loops.
fn wrap_front(input: &[f64], filt: &[f64; F], i: usize, j: &mut usize, padding: usize, sum: &mut f64) {
    let n = input.len();
    while i - *j >= n {
        let mut k = 0;
        while k < padding && i - *j >= n {
            *sum += filt[i - n - *j] * input[n - 1];
            k += 1;
            *j += 1;
        }
        k = 0;
        while k < n && i - *j >= n {
            *sum += filt[i - n - *j] * input[k];
            k += 1;
            *j += 1;
        }
    }
}

/// `upsampling_convolution_valid_sf_periodization`, accumulating into `out` (length 2N).
fn upsample_per_add(input: &[f64], filt: &[f64; F], out: &mut [f64]) {
    let n = input.len();
    let half = F / 2;
    let start = F / 4;
    let end = n + start - if half.is_multiple_of(2) { 1 } else { 0 };
    let mut i = start;
    let mut o = 0;
    if half.is_multiple_of(2) {
        let mut j = 0;
        while j < start {
            let mut k = 0;
            while k < n && j < start {
                out[2 * n - 1] += filt[2 * (start - 1 - j)] * input[k];
                out[0] += filt[2 * (start - 1 - j) + 1] * input[k];
                k += 1;
                j += 1;
            }
        }
        while j < n + start && j < half {
            out[2 * n - 1] += filt[2 * j] * input[n + start - 1 - j];
            out[0] += filt[2 * j + 1] * input[n + start - 1 - j];
            j += 1;
        }
        while j < half {
            let mut k = 0;
            while k < n && j < half {
                out[2 * n - 1] += filt[2 * j] * input[n - 1 - k];
                out[0] += filt[2 * j + 1] * input[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        o += 1;
    }
    while i < half && i < n {
        let mut j = 0;
        while j <= i {
            out[o] += filt[2 * j] * input[i - j];
            out[o + 1] += filt[2 * j + 1] * input[i - j];
            j += 1;
        }
        while j < half {
            let mut k = 0;
            while k < n && j < half {
                out[o] += filt[2 * j] * input[n - 1 - k];
                out[o + 1] += filt[2 * j + 1] * input[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        i += 1;
        o += 2;
    }
    while i < n {
        for j in 0..half {
            out[o] += filt[2 * j] * input[i - j];
            out[o + 1] += filt[2 * j + 1] * input[i - j];
        }
        i += 1;
        o += 2;
    }
    while i < half && i < end {
        let mut j = 0;
        upsample_wrap(input, filt, i, &mut j, o, out);
        while j <= i && j < half {
            out[o] += filt[2 * j] * input[i - j];
            out[o + 1] += filt[2 * j + 1] * input[i - j];
            j += 1;
        }
        while j < half {
            let mut k = 0;
            while k < n && j < half {
                out[o] += filt[2 * j] * input[n - 1 - k];
                out[o + 1] += filt[2 * j + 1] * input[n - 1 - k];
                k += 1;
                j += 1;
            }
        }
        i += 1;
        o += 2;
    }
    while i < end {
        let mut j = 0;
        upsample_wrap(input, filt, i, &mut j, o, out);
        while j <= i && j < half {
            out[o] += filt[2 * j] * input[i - j];
            out[o + 1] += filt[2 * j + 1] * input[i - j];
            j += 1;
        }
        i += 1;
        o += 2;
    }
}

fn upsample_wrap(input: &[f64], filt: &[f64; F], i: usize, j: &mut usize, o: usize, out: &mut [f64]) {
    let n = input.len();
    while i - *j >= n {
        let mut k = 0;
        while k < n && i - *j >= n {
            out[o] += filt[2 * (i - n - *j)] * input[k];
            out[o + 1] += filt[2 * (i - n - *j) + 1] * input[k];
            k += 1;
            *j += 1;
        }
    }
}

/// `pywt.dwt_max_level(n, 8)`.
fn max_level(n: usize) -> usize {
    if n < F - 1 { 0 } else { (usize::BITS - 1 - (n / (F - 1)).leading_zeros()) as usize }
}

/// `pywt.wavedec(x, 'db4', mode='per')`: `[cA_n, cD_n, ..., cD_1]`.
fn wavedec(x: &[f64]) -> Vec<Vec<f64>> {
    let (lo, hi) = (filter(&DEC_LO), filter(&DEC_HI));
    let mut a = x.to_vec();
    let mut coeffs = Vec::new();
    for _ in 0..max_level(x.len()) {
        let d = downsample_per(&a, &hi);
        a = downsample_per(&a, &lo);
        coeffs.push(d);
    }
    coeffs.push(a);
    coeffs.reverse();
    coeffs
}

/// `pywt.waverec(coeffs, 'db4', mode='per')`.
fn waverec(coeffs: &[Vec<f64>]) -> Vec<f64> {
    let (lo, hi) = (filter(&REC_LO), filter(&REC_HI));
    let mut a = coeffs[0].clone();
    for d in &coeffs[1..] {
        if a.len() == d.len() + 1 {
            a.truncate(d.len());
        }
        let mut out = vec![0.0; 2 * a.len()];
        upsample_per_add(&a, &lo, &mut out);
        upsample_per_add(d, &hi, &mut out);
        a = out;
    }
    a
}

/// Median absolute deviation scaled to a normal sigma (`tg_kmer.mad`).
fn mad(x: &[f64]) -> f64 {
    let med = numpy::median(x);
    let dev: Vec<f64> = x.iter().map(|v| (v - med).abs()).collect();
    1.4826 * numpy::median(&dev)
}

/// `tg_kmer.wavelet_smooth`: soft-threshold the detail coefficients and reconstruct.
///
/// Returns the input unchanged when there are too few levels or the signal is flat.
/// Like PyWavelets, an odd-length input comes back one element longer.
pub fn smooth(x: &[f64]) -> Vec<f64> {
    const SMOOTHING_LEVEL: usize = 5;
    const FAIL_SIGMA: f64 = 1e-3;
    let mut coeffs = wavedec(x);
    if coeffs.len() < SMOOTHING_LEVEL {
        return x.to_vec();
    }
    let sigma = mad(&coeffs[coeffs.len() - SMOOTHING_LEVEL]);
    if sigma < FAIL_SIGMA {
        return x.to_vec();
    }
    let uthresh = sigma * (2.0 * (x.len() as f64).ln()).sqrt();
    for c in &mut coeffs[1..] {
        for v in c.iter_mut() {
            // pywt.threshold(mode='soft'): data * clip(1 - value / |data|, 0)
            let keep = (1.0 - uthresh / v.abs()).max(0.0);
            *v *= keep;
        }
    }
    waverec(&coeffs)
}
