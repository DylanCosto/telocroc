//! The handful of NumPy reductions Telogator2 relies on, with NumPy's summation order so
//! that telomere-length summaries round the same way.

/// NumPy's pairwise summation (`pairwise_sum_DOUBLE`, block size 128, unrolled by 8).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Single-precision pairwise summation, as `np.mean` does for `float32` arrays.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f32;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// `np.mean` of a non-empty `float64` (or integer) sample.
pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `np.mean` of a non-empty `float32` sample.
pub fn mean_f32(a: &[f32]) -> f32 {
    pairwise_sum_f32(a) / a.len() as f32
}

/// `np.median` of a non-empty sample.
pub fn median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { mean(&v[n / 2 - 1..=n / 2]) }
}

/// `np.percentile(a, q)` with the default linear method, for a non-empty sample.
pub fn percentile(a: &[f64], q: f64) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let quantile = q / 100.0;
    // _compute_virtual_index with alpha = beta = 1
    let virtual_index = n as f64 * quantile + (1.0 - quantile) - 1.0;
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let below = (previous.max(0.0) as usize).min(n - 1);
    let above = ((previous + 1.0).max(0.0) as usize).min(n - 1);
    lerp(v[below], v[above], gamma)
}

/// NumPy's `_lerp`, which interpolates from the nearer end for numerical stability.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy() {
        // np.percentile([1, 2, 3, 4], 75) == 3.25; [10, 20, 40] -> 30.0
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 75.0), 3.25);
        assert_eq!(percentile(&[10.0, 20.0, 40.0], 75.0), 30.0);
        assert_eq!(percentile(&[5.0], 75.0), 5.0);
    }

    #[test]
    fn median_even_and_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
