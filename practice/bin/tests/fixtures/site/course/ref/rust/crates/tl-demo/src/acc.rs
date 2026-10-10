//! Compensated summation (fixture ds.90).

/// Kahan sum: carries the rounding error of each addition forward.
pub fn kahan_sum(xs: &[f64]) -> f64 {
    // SOLUTION-BEGIN ds.90
    let mut s = 0.0;
    let mut c = 0.0;
    for &x in xs {
        let y = x - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    s
    // SOLUTION-END
}
