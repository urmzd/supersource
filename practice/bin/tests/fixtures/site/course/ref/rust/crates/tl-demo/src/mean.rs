//! Mean over the compensated sum (fixture ds.91).
use crate::acc::kahan_sum;

/// The arithmetic mean, or None for an empty slice.
pub fn mean(xs: &[f64]) -> Option<f64> {
    // SOLUTION-BEGIN ds.91
    if xs.is_empty() {
        return None;
    }
    Some(kahan_sum(xs) / xs.len() as f64)
    // SOLUTION-END
}
