use tl_demo::acc::kahan_sum;

#[test]
fn sums_small() {
    assert_eq!(kahan_sum(&[1.0, 2.0, 3.0]), 6.0);
}

#[test]
fn compensates_tiny_terms() {
    let mut xs = vec![1.0];
    xs.extend(std::iter::repeat(1e-16).take(10));
    assert_eq!(kahan_sum(&xs), 1.0 + 1e-15);
}
