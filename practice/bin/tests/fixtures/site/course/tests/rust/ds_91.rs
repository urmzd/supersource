use tl_demo::mean::mean;

#[test]
fn hand_example() {
    // WHY: the chapter's worked example: mean of 1, 2, 3, 4 is 2.5.
    // KIND: unit
    assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
}

#[test]
fn empty_is_none() {
    // WHY: the mean of nothing is undefined, not 0 and not NaN.
    // KIND: boundary
    assert_eq!(mean(&[]), None);
}
