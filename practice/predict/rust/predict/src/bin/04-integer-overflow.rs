//! 04-integer-overflow: Rust panics on overflow in debug and wraps in release,
//! so the only honest way to write overflow-sensitive code is to say which one
//! you meant. Every operation here is explicit, so the output is the same in
//! both profiles.

fn main() {
    let a: u8 = 250;
    println!("1. {}", a.wrapping_add(10));
    println!("2. {:?}", a.checked_add(10));
    println!("3. {}", a.saturating_add(10));
    println!("4. {:?}", a.overflowing_add(10));

    // Division truncates toward zero; the remainder takes the sign of the
    // dividend. Euclidean division is the one that matches modular arithmetic.
    let b: i32 = -7;
    println!("5. {} {}", b / 2, b % 2);
    println!("6. {} {}", b.div_euclid(2), b.rem_euclid(2));

    // `as` is a truncating cast that never fails and never warns you.
    let big: i64 = 300;
    println!("7. {}", big as u8);
    println!("8. {}", -1i32 as u32);
    println!("9. {}", 2.9_f64 as i32);
    println!("10. {}", u8::MAX as i8);

    // Float to int saturates at the boundaries instead of being undefined.
    println!("11. {} {}", 1e10_f64 as i32, f64::NAN as i32);

    // Shifts are checked against the bit width, not silently masked.
    println!("12. {:?} {}", 1u8.checked_shl(9), 1u8.wrapping_shl(9));

    // Integer literals default to i32; the suffix is what picks the width.
    println!("13. {} {}", i32::MAX, i64::from(i32::MAX) + 1);
}
