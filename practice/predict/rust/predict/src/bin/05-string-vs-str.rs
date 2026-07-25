//! 05-string-vs-str: `String` owns bytes, `&str` borrows them, and both index
//! by byte even though everyone reads them as characters. Rust refuses to guess
//! which one you meant, which is why slicing can return None.

fn main() {
    let s = String::from("héllo");
    println!("1. {} {}", s.len(), s.chars().count());

    // 0..2 would split the two-byte 'é' in half, so it is not a valid slice.
    println!("2. {:?}", s.get(0..2));
    println!("3. {:?}", s.get(0..1));
    println!("4. {:?}", s.char_indices().collect::<Vec<_>>());

    // Case conversion can change the byte length.
    let upper = s.to_uppercase();
    println!("5. {} {}", upper, upper.len());

    // split yields empty strings for adjacent separators; split_whitespace does not.
    println!("6. {:?}", "a,b,,c".split(',').collect::<Vec<_>>());
    println!("7. {:?}", "  a  b  ".split_whitespace().collect::<Vec<_>>());

    // &str and String compare equal without an explicit conversion.
    let owned = String::from("abc");
    println!("8. {} {}", "abc" == owned, owned.as_str() == "abc");

    // Pushing to a String may reallocate; capacity is not length.
    let mut buf = String::with_capacity(4);
    buf.push_str("ab");
    println!("9. {} {}", buf.len(), buf.capacity());

    // chars().rev() reverses code points, which is not the same as reversing
    // user-visible characters, but is at least never mojibake.
    println!("10. {}", s.chars().rev().collect::<String>());

    // Trim borrows: no allocation, and the result points into the original.
    let padded = "  edge  ";
    println!("11. [{}] {}", padded.trim(), padded.len());
}
