//! 01-shadowing-and-blocks: almost everything in Rust is an expression, and
//! `let` does not mutate, it introduces a new binding that hides the old one.
//! Scope decides which one you are looking at.

fn main() {
    let x = 5;
    let x = x * 2;
    {
        let x = x + 1;
        println!("1. {x}");
    }
    println!("2. {x}");

    // A block is an expression whose value is its final expression.
    let y = {
        let a = 3;
        a * a
    };
    println!("3. {y}");

    // So is `if`, and so is `loop` with a value-carrying break.
    let size = if y > 5 { "big" } else { "small" };
    let mut n = 0;
    let counted = loop {
        n += 1;
        if n == 4 {
            break n * 10;
        }
    };
    println!("4. {size} {counted}");

    // Shadowing can change the type; assignment cannot.
    let spaces = "   ";
    let spaces = spaces.len();
    println!("5. {spaces}");

    // A trailing semicolon turns an expression into a statement of type ().
    let unit = {
        42;
    };
    println!("6. {unit:?}");

    // Shadowing inside a loop body starts fresh every iteration.
    let mut total = 0;
    for i in 1..=3 {
        let i = i * 100;
        total += i;
    }
    println!("7. {total}");
}
