//! 02-drop-order: destructors run in reverse declaration order, at the end of
//! the enclosing scope. The exception, `let _ = ...`, is the one that costs
//! people a day of debugging a lock that was never held.

struct Noisy(&'static str);

impl Drop for Noisy {
    fn drop(&mut self) {
        println!("  drop {}", self.0);
    }
}

fn main() {
    let _a = Noisy("a");
    let _b = Noisy("b");

    println!("1. inner scope starts");
    {
        let _c = Noisy("c");
        println!("2. inner scope ends");
    }

    println!("3. vec built");
    let v = vec![Noisy("v0"), Noisy("v1")];
    drop(v);

    // `_` is not a binding: this value has no owner and dies immediately.
    println!("4. before wildcard");
    let _ = Noisy("wildcard");
    println!("5. after wildcard");

    // A named binding starting with an underscore *is* a binding.
    let _named = Noisy("named");
    println!("6. after named");

    // A temporary inside a statement lives until the end of that statement.
    println!("7. {}", Noisy("temporary").0);

    println!("8. main ends");
}
