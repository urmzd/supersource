//! 06-move-and-rc: assignment moves unless the type is Copy, and shared
//! ownership is a runtime counter you can watch tick. Every count printed here
//! is a scope boundary you can see in the source.

use std::rc::Rc;

fn main() {
    let a = vec![1, 2, 3];
    let b = a.clone();
    println!("1. {a:?} {b:?}");

    // b is moved into c; b is no longer usable after this line.
    let c = b;
    println!("2. {c:?}");

    // i32 is Copy, so this is a copy and both bindings stay valid.
    let n = 5i32;
    let m = n;
    println!("3. {n} {m}");

    let r = Rc::new(String::from("shared"));
    println!("4. {}", Rc::strong_count(&r));

    let r2 = Rc::clone(&r);
    println!("5. {}", Rc::strong_count(&r));

    {
        let _r3 = Rc::clone(&r);
        println!("6. {}", Rc::strong_count(&r));
    }
    println!("7. {}", Rc::strong_count(&r));

    drop(r2);
    println!("8. {}", Rc::strong_count(&r));

    // Rc is a shared *immutable* handle: the pointee is one allocation.
    let r4 = Rc::clone(&r);
    println!("9. {}", Rc::ptr_eq(&r, &r4));

    // Passing by value moves; passing by reference does not.
    let owned = vec![10, 20];
    let len = takes_ref(&owned);
    let consumed = takes_ownership(owned);
    println!("10. {len} {consumed}");
}

fn takes_ref(v: &[i32]) -> usize {
    v.len()
}

fn takes_ownership(v: Vec<i32>) -> i32 {
    v.into_iter().sum()
}
