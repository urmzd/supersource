//! 03-iterator-laziness: adapters build a description of work, they do not do
//! it. Nothing runs until something consumes the chain, and then only as much
//! as the consumer asks for.

fn main() {
    let v = vec![1, 2, 3, 4, 5];

    let doubled = v.iter().map(|x| {
        println!("  mapping {x}");
        x * 2
    });
    println!("1. adapter built, nothing mapped yet");

    let first_two: Vec<i32> = doubled.take(2).collect();
    println!("2. {first_two:?}");

    // find stops at the first match, so the side effect stops too.
    let found = v
        .iter()
        .inspect(|x| println!("  checking {x}"))
        .find(|&&x| x == 3);
    println!("3. {found:?}");

    // Chained adapters make exactly one pass, not one per adapter.
    let n = v.iter().filter(|&&x| x % 2 == 1).map(|x| x * 10).count();
    println!("4. {n}");

    // size_hint asks the chain how long it will be, which needs no elements
    // at all, so the closure below never runs.
    let counted = v
        .iter()
        .map(|x| {
            println!("  never printed {x}");
            x
        })
        .size_hint();
    println!("5. {counted:?}");

    // Consuming twice is a compile error, so collect once and reuse.
    let evens: Vec<&i32> = v.iter().filter(|&&x| x % 2 == 0).collect();
    println!("6. {evens:?} {}", evens.len());

    // zip stops at the shorter side.
    let pairs: Vec<(i32, char)> = v.iter().copied().zip("ab".chars()).collect();
    println!("7. {pairs:?}");

    // sum/fold consume; the iterator is gone afterwards.
    let total: i32 = v.iter().sum();
    println!("8. {total}");
}
