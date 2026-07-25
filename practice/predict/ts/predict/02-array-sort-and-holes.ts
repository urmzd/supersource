// 02-array-sort-and-holes: sort compares strings by default, arrays can have
// holes that are not undefined, and half the array methods skip those holes
// while the other half do not.
//
// Arrays are printed with join so the output is exact text, not Node's
// inspection format.

console.log("1.", [10, 9, 1, 100].sort().join(","));
console.log("2.", [10, 9, 1, 100].sort((a, b) => a - b).join(","));

// sort mutates in place and returns the same array.
const nums = [3, 1, 2];
const sorted = nums.sort();
console.log("3.", nums.join(","), sorted === nums);

// A hole is not undefined: `in` can tell them apart.
const sparse = [1, , 3];
console.log("4.", sparse.length, 1 in sparse, sparse[1]);
console.log("5.", sparse.map((x) => (x as number) * 2).join(","));
console.log("6.", sparse.filter(() => true).length);

// Array(3) is three holes; spreading it fills them with undefined.
console.log("7.", Array(3).length, [...Array(3)].join("|"), Array(3).join("-"));

// map passes (value, index, array), so a one-argument function is safest.
console.log("8.", ["1", "7", "11"].map(Number).join(","));
console.log("9.", ["1", "7", "11"].map(parseInt as (s: string) => number).join(","));

// Every non-primitive compares by reference, including in includes/indexOf.
console.log("10.", [{ a: 1 }].includes({ a: 1 }), [NaN].includes(NaN), [NaN].indexOf(NaN));

// Sorting mixed types stringifies everything first.
console.log("11.", [10, "9", true, null].sort().join(","));

// splice returns what it removed; slice returns what it kept.
const xs = [1, 2, 3, 4];
const removed = xs.splice(1, 2);
console.log("12.", removed.join(","), xs.join(","));
