// 03-equality-coercion: `==` runs an algorithm, not a comparison. These values
// are typed `unknown` and cast, because that is exactly how coercion bugs reach
// production: the data crossed a boundary the type system could not see.

const one: unknown = 1;
const oneText: unknown = "1";

console.log("1.", one == oneText, one === oneText);
console.log("2.", null == undefined, null === undefined);

// null is coerced for relational operators but not for ==, so this trio splits.
console.log("3.", null == 0, null >= 0, null > 0);

console.log("4.", [] == false, "" == 0, "0" == 0, "" == "0");
console.log("5.", [] + {}, typeof ([] + []));

// NaN is the only value not equal to itself, and includes disagrees with indexOf.
console.log("6.", NaN === NaN, Object.is(NaN, NaN));

// typeof has exactly one famous bug and it has never been fixed.
console.log("7.", typeof null, typeof [], typeof NaN, typeof undefined);
console.log("8.", Array.isArray([]), null instanceof Object);

// Truthiness is not emptiness.
console.log("9.", Boolean([]), Boolean({}), Boolean(""), Boolean("0"), Boolean(0));

// ?? falls back only on null/undefined; || falls back on every falsy value.
const zero = 0;
const blank = "";
console.log("10.", zero || "fallback", zero ?? "fallback", blank || "x", blank ?? "x");

// Optional chaining short-circuits the whole chain, not just one link.
const maybe: { a?: { b: number } } = {};
console.log("11.", maybe.a?.b, maybe.a?.b ?? "absent");

// String conversion flattens arrays and gives up on objects.
console.log("12.", String([1, [2, [3]]]), String({}), String(null));
