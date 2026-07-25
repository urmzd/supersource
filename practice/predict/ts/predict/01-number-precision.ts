// 01-number-precision: every number in JavaScript is an IEEE-754 double, even
// the ones that look like integers. TypeScript's `number` type does not change
// that; it just means the compiler agrees with the arithmetic that will burn you.

console.log("1.", 0.1 + 0.2);
console.log("2.", 0.1 + 0.2 === 0.3);
console.log("3.", (0.1 + 0.2).toFixed(2));

// Past 2^53 the spacing between representable integers is larger than 1.
console.log("4.", Number.MAX_SAFE_INTEGER);
console.log("5.", Number.MAX_SAFE_INTEGER + 1 === Number.MAX_SAFE_INTEGER + 2);
console.log("6.", 9007199254740993);

// Negative zero is equal to zero but is not the same value.
console.log("7.", -0 === 0, Object.is(-0, 0), 1 / -0);

// Division never throws, it produces a non-finite number.
console.log("8.", 1 / 0, 0 / 0, Number.isNaN(0 / 0));

// Bitwise operators coerce to a 32-bit signed integer first.
console.log("9.", 2 ** 31 | 0, 4294967296 | 0, 5.9 | 0, -5.9 | 0);

// Parsing rules differ between the two obvious ways to do it.
console.log("10.", parseInt("08"), Number("08"), parseInt("0.9"), Number(""));

// Rounding half-up means -0.5 rounds to -0, which prints as -0.
console.log("11.", Math.round(-0.5), Math.round(0.5), Math.round(2.5));

// BigInt is the escape hatch, and it will not mix with number.
console.log("12.", 9007199254740993n, typeof 9007199254740993n);
