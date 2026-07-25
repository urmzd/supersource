// 05-types-are-erased: types exist only at compile time. Node runs this file by
// stripping them, so an assertion is a promise the runtime never checks and a
// generic is a comment with syntax highlighting.

interface User {
  id: number;
  name: string;
}

const raw: unknown = { id: "not-a-number" };
const user = raw as User;

console.log("1.", typeof user.id, user.name);
console.log("2.", user.id + 1);

// JSON.parse returns any, which is where most unsound values enter a program.
type Shape = { kind: "circle"; r: number } | { kind: "square"; s: number };
const shape: Shape = JSON.parse('{"kind":"circle","r":2}');
console.log("3.", shape.kind === "circle" ? shape.r * 2 : -1);

// A narrowed union is still just an object at runtime.
const lying: Shape = JSON.parse('{"kind":"square"}');
console.log("4.", lying.kind === "square" ? lying.s : -1);

// Generics are erased, so there is no T to inspect at runtime.
function first<T>(xs: T[]): T | undefined {
  return xs[0];
}
console.log("5.", first<number>([]), first(["a"]));

// Interfaces leave nothing behind, so the only runtime check is one you wrote.
const looksLikeUser = (v: unknown): v is User =>
  typeof v === "object" && v !== null && typeof (v as User).id === "number";
console.log("6.", looksLikeUser(user), looksLikeUser({ id: 1, name: "x" }));

// JSON drops undefined and functions, and turns them into holes in arrays.
console.log("7.", JSON.stringify({ a: undefined, b: null, c: () => 1, d: 1 }));
console.log("8.", JSON.stringify([undefined, () => 1, 3]));

// Object keys that look like array indices are ordered first, ascending.
console.log("9.", Object.keys({ b: 1, 2: "x", a: 3, 1: "y" }).join(","));

// Structural typing means "shape matches" is the only check there ever was.
type Meters = number;
type Feet = number;
const distance: Meters = 100;
const wrong: Feet = distance;
console.log("10.", wrong === 100);
