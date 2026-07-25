// 04-this-binding: `this` is decided by how a function is called, not where it
// was written. Arrow functions are the exception: they capture `this` from the
// enclosing scope and can never be rebound.

class Counter {
  n = 0;

  inc() {
    this.n += 1;
    return this.n;
  }

  incArrow = () => {
    this.n += 1;
    return this.n;
  };
}

const c = new Counter();
console.log("1.", c.inc(), c.inc());

// Detaching a method loses the receiver. Class bodies are strict mode, so
// `this` is undefined rather than the global object.
const loose = c.inc;
try {
  console.log("2.", loose());
} catch (err) {
  console.log("2. threw", (err as Error).constructor.name);
}

const bound = c.inc.bind(c);
console.log("3.", bound());

// The arrow field was created per instance with `this` already captured.
const looseArrow = c.incArrow;
console.log("4.", looseArrow(), c.n);

// call/apply set the receiver explicitly.
console.log("5.", c.inc.call({ n: 100 }), c.n);

const obj = {
  n: 7,
  get() {
    return this.n;
  },
  getArrow: () => (globalThis as { n?: number }).n,
};
console.log("6.", obj.get(), obj.getArrow());

// Extracting into a callback is the usual way this bug ships.
console.log("7.", [1, 2].map(c.incArrow).join(","));

// bind is permanent: a second bind cannot override the first.
const twice = c.inc.bind({ n: 1000 }).bind({ n: 0 });
console.log("8.", twice());
