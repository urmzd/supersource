// 06-async-ordering: the event loop drains every microtask before it touches
// the next macrotask, and the body of an async function runs synchronously up
// to its first await. Predict the ORDER of these lines, not their content.

console.log("A: sync start");

setTimeout(() => console.log("B: setTimeout 0"), 0);

queueMicrotask(() => console.log("C: queueMicrotask"));

Promise.resolve().then(() => console.log("D: promise.then"));

const run = async () => {
  console.log("E: async body, before await");
  await null;
  console.log("F: async body, after await");
};
run();

console.log("G: sync end");

// A second tick of microtasks, queued from inside a microtask, still runs
// before the timer above.
Promise.resolve().then(() => {
  console.log("H: first then");
  Promise.resolve().then(() => console.log("I: nested then"));
});

// await on a non-promise still yields to the microtask queue exactly once.
(async () => {
  await 1;
  console.log("J: after await 1");
})();

// Promise executors run synchronously; only the callbacks are deferred.
new Promise<void>((resolve) => {
  console.log("K: executor body");
  resolve();
}).then(() => console.log("L: after executor"));

console.log("M: really the end");
