# Puzzles

Interview-style algorithms organized by priority (most useful for general SWE roles first), with specialized tracks near the end (systems, FP, ML/Stats). ML notebooks were extracted into runnable `.py` scripts under `14-ml-statistics/`.

## Taxonomy (priority order)

1. Arrays & Hashing (core SWE)
   - `01-arrays-hashing/two-sum.js`
   - `01-arrays-hashing/contains-duplicate.js`
   - `01-arrays-hashing/product-of-array-expect-self.js`
   - `01-arrays-hashing/missing-number.js`
2. Two Pointers & Sliding Window (core SWE)
   - `02-two-pointers-sliding-window/3sum.js`
   - `02-two-pointers-sliding-window/container-with-most-water.js`
3. Binary Search (core SWE)
   - `03-binary-search/find-minimum-in-rotated-sorted-array.js`
   - `03-binary-search/search-in-rotated-sorted-array.js`
   - `03-binary-search/liss.js`
4. Linked Lists (core SWE)
   - `04-linked-lists/add-two-numbers.js`
5. Trees (core SWE)
   - `05-trees/example-7.py` (AVL tree + range query / LIS-style reconstruction)
6. Graphs (core SWE)
   - `06-graphs/clone-graph.js`
   - `06-graphs/course-schedule.js`
   - `06-graphs/number-of-islands.js`
   - `06-graphs/pacific-atlantic-water-flow.ts`
7. Dynamic Programming (high-leverage SWE)
   - `07-dynamic-programming/climbing-stairs.js`
   - `07-dynamic-programming/house-robber.js`
   - `07-dynamic-programming/rob-houses-pt-2.js`
   - `07-dynamic-programming/change-coin.js`
   - `07-dynamic-programming/unique-paths.js`
   - `07-dynamic-programming/decode-ways.js`
   - `07-dynamic-programming/word-break.js`
   - `07-dynamic-programming/maximum-subarray.js`
   - `07-dynamic-programming/maximum-product-subarray.js`
   - `07-dynamic-programming/lcs.js`
   - `07-dynamic-programming/example-1.py` (knapsack variant)
   - `07-dynamic-programming/example-2.py` (edit distance)
   - `07-dynamic-programming/example-3.py` (Kadane)
   - `07-dynamic-programming/example-4.py` (DP on graph)
   - `07-dynamic-programming/example-5.c` (optimal BST + Knuth optimization)
   - `07-dynamic-programming/example-6.py` (probabilistic transitions)
   - `07-dynamic-programming/example-8.c` (alignment reconstruction)
8. Greedy (medium-high SWE)
   - `08-greedy/best-time-to-buy-and-sell-stock.js`
   - `08-greedy/jump-game.js`
   - `08-greedy/example-1.py`
   - `08-greedy/example-2.py`
   - `08-greedy/example-3.py`
9. Backtracking / Search (medium SWE)
   - `09-backtracking/combination-sum.js`
   - `09-backtracking/example-2.py` (map coloring brute-force)
10. Math & Bit Manipulation (medium SWE)
   - `10-math-bit/count-number-of-bits.js`
   - `10-math-bit/number-of-1-bits.js`
   - `10-math-bit/reverse-bits.js`
   - `10-math-bit/sum-of-two-integers.js`
11. Recursion & Divide-and-Conquer (medium SWE)
   - `11-recursion-divide-conquer/example-1.py` (tree path)
   - `11-recursion-divide-conquer/example-3.py` (max subarray, D&C)
12. Concurrency & Systems (systems/infra roles)
   - `12-concurrency-systems/miner/` (C miner implementation)
   - `12-concurrency-systems/tests/` (test fixtures + harness)
13. Functional Programming (FP/academia)
   - `13-functional-programming/` (Scheme exercises: BST, visitors, iterators, math)
14. ML & Statistics (ML/data roles)
   - `14-ml-statistics/linear-regression.py`
   - `14-ml-statistics/clustering.py`
   - `14-ml-statistics/learn-pytorch.py`
   - `14-ml-statistics/framework.py`
   - `14-ml-statistics/ml-linear-algebra-and-linear-models.py`
   - `14-ml-statistics/ml-logistic-regression-and-neural-nets.py`
   - `14-ml-statistics/ml-naive-bayes-and-gmm.py`
   - `14-ml-statistics/ml-clustering-cnn-rnn.py`
   - `14-ml-statistics/nlp/` (NLP labs: Python/Prolog/Perl)
   - `14-ml-statistics/r-exercises/` (R exercises + datasets)
