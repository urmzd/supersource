# Study Plans

Structured schedules for self-paced learning. Pick the plan that matches your goals.

## Math Foundations (16 weeks)

~10-12 hours per week. Topics are paired so you work on two subjects in parallel.

### Weeks 1-4: Calculus 1 + Linear Algebra

| Week | Calculus 1 | Linear Algebra |
|------|-----------|----------------|
| 1 | Limits and continuity | Linear systems, Gauss's method |
| 2 | Derivatives (rules and computation) | Vector spaces, subspaces |
| 3 | Applications of derivatives (optimization, MVT) | Linear transformations, matrices |
| 4 | Integration and FTC | Determinants, eigenvalues intro |

### Weeks 5-8: Calculus 2 + Discrete Math 1

| Week | Calculus 2 | Discrete Math 1 |
|------|-----------|-----------------|
| 5 | Integration techniques (by parts, partial fractions) | Sets, logic, truth tables |
| 6 | Applications of integration | Direct proof, contrapositive |
| 7 | Sequences and convergence tests | Proof by contradiction, induction |
| 8 | Power series, Taylor series | Relations, functions, cardinality |

### Weeks 9-12: Calculus 3 + Discrete Math 2

| Week | Calculus 3 | Discrete Math 2 |
|------|-----------|-----------------|
| 9 | Vectors in space, dot/cross product | Graph theory fundamentals |
| 10 | Partial derivatives, gradient | Graph coloring, Euler/Hamilton |
| 11 | Multiple integrals | Number theory, modular arithmetic |
| 12 | Vector calculus (Green's, Stokes') | Recurrence relations, generating functions |

### Weeks 13: Review & Catch-up

Revisit the hardest topics from each subject. Work through challenge problems.

### Weeks 14-16: Probability & Statistics

| Week | Focus |
|------|-------|
| 14 | Probability axioms, conditional probability, Bayes' theorem, discrete distributions |
| 15 | Continuous distributions, joint distributions, expectation & variance |
| 16 | Law of large numbers, CLT, hypothesis testing, confidence intervals, regression |

### Daily Routine (Math)

1. **Read** (30 min) -- textbook sections for the day's concept
2. **Work problems** (60-90 min) -- essential problems first, then challenge problems
3. **Review** (30 min) -- revisit previous day's concepts, rework one problem from memory
4. **Connect** (15 min) -- read the "Connections to CS" section, think about how the math applies

---

## Algorithm Mastery (21 days)

~2-3 hours per day. Each day targets specific patterns and problems.

### Week 1: Foundations

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **1** | Arrays & Hashing | [Two Sum](algorithms/01-arrays-hashing/two-sum.js), [Contains Duplicate](algorithms/01-arrays-hashing/contains-duplicate.js), [Missing Number](algorithms/01-arrays-hashing/missing-number.js), [Product Except Self](algorithms/01-arrays-hashing/product-of-array-expect-self.js) | [Hash map complement lookup, frequency counting, index-as-hash](algorithms/01-arrays-hashing/README.md) |
| **2** | Two Pointers & Sliding Window | [3Sum](algorithms/02-two-pointers-sliding-window/3sum.js), [Container With Most Water](algorithms/02-two-pointers-sliding-window/container-with-most-water.js) | [Opposite-end pointers, fast/slow pointers, window expand/shrink](algorithms/02-two-pointers-sliding-window/README.md) |
| **3** | Binary Search | [Find Min in Rotated Array](algorithms/03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Array](algorithms/03-binary-search/search-in-rotated-sorted-array.js), [LISS](algorithms/03-binary-search/liss.js) | [Invariant-based search, search on answer space](algorithms/03-binary-search/README.md) |
| **4** | Linked Lists | [Add Two Numbers](algorithms/04-linked-lists/add-two-numbers.js) | [Runner technique, dummy head, reversal](algorithms/04-linked-lists/README.md) |
| **5** | Trees | [AVL + Range Query](algorithms/05-trees/example-7.py) | [DFS/BFS traversal, path problems, BST properties](algorithms/05-trees/README.md) |
| **6** | Graphs (traversal) | [Number of Islands](algorithms/06-graphs/number-of-islands.js), [Clone Graph](algorithms/06-graphs/clone-graph.js) | [DFS/BFS on grids, graph copying](algorithms/06-graphs/README.md) |
| **7** | Graphs (dependencies) | [Course Schedule](algorithms/06-graphs/course-schedule.js), [Pacific Atlantic](algorithms/06-graphs/pacific-atlantic-water-flow.ts) | [Topological sort, multi-source BFS](algorithms/06-graphs/README.md) |

### Week 2: Optimization & Search

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **8** | DP: 1D basics | [Climbing Stairs](algorithms/07-dynamic-programming/climbing-stairs.js), [House Robber](algorithms/07-dynamic-programming/house-robber.js), [House Robber II](algorithms/07-dynamic-programming/rob-houses-pt-2.js), [Coin Change](algorithms/07-dynamic-programming/change-coin.js) | [Fibonacci-style, take/skip, unbounded knapsack](algorithms/07-dynamic-programming/README.md) |
| **9** | DP: sequences | [Decode Ways](algorithms/07-dynamic-programming/decode-ways.js), [Word Break](algorithms/07-dynamic-programming/word-break.js), [LCS](algorithms/07-dynamic-programming/lcs.js), [Max Subarray](algorithms/07-dynamic-programming/maximum-subarray.js) | [Partition DP, two-string DP, Kadane's](algorithms/07-dynamic-programming/README.md) |
| **10** | DP: advanced | [Max Product Subarray](algorithms/07-dynamic-programming/maximum-product-subarray.js), [Unique Paths](algorithms/07-dynamic-programming/unique-paths.js), [Knapsack](algorithms/07-dynamic-programming/example-1.py), [Edit Distance](algorithms/07-dynamic-programming/example-2.py) | [Grid DP, min/max tracking, 0/1 knapsack](algorithms/07-dynamic-programming/README.md) |
| **11** | DP: hard variants | [Kadane](algorithms/07-dynamic-programming/example-3.py), [DP on Graph](algorithms/07-dynamic-programming/example-4.py), [Optimal BST](algorithms/07-dynamic-programming/example-5.c), [Alignment](algorithms/07-dynamic-programming/example-8.c) | [Interval DP, Knuth optimization, reconstruction](algorithms/07-dynamic-programming/README.md) |
| **12** | Greedy | [Buy/Sell Stock](algorithms/08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](algorithms/08-greedy/jump-game.js), [Huffman Coding](algorithms/08-greedy/huffman-coding/), [Examples 1-3](algorithms/08-greedy/example-1.py) | [Greedy choice property, exchange argument, priority queues](algorithms/08-greedy/README.md) |
| **13** | Backtracking & Search | [Combination Sum](algorithms/09-backtracking/combination-sum.js), [Map Coloring](algorithms/09-backtracking/example-2.py) | [Choice/explore/unchoose, pruning, constraint propagation](algorithms/09-backtracking/README.md) |
| **14** | Math, Bits & Recursion | [Count Bits](algorithms/10-math-bit/count-number-of-bits.js), [1 Bits](algorithms/10-math-bit/number-of-1-bits.js), [Reverse Bits](algorithms/10-math-bit/reverse-bits.js), [Sum Without +](algorithms/10-math-bit/sum-of-two-integers.js), [Tree Path](algorithms/11-recursion-divide-conquer/example-1.py), [Max Subarray D&C](algorithms/11-recursion-divide-conquer/example-3.py) | [Bit tricks, divide & conquer](algorithms/10-math-bit/README.md), [Recursion patterns](algorithms/11-recursion-divide-conquer/README.md) |

### Week 3: Specialized Tracks & Interview Prep

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **15** | Concurrency & Systems | [C Miner project](algorithms/12-concurrency-systems/miner/) | [Thread pools, producer-consumer, lock-free structures](algorithms/12-concurrency-systems/README.md) |
| **16** | Functional Programming | [BST](algorithms/13-functional-programming/bst.scm), [Visitors](algorithms/13-functional-programming/list-visitor.scm), [Iterators](algorithms/13-functional-programming/new-sqrt-iterator.scm), [Primes](algorithms/13-functional-programming/IsPrime.scm) | [Immutable data, higher-order functions, recursion schemes](algorithms/13-functional-programming/README.md) |
| **17** | ML & Statistics (models) | [Linear Regression](algorithms/14-ml-statistics/linear-regression.py), [Logistic Reg & NNs](algorithms/14-ml-statistics/ml-logistic-regression-and-neural-nets.py), [PyTorch](algorithms/14-ml-statistics/learn-pytorch.py) | [Gradient descent, loss functions, backprop](algorithms/14-ml-statistics/README.md) |
| **18** | ML (unsupervised) + Probabilistic | [Clustering](algorithms/14-ml-statistics/clustering.py), [Naive Bayes & GMM](algorithms/14-ml-statistics/ml-naive-bayes-and-gmm.py), [CNN & RNN](algorithms/14-ml-statistics/ml-clustering-cnn-rnn.py), [Bloom Filter](algorithms/15-probabilistic-structures/bloom.py) | [Clustering, probabilistic models](algorithms/14-ml-statistics/README.md), [Bloom filters, HyperLogLog, skip lists](algorithms/15-probabilistic-structures/README.md) |
| **19** | Practice: implement from scratch | [K-Means](practice/k_means.py), [TF-IDF Vector Search](practice/tf_idf_vector_search.py) | Re-read weakest topic READMEs |
| **20** | Company-targeted review | Pick 2-3 companies from [interviews/](interviews/README.md) and study their guides | [Shared concepts across companies](interviews/shared-concepts/README.md) |
| **21** | Mock interview day | Revisit 1 problem from each of topics 01-11 under timed conditions (45 min each) | Review all pattern READMEs as quick reference |

### Daily Routine (Algorithms)

1. **Read patterns first** (15 min) -- read the topic README before touching code
2. **Solve problems** (60-90 min) -- attempt each problem for 20 min before looking at the solution
3. **Review and annotate** (30 min) -- understand the solution, note edge cases, trace through examples
4. **Spaced review** (15 min) -- re-solve one problem from a previous day without looking at code

---

## Combined Path (24 weeks)

For learners starting from scratch who want the full journey: math foundations through to interview readiness.

| Weeks | Focus | Details |
|-------|-------|---------|
| 1-4 | Calculus 1 + Linear Algebra | Parallel math study (see Math plan above) |
| 5-8 | Calculus 2 + Discrete Math 1 | Parallel math study |
| 9-12 | Calculus 3 + Discrete Math 2 | Parallel math study |
| 13 | Math review & catch-up | Revisit hardest topics |
| 14-16 | Probability & Statistics | Completes math track |
| 17-19 | Algorithm Mastery (21-day plan) | One week per study-plan week |
| 20-22 | Practice & deep dives | From-scratch implementations, hard variants |
| 23-24 | Interview preparation | Company-targeted study, mock interviews |

### How to Use the Combined Path

1. Start with the math plan -- it builds the reasoning skills that make algorithms easier
2. When you reach algorithms, the discrete math and probability background will make graph theory, DP recurrences, and probabilistic structures click immediately
3. During interview prep, use the company guides to focus your final review
