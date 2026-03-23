# Puzzles

Interview-style algorithms organized by priority (most useful for general SWE roles first), with specialized tracks near the end (systems, FP, ML/Stats). Each topic directory contains a README with pattern guides and solution walkthroughs.

## Topics

| # | Topic | Problems | Lang | README |
|---|-------|----------|------|--------|
| 01 | [Arrays & Hashing](01-arrays-hashing/) | [Two Sum](01-arrays-hashing/two-sum.js), [Contains Duplicate](01-arrays-hashing/contains-duplicate.js), [Product of Array Except Self](01-arrays-hashing/product-of-array-expect-self.js), [Missing Number](01-arrays-hashing/missing-number.js) | JS | [Patterns](01-arrays-hashing/README.md) |
| 02 | [Two Pointers & Sliding Window](02-two-pointers-sliding-window/) | [3Sum](02-two-pointers-sliding-window/3sum.js), [Container With Most Water](02-two-pointers-sliding-window/container-with-most-water.js) | JS | [Patterns](02-two-pointers-sliding-window/README.md) |
| 03 | [Binary Search](03-binary-search/) | [Find Min in Rotated Array](03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Array](03-binary-search/search-in-rotated-sorted-array.js), [LISS](03-binary-search/liss.js) | JS | [Patterns](03-binary-search/README.md) |
| 04 | [Linked Lists](04-linked-lists/) | [Add Two Numbers](04-linked-lists/add-two-numbers.js) | JS | [Patterns](04-linked-lists/README.md) |
| 05 | [Trees](05-trees/) | [AVL Tree + Range Query](05-trees/example-7.py) | Python | [Patterns](05-trees/README.md) |
| 06 | [Graphs](06-graphs/) | [Clone Graph](06-graphs/clone-graph.js), [Course Schedule](06-graphs/course-schedule.js), [Number of Islands](06-graphs/number-of-islands.js), [Pacific Atlantic Water Flow](06-graphs/pacific-atlantic-water-flow.ts), [Corgi Conundrum](06-graphs/charles-and-the-corgi-conundrum.c) | JS/TS/C | [Patterns](06-graphs/README.md) |
| 07 | [Dynamic Programming](07-dynamic-programming/) | [Climbing Stairs](07-dynamic-programming/climbing-stairs.js), [House Robber](07-dynamic-programming/house-robber.js), [House Robber II](07-dynamic-programming/rob-houses-pt-2.js), [Coin Change](07-dynamic-programming/change-coin.js), [Unique Paths](07-dynamic-programming/unique-paths.js), [Decode Ways](07-dynamic-programming/decode-ways.js), [Word Break](07-dynamic-programming/word-break.js), [Max Subarray](07-dynamic-programming/maximum-subarray.js), [Max Product Subarray](07-dynamic-programming/maximum-product-subarray.js), [LCS](07-dynamic-programming/lcs.js), [Knapsack](07-dynamic-programming/example-1.py), [Edit Distance](07-dynamic-programming/example-2.py), [Kadane](07-dynamic-programming/example-3.py), [DP on Graph](07-dynamic-programming/example-4.py), [Optimal BST](07-dynamic-programming/example-5.c), [Probabilistic Transitions](07-dynamic-programming/example-6.py), [Alignment](07-dynamic-programming/example-8.c) | JS/Python/C | [Patterns](07-dynamic-programming/README.md) |
| 08 | [Greedy](08-greedy/) | [Best Time to Buy/Sell Stock](08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](08-greedy/jump-game.js), [Example 1](08-greedy/example-1.py), [Example 2](08-greedy/example-2.py), [Example 3](08-greedy/example-3.py), [Huffman Coding](08-greedy/huffman-coding/) | JS/Python/Java | [Patterns](08-greedy/README.md) |
| 09 | [Backtracking](09-backtracking/) | [Combination Sum](09-backtracking/combination-sum.js), [Map Coloring](09-backtracking/example-2.py) | JS/Python | [Patterns](09-backtracking/README.md) |
| 10 | [Math & Bit Manipulation](10-math-bit/) | [Count Bits](10-math-bit/count-number-of-bits.js), [Number of 1 Bits](10-math-bit/number-of-1-bits.js), [Reverse Bits](10-math-bit/reverse-bits.js), [Sum of Two Integers](10-math-bit/sum-of-two-integers.js) | JS | [Patterns](10-math-bit/README.md) |
| 11 | [Recursion & Divide-and-Conquer](11-recursion-divide-conquer/) | [Tree Path](11-recursion-divide-conquer/example-1.py), [Max Subarray D&C](11-recursion-divide-conquer/example-3.py) | Python | [Patterns](11-recursion-divide-conquer/README.md) |
| 12 | [Concurrency & Systems](12-concurrency-systems/) | [C Miner](12-concurrency-systems/miner/), [Test Suite](12-concurrency-systems/tests/) | C/Python | [Patterns](12-concurrency-systems/README.md) |
| 13 | [Functional Programming](13-functional-programming/) | [BST](13-functional-programming/bst.scm), [Factors](13-functional-programming/Factors.scm), [Primes](13-functional-programming/IsPrime.scm), [Visitors](13-functional-programming/list-visitor.scm), [Iterators](13-functional-programming/new-sqrt-iterator.scm), [I/O](13-functional-programming/io.scm) | Scheme | [Patterns](13-functional-programming/README.md) |
| 14 | [ML & Statistics](14-ml-statistics/) | [Linear Regression](14-ml-statistics/linear-regression.py), [Clustering](14-ml-statistics/clustering.py), [PyTorch](14-ml-statistics/learn-pytorch.py), [Framework](14-ml-statistics/framework.py), [Linear Algebra & Models](14-ml-statistics/ml-linear-algebra-and-linear-models.py), [Logistic Reg & NNs](14-ml-statistics/ml-logistic-regression-and-neural-nets.py), [Naive Bayes & GMM](14-ml-statistics/ml-naive-bayes-and-gmm.py), [CNN & RNN](14-ml-statistics/ml-clustering-cnn-rnn.py), [NLP Labs](14-ml-statistics/nlp/), [R Exercises](14-ml-statistics/r-exercises/) | Python/R | [Patterns](14-ml-statistics/README.md) |
| 15 | [Probabilistic Structures](15-probabilistic-structures/) | [Bloom Filter](15-probabilistic-structures/bloom.py) | Python | [Patterns](15-probabilistic-structures/README.md) |

## 21-Day Study Plan

A structured plan covering all topics. Each day targets ~2-3 hours. Days 1-14 cover core SWE, days 15-18 cover specialized tracks, and days 19-21 are for interview-targeted review.

### Week 1: Foundations

| Day | Focus | Problems | Patterns to Read |
|-----|-------|----------|-----------------|
| **1** | Arrays & Hashing | [Two Sum](01-arrays-hashing/two-sum.js), [Contains Duplicate](01-arrays-hashing/contains-duplicate.js), [Missing Number](01-arrays-hashing/missing-number.js), [Product Except Self](01-arrays-hashing/product-of-array-expect-self.js) | [Hash map complement lookup, frequency counting, index-as-hash](01-arrays-hashing/README.md) |
| **2** | Two Pointers & Sliding Window | [3Sum](02-two-pointers-sliding-window/3sum.js), [Container With Most Water](02-two-pointers-sliding-window/container-with-most-water.js) | [Opposite-end pointers, fast/slow pointers, window expand/shrink](02-two-pointers-sliding-window/README.md) |
| **3** | Binary Search | [Find Min in Rotated Array](03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Array](03-binary-search/search-in-rotated-sorted-array.js), [LISS](03-binary-search/liss.js) | [Invariant-based search, search on answer space](03-binary-search/README.md) |
| **4** | Linked Lists | [Add Two Numbers](04-linked-lists/add-two-numbers.js) | [Runner technique, dummy head, reversal](04-linked-lists/README.md) |
| **5** | Trees | [AVL + Range Query](05-trees/example-7.py) | [DFS/BFS traversal, path problems, BST properties](05-trees/README.md) |
| **6** | Graphs (traversal) | [Number of Islands](06-graphs/number-of-islands.js), [Clone Graph](06-graphs/clone-graph.js) | [DFS/BFS on grids, graph copying](06-graphs/README.md) |
| **7** | Graphs (dependencies) | [Course Schedule](06-graphs/course-schedule.js), [Pacific Atlantic](06-graphs/pacific-atlantic-water-flow.ts) | [Topological sort, multi-source BFS](06-graphs/README.md) |

### Week 2: Optimization & Search

| Day | Focus | Problems | Patterns to Read |
|-----|-------|----------|-----------------|
| **8** | DP: 1D basics | [Climbing Stairs](07-dynamic-programming/climbing-stairs.js), [House Robber](07-dynamic-programming/house-robber.js), [House Robber II](07-dynamic-programming/rob-houses-pt-2.js), [Coin Change](07-dynamic-programming/change-coin.js) | [Fibonacci-style, take/skip, unbounded knapsack](07-dynamic-programming/README.md) |
| **9** | DP: sequences | [Decode Ways](07-dynamic-programming/decode-ways.js), [Word Break](07-dynamic-programming/word-break.js), [LCS](07-dynamic-programming/lcs.js), [Max Subarray](07-dynamic-programming/maximum-subarray.js) | [Partition DP, two-string DP, Kadane's](07-dynamic-programming/README.md) |
| **10** | DP: advanced | [Max Product Subarray](07-dynamic-programming/maximum-product-subarray.js), [Unique Paths](07-dynamic-programming/unique-paths.js), [Knapsack](07-dynamic-programming/example-1.py), [Edit Distance](07-dynamic-programming/example-2.py) | [Grid DP, min/max tracking, 0/1 knapsack](07-dynamic-programming/README.md) |
| **11** | DP: hard variants | [Kadane](07-dynamic-programming/example-3.py), [DP on Graph](07-dynamic-programming/example-4.py), [Optimal BST](07-dynamic-programming/example-5.c), [Alignment](07-dynamic-programming/example-8.c) | [Interval DP, Knuth optimization, reconstruction](07-dynamic-programming/README.md) |
| **12** | Greedy | [Buy/Sell Stock](08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](08-greedy/jump-game.js), [Huffman Coding](08-greedy/huffman-coding/), [Examples 1-3](08-greedy/example-1.py) | [Greedy choice property, exchange argument, priority queues](08-greedy/README.md) |
| **13** | Backtracking & Search | [Combination Sum](09-backtracking/combination-sum.js), [Map Coloring](09-backtracking/example-2.py) | [Choice/explore/unchoose, pruning, constraint propagation](09-backtracking/README.md) |
| **14** | Math, Bits & Recursion | [Count Bits](10-math-bit/count-number-of-bits.js), [1 Bits](10-math-bit/number-of-1-bits.js), [Reverse Bits](10-math-bit/reverse-bits.js), [Sum Without +](10-math-bit/sum-of-two-integers.js), [Tree Path](11-recursion-divide-conquer/example-1.py), [Max Subarray D&C](11-recursion-divide-conquer/example-3.py) | [Bit tricks, divide & conquer](10-math-bit/README.md), [Recursion patterns](11-recursion-divide-conquer/README.md) |

### Week 3: Specialized Tracks & Interview Prep

| Day | Focus | Problems | Patterns to Read |
|-----|-------|----------|-----------------|
| **15** | Concurrency & Systems | [C Miner project](12-concurrency-systems/miner/) | [Thread pools, producer-consumer, lock-free structures](12-concurrency-systems/README.md) |
| **16** | Functional Programming | [BST](13-functional-programming/bst.scm), [Visitors](13-functional-programming/list-visitor.scm), [Iterators](13-functional-programming/new-sqrt-iterator.scm), [Primes](13-functional-programming/IsPrime.scm) | [Immutable data, higher-order functions, recursion schemes](13-functional-programming/README.md) |
| **17** | ML & Statistics (models) | [Linear Regression](14-ml-statistics/linear-regression.py), [Logistic Reg & NNs](14-ml-statistics/ml-logistic-regression-and-neural-nets.py), [PyTorch](14-ml-statistics/learn-pytorch.py) | [Gradient descent, loss functions, backprop](14-ml-statistics/README.md) |
| **18** | ML & Statistics (unsupervised) + Probabilistic | [Clustering](14-ml-statistics/clustering.py), [Naive Bayes & GMM](14-ml-statistics/ml-naive-bayes-and-gmm.py), [CNN & RNN](14-ml-statistics/ml-clustering-cnn-rnn.py), [Bloom Filter](15-probabilistic-structures/bloom.py) | [Clustering, probabilistic models](14-ml-statistics/README.md), [Bloom filters, HyperLogLog, skip lists](15-probabilistic-structures/README.md) |
| **19** | Practice: implement from scratch | [K-Means](practice/k_means.py), [TF-IDF Vector Search](practice/tf_idf_vector_search.py) | Re-read weakest topic READMEs |
| **20** | Company-targeted review | Pick 2-3 companies from [interviews/](interviews/README.md) and study their guides | [Shared concepts across companies](interviews/shared-concepts/README.md) |
| **21** | Mock interview day | Revisit 1 problem from each of topics 01-11 under timed conditions (45 min each) | Review all pattern READMEs as quick reference |

### Daily Routine

1. **Read patterns first** (15 min) -- read the topic README before touching code
2. **Solve problems** (60-90 min) -- attempt each problem for 20 min before looking at the solution
3. **Review and annotate** (30 min) -- understand the solution, note edge cases, trace through examples
4. **Spaced review** (15 min) -- re-solve one problem from a previous day without looking at code

## Interviews

Company-specific interview guides covering focus areas, question styles, and preparation strategies for 19 companies.

See [`interviews/README.md`](interviews/README.md) for the full breakdown:

- **Big Tech & AI Labs**: [Anthropic](interviews/anthropic/), [Google](interviews/google/), [DeepMind](interviews/deepmind/), [OpenAI](interviews/openai/), [Meta](interviews/meta/), [Apple](interviews/apple/), [NVIDIA](interviews/nvidia/), [Moonshot](interviews/moonshot/)
- **Infrastructure & Data**: [Netflix](interviews/netflix/), [Amazon](interviews/amazon/), [Databricks](interviews/databricks/), [Stripe](interviews/stripe/), [Palantir](interviews/palantir/)
- **Quant & Trading**: [Jane Street](interviews/jane-street/), [Citadel](interviews/citadel/), [Two Sigma](interviews/two-sigma/), [HRT](interviews/hrt/), [Renaissance](interviews/renaissance-technologies/)
- **Frontier**: [SpaceX](interviews/spacex/)
- **Cross-cutting**: [Shared Concepts](interviews/shared-concepts/)

## Practice

Standalone implementations for hands-on practice:

- [K-Means Clustering](practice/k_means.py) -- from-scratch implementation
- [TF-IDF Vector Search](practice/tf_idf_vector_search.py) -- text similarity search
