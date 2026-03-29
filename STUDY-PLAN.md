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
| **19** | Practice: implement from scratch | [K-Means](algorithms/14-ml-statistics/k-means.py), [TF-IDF Vector Search](algorithms/14-ml-statistics/tf-idf-vector-search.py) | Re-read weakest topic READMEs |
| **20** | Company-targeted review | Pick 2-3 companies from [interviews/](interviews/README.md) and study their guides | [Shared concepts across companies](interviews/shared-concepts/README.md) |
| **21** | Mock interview day | Revisit 1 problem from each of topics 01-11 under timed conditions (45 min each) | Review all pattern READMEs as quick reference |

### Daily Routine (Algorithms)

1. **Read patterns first** (15 min) -- read the topic README before touching code
2. **Solve problems** (60-90 min) -- attempt each problem for 20 min before looking at the solution
3. **Review and annotate** (30 min) -- understand the solution, note edge cases, trace through examples
4. **Spaced review** (15 min) -- re-solve one problem from a previous day without looking at code

---

## ML & AI (14-19 weeks)

~10-12 hours per week. Requires math foundations (linear algebra, calculus, probability).

### Weeks 1-5: Statistical Learning

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Statistical learning framework, bias-variance | ISLR Ch 2 |
| 2 | Linear regression, model selection | ISLR Ch 3 |
| 3 | Classification (logistic regression, LDA, Naive Bayes) | ISLR Ch 4 |
| 4 | Resampling, regularization (ridge, lasso) | ISLR Ch 5-6 |
| 5 | Trees, random forests, boosting, SVMs, unsupervised | ISLR Ch 8-9, 12 |

### Weeks 6-13: Deep Learning

| Week | Focus | Reference |
|------|-------|-----------|
| 6 | ML basics, feedforward networks, backpropagation | Goodfellow Ch 5-6 |
| 7 | Regularization, optimization (SGD, Adam) | Goodfellow Ch 7-8 |
| 8 | CNNs -- architectures (LeNet to ResNet), applications | Goodfellow Ch 9 |
| 9 | RNNs, LSTMs, sequence-to-sequence | Goodfellow Ch 10 |
| 10 | Practical methodology, debugging, hyperparameter tuning | Goodfellow Ch 11 |
| 11 | Autoencoders, generative models (VAE, GAN) | Goodfellow Ch 14, 20 |
| 12 | Transformers, attention, BERT, GPT, scaling laws | Beyond book -- papers |
| 13 | Representation learning, self-supervised, transfer learning | Goodfellow Ch 15 + papers |

### Weeks 14-19: Reinforcement Learning

| Week | Focus | Reference |
|------|-------|-----------|
| 14 | Bandits, MDPs, Bellman equations | Sutton & Barto Ch 2-3 |
| 15 | DP, Monte Carlo, TD learning, Q-learning | Sutton & Barto Ch 4-6 |
| 16 | Function approximation, DQN | Sutton & Barto Ch 9-10 |
| 17 | Policy gradient, actor-critic, A2C/A3C | Sutton & Barto Ch 13 |
| 18 | PPO, DDPG, TD3, SAC | Spinning Up |
| 19 | RLHF, alignment, model-based RL | Research papers |

---

## Systems & Architecture (11-15 weeks)

~8-10 hours per week.

### Weeks 1-5: System Design

| Week | Focus | Reference |
|------|-------|-----------|
| 1 | Scalability, load balancing, caching, CDN | ByteByteGo |
| 2 | Database design, SQL vs NoSQL, CAP theorem, replication | ByteByteGo + DDIA |
| 3 | Distributed systems primitives (consensus, distributed transactions) | ByteByteGo |
| 4 | Messaging, event systems, API design | ByteByteGo |
| 5 | Classic problems (URL shortener, chat, news feed, etc.) | ByteByteGo |

### Weeks 6-8: Software Architecture

| Week | Focus | Reference |
|------|-------|-----------|
| 6 | Layered, event-driven, microkernel patterns | Software Architecture Patterns |
| 7 | Microservices, space-based architecture | Software Architecture Patterns |
| 8 | ADRs, quality attributes, trade-off analysis | Software Architecture Patterns |

### Weeks 9-12: Cloud Native

| Week | Focus | Reference |
|------|-------|-----------|
| 9 | Containers, Docker, multi-stage builds | Cloud Native DevOps with K8s |
| 10 | Kubernetes fundamentals (pods, deployments, services) | Cloud Native DevOps with K8s |
| 11 | Advanced K8s (StatefulSets, CRDs, operators, RBAC) | Cloud Native DevOps with K8s |
| 12 | Helm, GitOps, CI/CD, service mesh | Cloud Native DevOps with K8s |

### Weeks 13-15: Observability

| Week | Focus | Reference |
|------|-------|-----------|
| 13 | Observability vs monitoring, instrumentation, OpenTelemetry | Observability Engineering |
| 14 | SLOs, error budgets, debugging with observability | Observability Engineering + SRE Book |
| 15 | Production excellence, incident response, chaos engineering | Observability Engineering |

---

## Combined Path (40+ weeks)

The full journey from foundations through PhD-level depth and Staff+ engineering expertise.

| Weeks | Focus | Details |
|-------|-------|---------|
| 1-4 | Calculus 1 + Linear Algebra | Parallel math study |
| 5-8 | Calculus 2 + Discrete Math 1 | Parallel math study |
| 9-12 | Calculus 3 + Discrete Math 2 | Parallel math study |
| 13 | Math review & catch-up | Revisit hardest topics |
| 14-16 | Probability & Statistics | Completes math track |
| 17-19 | Algorithm Mastery | 21-day plan (one week per study-plan week) |
| 20-21 | Competitive Programming | Advanced techniques, contest practice |
| 22-26 | Statistical Learning + Deep Learning | ISLR then Goodfellow |
| 27-30 | Deep Learning (continued) + Info Theory | Research-level depth |
| 31-34 | Reinforcement Learning | Sutton & Barto + Spinning Up |
| 35-39 | Systems & Architecture | System design through observability |
| 40-42 | Interview preparation | Company-targeted study, mock interviews |

---

## PhD Research Track (20+ weeks)

For those pursuing research depth in ML/AI. Assumes math foundations and statistical learning are complete.

| Weeks | Focus | Reading |
|-------|-------|---------|
| 1-8 | Deep Learning (full book) | Goodfellow Parts I-III, all 20 chapters |
| 9-10 | Information Theory | MacKay -- entropy, KL divergence, channel coding |
| 11-14 | Reinforcement Learning | Sutton & Barto full + Spinning Up key papers |
| 15-17 | Transformer architectures | Attention Is All You Need, BERT, GPT, scaling laws papers |
| 18-19 | Alignment & Safety | Constitutional AI, RLHF, DPO papers |
| 20+ | Research exploration | Pick a subfield, read 10+ papers, implement one |

### How to Use These Plans

1. **Pick your starting point** -- if you have a math background, skip to algorithms or ML
2. **Parallelize where possible** -- math topics are designed for parallel study
3. **Don't skip the connections** -- the cross-references between tracks are where real understanding lives
4. **Adjust the pace** -- these timelines assume ~10 hrs/week; scale up or down as needed
5. **Use interview guides last** -- they're most effective after building deep understanding
