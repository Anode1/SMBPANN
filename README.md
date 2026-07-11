# SMBPANN

**Self-Modifying Backpropagation Feed-Forward Neural Network.** A network that is
not told its architecture but searches for it: an evolutionary search over
network *topology*, where each candidate's *weights* are trained by
backpropagation. In today's terms this is **evolutionary Neural Architecture
Search**.

It began as a 1997 seminar thesis deriving the backpropagation delta rule in
tensor form, was prototyped in Java in the early 2000s, and was set aside in 2015
on the arrival of TensorFlow. The idea, letting a search choose the architecture
instead of a human, lines up with what the field now calls **Neural Architecture
Search (NAS)** and **AutoML**. This is a clean re-implementation of that idea in
C99.

## The goal

Not a predictor whose architecture you hand-specify:

```text
predict(TrainingSet, TestingSet, Topology, Activation, Jitter, ...)
```

but one handed only the problem, which finds the rest itself:

```text
self_modifying_predict(TrainingSet, TestingSet, Error)
```

A population of networks is searched by mutating layer counts and widths and
selecting on held-out validation performance; each candidate's weights are
trained by backpropagation (the 1997 delta rule). The search returns the
architecture that generalizes best on validation, with the standard caveat that
the search itself can overfit the validation split, so a final untouched test set
is needed to judge it.

Recombining two arbitrary topologies by crossover is a known hard problem (the
competing-conventions problem, where two networks encode the same function with
permuted units), so the default here follows Real et al.'s mutation-only
evolution, with crossover left as an open design question.

## Where this sits today

The idea lines up with what the field now calls:

- **Neural Architecture Search (NAS)**: automating the design of network
  topology, brought to scale by Zoph and Le (2017) and surveyed by Elsken et al.
  (2019).
- **AutoML**: automating the whole model-construction pipeline (Hutter et al.,
  2019).

Following the Elsken survey, NAS methods differ mainly in *search strategy*:
reinforcement learning (Zoph and Le, 2017), gradient-based or differentiable
(DARTS, 2019), Bayesian optimization, random search, and evolutionary search.
**SMBPANN uses evolutionary search**, evolving the architecture while training
weights by gradient descent. Its closest relatives are Real et al. (*Large-Scale
Evolution*, 2017; *AmoebaNet*, 2019), which do exactly that. This contrasts with
classical **neuroevolution** such as NEAT (Stanley and Miikkulainen, 2002), which
evolves both topology *and* weights and uses no gradient information at all:
SMBPANN sits on the evolutionary-NAS side of that line, not the NEAT side.

Population-based search is not new ground for the author: an earlier project
applied a genetic algorithm to **label placement** (positioning labels without
overlap in a limited space, a classic combinatorial optimization problem), the
same family of search now turned on network topology.

**An honest, falsifiable question drives the project:** does an evolutionary
architecture search justify its compute over cheap baselines, random search in
particular? Bergstra and Bengio (2012) showed plain random search is a strong
baseline for hyper-parameter optimization; Li and Talwalkar (2019) showed random
search *with weight sharing* is competitive with ENAS and DARTS on standard NAS
benchmarks, while also documenting the field's reproducibility problems. SMBPANN
is built to test its own genetic search against a matched-compute random-search
control. This becomes a real contest only in a search space large and structured
enough for a GA to exploit topological structure a random sampler cannot;
enlarging the space beyond the current small one is a design target, not yet
reached.

## Language: C

SMBPANN is written in **C99**, in the coding discipline of the
[AIS](https://github.com/Anode1/ais) project: one concept per `.c` and `.h`,
bounded strings, return codes, single-exit cleanup, and a plain `make` build with
no framework or dependency. The choice is about fit: for this workload C is the
simplest tool that does the job well.

- **Concurrency by process, not by thread.** The evolutionary search is
  embarrassingly parallel, because candidates are independent. Each candidate is
  evaluated in its own **process**, launched from a shell coordinator sized to the
  CPU count, exchanging plain text with the parent. There is no shared memory, so
  there are no data races and no synchronization code to get wrong, and a
  candidate that crashes takes down only its own worker, not the run. For
  independent-candidate search this is the simplest robust concurrency there is.
- **Streaming data, bounded memory.** Backpropagation is inherently piecewise: the
  engine trains on one example at a time. A worker streams its dataset in pieces
  with a bounded footprint, so dataset size never dictates memory, and a read-only
  data file is shared across all workers for free by the operating system's page
  cache.
- **Manual memory, disciplined.** Following AIS, allocation happens only at
  construction (`net_new`, `trainer_new`), the train and infer hot path allocates
  nothing, and the population carves each generation from a Mark/Release arena.
  Peak footprint is computable by hand.
- **Safety by isolation and tools, stated honestly.** C is not memory-safe the way
  Java, Rust, or SPARK-verified Ada are; a bug can corrupt memory. Two things
  contain that here: process isolation (a fault stays in one worker), and the test
  suite run under AddressSanitizer and UBSan on every change, over the stack-first
  discipline that keeps most of the bug classes from arising in the first place.
  This is robustness suited to a research tool, not a formal proof.

### Why C, and not Ada or Rust?

Both Ada (with SPARK) and Rust are memory-safe and were seriously considered; an
Ada 2012 implementation of this engine was written and is preserved in the git
history. C won on fit for *this* problem. The search is embarrassingly parallel,
so process isolation gives robust, synchronization-free concurrency without a
language-level thread-safety guarantee; the data streams, so there is no large
shared structure to manage; and the memory pattern (allocate once, an arena per
generation) is simple enough that manual management plus sanitizers is adequate.
The safety the heavier languages add applies mostly to shared-memory threading
and complex ownership, neither of which this design uses. C also keeps the
toolchain and the code smallest and most portable, and reuses the disciplined C
foundation the project already had.

## Design principles

From the [AIS](https://github.com/Anode1/ais) project:

- one concept per `.c` and `.h`; clear, literal names;
- allocation only at construction, so the train and infer hot path allocates
  nothing;
- no framework, no dependency; a plain `make` and `cc` build;
- reproducible by construction: a seeded PRNG, no wall-clock in the engine;
- regression tests from the smallest cases up, run under sanitizers (XOR is the
  first gate).

## Layout

```
.                 the active engine (C99): rng, act, net, train, data, arena, main
legacy/java/      the original early-2000s Java prototype (object-graph design)
```

An Ada 2012 implementation lived here too and is preserved in the git history; C
is the active language.

## Status

**The genetic search, the project's actual subject, is not yet implemented.**
What runs today is the backpropagation engine each candidate will use: a
flat-array feed-forward network, backprop with momentum, plain-text dataset
loading with a train/test split, and the arena allocator, all under a
sanitizer-checked unit-test suite. The canonical gate is the **XOR** regression,
the linearly-non-separable problem a single perceptron cannot learn, and the
smallest proof that backprop through a hidden layer works.

```sh
make          # build ./smbpann
./smbpann     # train the XOR demo
make ut       # run the unit-test suite (also: make ut-asan, make ut-ubsan)
```

```text
XOR  topology 2-4-1  weights 12  rate 0.5  momentum 0.9  seed 1
learned:
  0 XOR 0  ->  0.0031  (target 0)
  0 XOR 1  ->  0.9970  (target 1)
  1 XOR 0  ->  0.9970  (target 1)
  1 XOR 1  ->  0.0036  (target 0)
```

## Roadmap

1. **Foundation**: flat-array FFNN, backprop with momentum, XOR. *(done)*
2. **Arena allocator**: the Mark/Release pool the population carves from. *(done)*
3. **Datasets**: plain-text train and test data. *(done)*
4. **Parallel evaluation**: a shell coordinator that launches one worker process
   per candidate, sized to the CPU count, exchanging fitness as plain text.
5. **The evolutionary search**: a population of topologies, mutation, selection on
   validation fitness, evaluated against a matched-compute random-search control.
6. **A reproducible benchmark**: run the search and its random-search control on a
   standard NAS benchmark (NAS-Bench-101 or 201), which provides pre-computed
   accuracies for a fixed search space, to answer the "does evolution beat random
   search?" question cheaply and reproducibly.

See [`AGENTS.md`](AGENTS.md) for the developer contract and module map.

## Literature

*Evolutionary Neural Architecture Search (SMBPANN's approach: evolve topology,
train weights by backprop):*
- E. Real, S. Moore, A. Selle, et al. *Large-Scale Evolution of Image
  Classifiers.* ICML 2017.
- E. Real, A. Aggarwal, Y. Huang, Q. V. Le. *Regularized Evolution for Image
  Classifier Architecture Search* (AmoebaNet). AAAI 2019.

*Neural Architecture Search and AutoML, broadly:*
- B. Zoph, Q. V. Le. *Neural Architecture Search with Reinforcement Learning.*
  ICLR 2017.
- H. Liu, K. Simonyan, Y. Yang. *DARTS: Differentiable Architecture Search.*
  ICLR 2019.
- T. Elsken, J. H. Metzen, F. Hutter. *Neural Architecture Search: A Survey.*
  JMLR, 2019.
- F. Hutter, L. Kotthoff, J. Vanschoren (eds.). *Automated Machine Learning:
  Methods, Systems, Challenges.* Springer, 2019 (open access).

*Classical neuroevolution (evolving topology and weights, the contrast case):*
- K. O. Stanley, R. Miikkulainen. *Evolving Neural Networks through Augmenting
  Topologies* (NEAT). Evolutionary Computation 10(2), 2002.
- K. O. Stanley, J. Clune, J. Lehman, R. Miikkulainen. *Designing neural networks
  through neuroevolution.* Nature Machine Intelligence, 2019.
- X. Yao. *Evolving Artificial Neural Networks.* Proceedings of the IEEE 87(9),
  1999.

*Baselines and reproducible benchmarks:*
- J. Bergstra, Y. Bengio. *Random Search for Hyper-Parameter Optimization.*
  JMLR, 2012.
- L. Li, A. Talwalkar. *Random Search and Reproducibility for Neural Architecture
  Search.* UAI 2019.
- C. Ying, A. Klein, E. Christiansen, E. Real, K. Murphy, F. Hutter.
  *NAS-Bench-101: Towards Reproducible Neural Architecture Search.* ICML 2019.
- X. Dong, Y. Yang. *NAS-Bench-201: Extending the Scope of Reproducible Neural
  Architecture Search.* ICLR 2020.

*Foundations:*
- D. E. Rumelhart, G. E. Hinton, R. J. Williams. *Learning representations by
  back-propagating errors.* Nature 323, 1986.
- V. Gavrilov. *Backpropagation Feed-Forward Neural Networks.* Seminar thesis,
  1997 (the mathematics the engine implements, cited by section in the source).

## License

GNU GPL v2 or later. Copyright (C) 2001 Vasili Gavrilov.
