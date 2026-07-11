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
.                 the C99 engine: rng, act, net, train, data, arena, genome
                  smbpann (worker) + evolve (search); scripts/evaluate.sh (coordinator)
legacy/java/      the original early-2000s Java prototype (object-graph design)
```

An Ada 2012 implementation lived here too and is preserved in the git history; C
is the active language.

## Status

The engine and the search both run end to end:

- `smbpann` trains a network of a given topology (built-in XOR, or a plain-text
  dataset) and prints a machine-readable `fitness` line;
- `scripts/evaluate.sh` evaluates a whole population in parallel, one worker
  process per candidate, and ranks it;
- `evolve` runs the evolutionary architecture search against a matched-compute
  random-search control.

The unit-test suite (29 checks: rng, act, net, the XOR backprop regression,
arena, data, genome) runs clean under AddressSanitizer and UBSan.

```sh
make                            # build ./smbpann and ./evolve
./smbpann                       # train the built-in XOR demo
./evolve -i 2 -o 1 -P 8 -G 8    # evolve XOR topologies, GA vs random search
make ut                         # run the unit-test suite
```

```text
evolving 8 topologies over 8 generations (elite 2, seed 1)
gen   1   GA best=4.19705e-06 (2,8,11,15,1)   RAND best=4.78733e-06 (2,4,4,1)
...
final:  GA 2.62043e-06 (2,14,15,1)  vs  RAND 3.20376e-06 (2,7,14,1)   [64 evaluations each]
```

XOR is only a smoke test (its space is tiny and every topology solves it). For a
real test, `scripts/benchmark.sh` runs the same race on a self-contained synthetic
task where topology actually matters (generated by `gentask`, reproducible from a
seed), over several seeds:

```text
task: dim=4 samples=600 freq=3 noise=5%   |   5 runs, pop=10 gens=10 epochs=1500
run 1 (seed 1):  GA=0.0747   RAND=0.0698   RAND wins/ties
run 3 (seed 3):  GA=0.0697   RAND=0.0786   GA wins
...
GA beat random search in 1 of 5 runs; mean (RAND - GA) test-MSE gap = -0.0054
```

**The honest first finding:** at this setting the GA does *not* reliably beat its
matched random-search control (random was slightly better on average). That is
consistent, in miniature, with the well-known result that random search is a
strong baseline in small architecture spaces (Bergstra and Bengio 2012; Li and
Talwalkar 2019). But it is a *first* result, not a verdict: whether evolution pays
off is sensitive to the GA's own hyper-parameters, the mutation rate chief among
them (tunable via `evolve -M`), and to the size and structure of the search space.
This run used one mutation setting, and the mutation rate matters a great deal. A
quick sweep over it (3 seeds each, same task) swings the outcome:

| mutation moves per offspring (`-M`) | GA wins, of 3 | mean (RAND - GA) gap |
|---|---|---|
| 1 | 1 | -0.0005  (random better) |
| 3 | 2 | +0.0004  (GA better) |
| 6 | 1 | +0.0006 |

Too little mutation stalls near the random start, a moderate rate wins, too much
drifts back toward a random walk. So even at this tiny scale a moderate mutation
rate flips the result from a loss to a win. A full sweep (more seeds, a wider grid,
a larger search space) is the real experiment; the point is that the harness makes
it a one-command, reproducible question.

## Roadmap

1. **Foundation**: flat-array FFNN, backprop with momentum, XOR. *(done)*
2. **Arena allocator**: the Mark/Release pool the population carves from. *(done)*
3. **Datasets**: plain-text train and test data. *(done)*
4. **Parallel evaluation**: a shell coordinator (`scripts/evaluate.sh`) that
   launches one worker process per candidate, sized to the CPU count, exchanging
   fitness as plain text and ranking the results. *(done)*
5. **The evolutionary search** (`evolve`): a population of topologies, mutation,
   selection on validation fitness, raced against a matched-compute random-search
   control. *(done)*
6. **A reproducible benchmark** (`gentask` + `scripts/benchmark.sh`): the GA and
   its matched random-search control on a self-contained synthetic task where
   topology matters, over several seeds, reproducible from a seed with no
   downloads. *(done)* Bridging to the real NAS-Bench-101/201 (a cell-based search
   space shipped as multi-gigabyte PyTorch files) is future work.

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
