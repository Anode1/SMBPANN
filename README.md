# SMBPANN 2001-2015-2026

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

A population of networks is searched by mutating layer counts and widths, and by
co-evolving each candidate's own **training hyper-parameters** (learning rate,
momentum, and hidden activation function), selecting on held-out validation
performance; each candidate's weights are trained by backpropagation (the 1997
delta rule). Those hyper-parameters were arguments of the top signature and are
now discovered instead of specified, a concrete step toward the bottom one. (On
XOR the search reliably finds, on its own, that a ReLU hidden layer with high
momentum solves it near-exactly.)

Layers are also **structural**: each hidden layer can evolve between a dense one
and a weight-shared, locally-connected **convolution-like** kind. This was the
project's original ambition, stated in the 1997 thesis: to let the search
**rediscover LeCun's topology** on its own, the local receptive field and shared
weights that carry much of the work on images, rather than being handed it. The
thesis already drew the idea, a single unit fed by a small window of the previous
layer, the local feature detector:

![Local receptive field from the 1997 thesis: one unit in the next layer wired to a small window of the previous layer, the weight-sharing feature detector that convolution generalizes.](doc/feature_detector_field.png)

*Local receptive field, drawn in the 1997 thesis: one unit fed by a small window of
the previous layer. Shared across positions, this is a convolution.*

So the search can, in principle, discover this shift-invariant, weight-sharing
structure rather than being told it, the thesis's central point that the
architecture's inductive bias carries much of the work. The convolutional backprop
is verified by a numerical gradient check in the test suite.

The search returns the architecture that generalizes best on validation, with the
standard caveat that the search itself can overfit the validation split, so a
final untouched test set is needed to judge it.

Recombining two arbitrary topologies by crossover is a known hard problem (the
competing-conventions problem, where two networks encode the same function with
permuted units), so the default follows Real et al.'s mutation-only evolution.
Crossover is no longer left open; it is now studied directly (see the paper
below). On deceptive trap functions, where separable building blocks exist by
construction, crossover decisively beats mutation, and its advantage grows with the
number of blocks. On the real NAS-Bench-101 cell space, no crossover meaningfully
beats random search: operators that recombine the cell's wiring, labeling, node
roles, or whole paths are all consistently worse than a plain structure-blind one,
and none clears the benchmark's own training noise over random. Crossover helps only
where the space carries recombinable building blocks; the real cell space does not.

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
validation/       standalone probes: bbtest (trap control), nasxover (NAS-Bench-201),
                  nb101 + nb101_extract (NAS-Bench-101 crossover study)
paper/            the write-up (nas_crossover) and the companion essay
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

The unit-test suite (40 checks: rng, act, net, the XOR backprop regression,
1D and 2D convolution gradient checks, checkpoints, arena, data, genome) runs
clean under AddressSanitizer and UBSan.

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
seed). It compares a fixed mutation rate against the same rate left free to
**self-adapt** (the ES idea, Rechenberg and Schwefel: the rate lives in the genome,
perturbs log-normally at each birth, and selection keeps whatever value produced
good offspring). Ten seeds per cell:

| starting rate (`-M`) | fixed: GA wins / 10 (mean gap) | self-adaptive: GA wins / 10 (mean gap) |
|---|---|---|
| 1 | 5  (+0.0021) | 8  (+0.0026) |
| 3 | 5  (+0.0004) | 6  (+0.0017) |
| 6 | 5  (+0.0009) | 7  (+0.0023) |

Two honest reads. First, with a **fixed** rate the GA is about a coin-flip against
random (5 of 10) whatever the rate, with only a small positive mean gap: in a
search space this small it does not clearly beat an exhaustive random sampler,
reproducing the Bergstra and Bengio / Li and Talwalkar result. Second,
**self-adaptation helps, consistently and modestly**: it wins more often (6 to 8 of
10) and by a wider margin at every starting rate, and it removes the need to pick
the rate at all. You can watch the rate anneal within a run, higher early to
explore, lower later to refine (for example 1.00, then 2.28, then 1.35 on XOR).

The effect is real but small, and quieter than a first, noisier pass suggested (a
3-seed run had shown a fixed rate *losing* at M=1 and a mutation-rate "sweet spot";
ten seeds regress both to the mean). That is the value of more seeds.

Does a *bigger* search space favor the directed search, as the small-space result
hinted? At a fixed budget, the opposite. Enlarging the space to five layers by
twenty-eight wide (from three by sixteen), holding the roughly 100 evaluations
fixed, and re-drawing a fresh learnable task for each of 15 seeds, the GA does
**worse** than random: it wins 5 of 15 self-adaptive and 4 of 15 fixed, with
random better on average (gap about -0.003). The cause is budget, not the search:
the same roughly 100 evaluations now cover a far larger space, so the GA's local,
gradient-like convergence settles near its random start while an exhaustive random
sampler keeps covering ground. It is the exploration-versus-exploitation trade-off
in the open, and it confirms the intuition that broad random sampling wins once
the space outgrows the search's budget. Giving evolution a fair chance in a bigger
space means scaling the population and generations with it.

So that is the next experiment: the same bigger space, scaling the budget from 96
up through 384 to 768 evaluations, eight fresh-task seeds. The gap narrows toward
zero (from about -0.003 to about -0.0006), and at the largest budget the fixed
variant even edges positive (5 of 8, +0.0021), but the GA never decisively
overtakes random, and the 8-seed win counts are noisy and non-monotone. Two things
keep this honest: the win margin is small and the landscape nearly flat; and these
budget runs are *not* a controlled sweep, because to bound wall-clock the
per-candidate training was cut as the budget grew (epochs 1000 to 500), and fewer
epochs flatten the fitness signal on their own, so the narrowing cannot be credited
to budget alone. A controlled sweep (vary only the generations) is the clean next
step.

The honest overall tendency: random search is a strong baseline for architecture
search at this scale; self-adaptation beats a fixed mutation rate reliably by a
little (the one cleanly controlled comparison); and in a bigger search space the GA
does not reliably beat random at any budget we tried, hovering near parity.
Consistent with the field (Bergstra and Bengio 2012; Li and Talwalkar 2019), and
reproduced here from scratch. Each experiment is a one-command run
(`scripts/benchmark.sh`, with `ADAPT`, `LMAX`, `WMAX`, `POP`, `GENS`, `ELITE`,
`MUT`, `RUNS`, and the task size as environment variables).

## Error control

The comparison above fixes the budget and asks who has the lower error. But the
top-line signature `self_modifying_predict(TrainingSet, TestingSet, Error)` asks
the opposite: you hand it the error you want, and it searches UNTIL it gets there.
`evolve -E error` does exactly that, running until the best validation error
reaches the target (with `-G` a safety cap) and reporting which generation each
method arrived. The fitness the search minimizes IS that validation error, so a
single number is both the objective and the stop.

This turns the question from "whose error is lower at a fixed budget" into "who
reaches the target with less work", which is what a user of the interface actually
cares about. `scripts/errortest.sh` runs the time-to-target race over many seeds
and reports, for each method, how often and in how few generations it reaches the
target:

```sh
TARGET=0.12 RUNS=30 GENS=60 scripts/errortest.sh
```

## Paper

A short write-up in the style of the 1997 thesis, with the full method, exact
settings, and per-run statistics:
[`paper/nas_crossover.pdf`](paper/nas_crossover.pdf) (source
[`paper/nas_crossover.tex`](paper/nas_crossover.tex)). It pairs a trap-function
positive control (crossover wins decisively where separable building blocks exist)
with a real-space negative (on NAS-Bench-101 no crossover meaningfully beats random,
and the structure-aware operators are worse than a plain uniform one), and explains
why: good cells are common and the gap to the optimum is smaller than the
benchmark's training noise, so random search is hard to beat. The companion essay is
[`paper/essay.md`](paper/essay.md).

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
   downloads. *(done)*
7. **Error control** (`evolve -E` + `scripts/errortest.sh`): the search runs until
   it reaches a target validation error, realizing the top-line
   `self_modifying_predict(Train, Test, Error)` signature, and the time-to-target
   race replaces fixed-budget fitness as the measure. *(done)*
8. **Crossover on real NAS benchmarks** (`bbtest`, `nasxover`, `nb101`): a
   trap-function positive control, then a four-operator crossover study on
   NAS-Bench-201 and NAS-Bench-101. The accuracy tables are extracted from the
   official releases by a small pure-C decoder (`nb101_extract.c`), no PyTorch or
   TensorFlow; the NAS-Bench-101 fitness oracle is isomorphism-correct by
   brute-force canonicalization and self-tested. Finding: crossover wins where
   building blocks exist (traps) but not on the real cell space, where nothing
   meaningfully beats random. *(done)*

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
