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
Ada.

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

## Language: Ada

SMBPANN is written in **Ada 2012**. The numeric core is written to be amenable to
SPARK proof (a planned step); the memory-management layer is deliberately outside
the SPARK subset, as noted below. The choice follows the work:

- **Proof, not just tests, for the numeric core.** SPARK with GNATprove can
  *prove* the absence of run-time errors (integer overflow, out-of-bounds
  indexing, division by zero) in the flat-array math, rather than testing for
  them.
- **A native arena allocator.** The manual heap with markers the population wants
  (Mark and Release) is a first-class Ada storage pool. This memory layer is
  intentionally outside SPARK: custom storage pools and post-Release dangling
  pointers are exactly what SPARK's ownership model excludes, so it is guaranteed
  by construction and by test, not by proof.
- **A parallel population, in the language.** Evaluating many candidates at once
  is Ada tasking: a fixed pool of workers drawing from a `protected` work queue.
  Because several workers wait on one queue, this fits the Jorvik profile (Ada
  2022), under which SPARK can prove freedom from data races and deadlocks.
  (Planned; see roadmap.)
- **A shared GCC back end.** GNAT is a GCC front-end, so with run-time checks
  suppressed the numeric loops optimize to machine code comparable to C, with no
  interpreter or virtual-machine overhead. A C reference implementation
  (`legacy/c/`) cross-checks the Ada's output, matching to the last printed digit.

### Why Ada, and not Rust?

Rust is the other modern memory-safe systems language, and a good one; this is
not a claim that Ada is safer, since both prevent the classic memory errors. The
reasons are specific:

- **Familiarity and revival.** The author programmed in Ada (83 and 95) in
  1995-96, including multi-agent interaction programs, and wants to bring the
  language back for what it is good at: non-mobile, concurrent, well-parallelizable
  applications on the desktop. A known tool shortens the path from idea to working
  code.
- **Where the two draw the safety line.** Rust's borrow checker delivers memory
  safety and data-race freedom automatically, with no annotations, for all safe
  code, but it does not prove absence of arithmetic overflow, panics, or
  functional correctness. SPARK with GNATprove reaches further in *what* it can
  establish, absence of run-time errors (overflow, out-of-bounds, division by
  zero) and functional correctness against contracts, but that reach is not free:
  it takes explicit contracts and proof effort, and applies only to code in the
  SPARK subset. For a numeric engine whose correctness we want to state and check,
  that broader, contract-based reach is the draw, not a claim that Rust is less
  safe. (The Mark/Release arena here is outside the SPARK subset; Rust's borrow
  checker would likewise reject it in safe code and push it into `unsafe`. On
  manual memory the two are symmetric.)
- **Model, as a preference.** The friction people report with Rust is less its
  surface syntax than its ownership, lifetime, and trait machinery, whose learning
  curve is well documented; the author finds Ada's explicit, readable style an
  easier fit. This is a preference, not a metric: Ada has its own verbosity, and
  Rust users are productive once past the initial climb.
- **Concurrency native to the language.** Ada has had `task` and `protected`
  objects since Ada 83, with parallel loops added in Ada 2022, a mature model well
  suited to a CPU-bound, parallelizable desktop search, and one SPARK can prove
  free of data races and deadlocks under the Jorvik profile. Rust matches the
  automatic side of this: its ownership model gives compile-time data-race freedom
  for free (`Send`/`Sync`), where Ada relies on you actually using `protected`
  objects. The deciding factor here is fit, not a claim that Rust's threads are
  less capable.

Rust's larger ecosystem and zero-annotation guarantees are real advantages; the
choice here is about fit, and about reviving a language the author knows and
values.

## Design principles

Inherited from the [AIS](https://github.com/Anode1/ais) project, many of them now
enforced by the compiler (strong typing, `Pre` and `Post` contracts) rather than
by discipline:

- one concept per compilation unit; clear, literal names;
- allocation only at construction, kept by convention (`new` appears only in the
  `Create` functions), so the train and infer hot path allocates nothing;
- no framework, no dependency; a plain build with GNAT;
- reproducible by construction: a seeded PRNG and no wall-clock in the engine (a
  property the future parallel search will have to preserve deliberately);
- regression tests from the smallest cases up (XOR is the first gate).

## Layout

```
.                 the active engine (Ada 2012): rng, act, nets, training, data, arena
legacy/c/         a C99 foundation, kept as a reference oracle
legacy/java/      the original early-2000s Java prototype (object-graph design)
```

## Status

**The genetic search, the project's actual subject, is not yet implemented.**
What runs today is the backpropagation engine each candidate will use: a
flat-array feed-forward network, backprop with momentum, plain-text dataset
loading with a train/test split, and the arena allocator, all under a unit-test
suite. The canonical gate is the **XOR** regression, the linearly-non-separable
problem a single perceptron cannot learn, and the smallest proof that backprop
through a hidden layer works.

```sh
make run     # train the XOR demo
make test    # run the unit-test suite
```

```text
XOR  topology 2-4-1  weights 12  rate 0.5  momentum 0.9  seed 1
learned:
   0 XOR 0  ->  3.13E-03   (target 0)
   0 XOR 1  ->  9.97E-01   (target 1)
   1 XOR 0  ->  9.97E-01   (target 1)
   1 XOR 1  ->  3.56E-03   (target 0)
```

## Roadmap

1. **Foundation**: flat-array FFNN, backprop with momentum, XOR. *(done)*
2. **Arena storage pool**: the Mark and Release allocator the population carves
   from. *(done)*
3. **Datasets**: plain-text train and test data. *(done)*
4. **Task pool**: a fixed pool of workers sized to the CPU count, over a
   `protected` queue.
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
