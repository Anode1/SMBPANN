# SMBPANN

**Self-Modifying Backpropagation Feed-Forward Neural Network** — a network that
is not *told* its architecture but *discovers* it, evolving a population of
candidate topologies against a fitness function.

It began as a 1997 seminar thesis deriving the backpropagation delta rule in
tensor form, was prototyped in Java in the early 2000s, and was shelved when
TensorFlow arrived. What was an unusual idea then — *let the network design
itself* — is now an established field: **Neural Architecture Search (NAS)** and
**AutoML**, of which the evolutionary approach taken here is the **neuroevolution**
branch. This is that idea, rebuilt from first principles.

## The goal

Not a predictor whose architecture you hand-specify:

```text
predict(TrainingSet, TestingSet, Topology, Activation, Jitter, ...)
```

but one handed only the problem, which finds the rest itself:

```text
self_modifying_predict(TrainingSet, TestingSet, Error)
```

A genetic search runs a population of networks — mutating layer counts and
widths, crossing over, selecting on validation fitness — and returns the
architecture that generalizes best. The backpropagation delta rule (the 1997
thesis) trains each candidate; the novelty is the *search around it*.

## Where this sits today

The early-2000s framing anticipated, without the vocabulary, what the field now
calls:

- **Neural Architecture Search (NAS)** — automating the design of network
  topology, kicked off at scale by Zoph & Le (2017), surveyed by Elsken et al.
  (2019).
- **AutoML** — automating the whole model-construction pipeline (Hutter et al.,
  2019).
- **Neuroevolution** — evolving networks (topology *and* weights) with
  evolutionary algorithms. NEAT (Stanley & Miikkulainen, 2002) is the canonical
  method and is essentially SMBPANN's goal stated precisely; the modern
  evolutionary-NAS line (Real et al., 2017 / 2019) scaled it to image classifiers.

SMBPANN is deliberately the **evolutionary** branch — a genetic search over dense
feed-forward topologies — rather than the reinforcement-learning (Zoph & Le) or
differentiable (DARTS) branches.

**An honest open question drives the project.** Whether an evolved architecture
search reliably beats cheap baselines — random search in particular — is not
settled; Li & Talwalkar (2019) showed random search is a surprisingly strong NAS
baseline. Testing whether this genetic search earns its keep against that
baseline is exactly what SMBPANN is for.

## Language: Ada

SMBPANN is written in **Ada 2012** (SPARK-ready). The choice follows the work,
not fashion — each real requirement lands on an Ada strength:

- **Safety by proof, not by lint.** The search does manual, arena-style memory
  management over a churning population; SPARK can *prove* the absence of
  overflow, out-of-bounds, and use-after-free rather than testing for them. Ada
  can even forbid post-initialization allocation at compile time.
- **A native arena allocator.** The "manual heap with markers" this design wants
  is a first-class Ada storage pool, not something hand-rolled in unchecked code.
- **A parallel population, in the language.** Evaluating many candidates at once
  is Ada tasking — a fixed pool sized to the CPU count, drawing work from a
  `protected` queue, provably race-free under the Ravenscar/SPARK subset.
- **No speed penalty.** GNAT is a GCC front-end; the numeric loops compile to the
  same code C would produce. A C reference implementation (`legacy/c/`) confirms
  this and cross-checks the Ada bit-for-bit.

## Design principles

Inherited from the [AIS](https://github.com/Anode1/ais) project, but here
enforced by the compiler rather than by discipline:

- one concept per compilation unit; clear, literal names;
- allocation only at construction — the train/infer hot path allocates nothing;
- no framework, no dependency; a plain build with GNAT;
- reproducible by construction — a seeded PRNG, no wall-clock in the engine;
- regression tests from the smallest cases up (XOR is the first gate).

## Layout

```
.                 the active engine (Ada 2012): rng · act · nets · training · demo
legacy/c/         a C99 foundation, kept as a bit-for-bit reference oracle
legacy/java/      the original early-2000s Java prototype (object-graph design)
```

## Status

Foundation complete: flat-array feed-forward network, backpropagation with
momentum, and the canonical **XOR** regression — the linearly-non-separable
problem a single perceptron cannot learn, and the smallest proof that backprop
through a hidden layer works.

```sh
make run
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

1. **Foundation** — flat-array FFNN, backprop + momentum, XOR. *(done)*
2. **Arena storage pool** — the marker / Mark-Release allocator the population carves from.
3. **Task pool** — a fixed pool of workers sized to the CPU count, over a `protected` queue.
4. **Datasets** — plain-text train/test data; a real problem beyond XOR.
5. **The genetic search** — a population of topologies, mutation and crossover,
   selection on fitness. The part worth testing.

See [`AGENTS.md`](AGENTS.md) for the developer contract and module map.

## Literature

The field this project belongs to, and the classics under it.

*Neuroevolution — evolving topology and weights (SMBPANN's own approach):*
- K. O. Stanley, R. Miikkulainen. *Evolving Neural Networks through Augmenting
  Topologies* (NEAT). Evolutionary Computation 10(2), 2002.
- K. O. Stanley, J. Clune, J. Lehman, R. Miikkulainen. *Designing neural networks
  through neuroevolution.* Nature Machine Intelligence, 2019.

*Evolutionary Neural Architecture Search — the GA over architectures, at scale:*
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

*Hyper-parameter optimization, and the baseline that keeps the field honest:*
- J. Bergstra, Y. Bengio. *Random Search for Hyper-Parameter Optimization.*
  JMLR, 2012.
- L. Li, A. Talwalkar. *Random Search and Reproducibility for Neural Architecture
  Search.* UAI 2019.

*Foundations:*
- D. E. Rumelhart, G. E. Hinton, R. J. Williams. *Learning representations by
  back-propagating errors.* Nature 323, 1986.
- V. Gavrilov. *Backpropagation Feed-Forward Neural Networks.* Seminar thesis,
  1997 — the mathematics the engine implements, cited by section in the source.

## License

GNU GPL v2 or later. Copyright (C) 2001 Vasili Gavrilov.
