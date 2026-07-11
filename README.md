# SMBPANN

**Self-Modifying Backpropagation Feed-Forward Neural Network.** Instead of being
told its architecture, the network **discovers its own topology** — how many
layers, how wide, which parameters — by evolving a population of candidate
networks against a fitness function. This is the early-2000s Java project's goal,
"hyper-parameterization of the ANN," rebuilt properly.

The mathematics is the author's 1997 seminar thesis, *Backpropagation
Feed-Forward Neural Networks* — the generalized delta rule derived in tensor
form. The engine's comments cite it section by section.

## The goal

Not this, with the architecture hand-specified:

```text
predict(TrainingSet, TestingSet, Topology, Activation, ...)
```

but this, with the architecture discovered:

```text
self_modifying_predict(TrainingSet, TestingSet, Error)
```

A genetic architecture search runs a population of candidate networks — mutating
layer counts and widths, crossing over, selecting on validation fitness — and
returns the topology that generalizes best. **Whether that search actually finds
good architectures is the interesting thing to test**, and the reason the project
exists.

## Language: Ada

SMBPANN is written in **Ada 2012** (SPARK-ready). The choice follows the work,
not fashion — every real requirement lands on an Ada strength:

- **Safety by proof, not by lint.** The engine does manual, arena-style memory
  management for the population; SPARK can *prove* the absence of overflow,
  out-of-bounds, and use-after-free rather than testing for them. Ada can even
  forbid post-init allocation at compile time.
- **A native arena allocator.** The "manual heap with markers" this design wants
  is a first-class Ada storage pool, not a thing hand-rolled in unchecked code.
- **A parallel population, in the language.** Evaluating many candidates at once
  is Ada tasking — a fixed pool sized to the CPU count, pulling work from a
  `protected` queue, provably race-free under the Ravenscar/SPARK subset.
- **No speed penalty.** GNAT is a GCC front-end; the numeric loops compile to the
  same code C would.

It carries the coding discipline of the [AIS](https://github.com/Anode1/ais)
project — one concept per unit, allocation only at construction, no framework —
except here the compiler enforces it (contracts, controlled types) rather than a
human remembering to.

## Layout

```
.                 the active engine (Ada 2012): rng · act · nets · training · demo
legacy/c/         the C99 foundation — kept as a bit-for-bit reference oracle
legacy/java/      the original early-2000s Java project (object-graph design)
```

The C foundation preceded the Ada and agrees with it to the printed digit on the
XOR regression, so it stays as a cross-check.

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
2. **Arena storage pool** — the marker/Mark-Release allocator the population carves from.
3. **Task pool** — a fixed pool of workers sized to the CPU count, over a `protected` queue.
4. **Datasets** — plain-text train/test data; a real problem beyond XOR.
5. **The genetic search** — a population of topologies, mutation and crossover,
   selection on fitness. The part worth testing.

See [`AGENTS.md`](AGENTS.md) for the developer contract and module map.

## License

GNU GPL v2 or later. Copyright (C) 2001 Vasili Gavrilov.
