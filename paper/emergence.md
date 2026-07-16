# Directed emergence: growing a network's topology under an energy budget

*Research note. Positive-and-honest counterpart to the NAS-Bench-101 crossover study
(`paper/nas_crossover.tex`). Reproduce: `make conv_emerge && ./conv_emerge`
(env: `SEEDS`, `GENS`, `TARGET`, `PADD`). All numbers below: 24 seeds × 150 generations.*

## The idea

The original SMBPANN ambition was that a network's **architecture should emerge** from an
initial condition under selection, not be designed by hand. The way to make it visible — as it
was on a 486, where compute was the constraint — is to put a **cost on resources**: start from a
fully-connected (dense) net and select for the structure that reaches a target accuracy with the
**least energy** (fewest connections). Removing a connection saves energy; removing a *useful* one
breaks the accuracy floor and is punished. So the search settles on a minimal structure that
still solves the task.

Convolution was the motivating test case: for a task whose information is local, is the
minimal-energy structure the **local receptive field**? The answer below is partial and, I think,
more interesting than a clean yes. Energy-constrained emergence reliably discovers **which
connections matter** (sparse, task-relevant connectivity — cleanly, for a sparse task), and with a
weight-tying gene it even discovers **weight sharing** (turning on a convolution when
translation-invariance rewards it, improving generalization). The annealing and energy/accuracy
mechanics behave exactly as the physics predicts. What still does *not* fall out is the last,
specific piece — a **compact aligned local filter**: pruning gives sparsity but not aligned windows,
and the tying gene gives sharing but not a tight kernel. Convolution's parts emerge separately; the
whole, compact convolution needs a pressure the plain energy term does not supply — matching the
pruning literature and this project's other finding that structural cleverness does not come for free.

## The mechanism, and where its pieces come from

This sits at an intersection of several fields, which is where the idea gets its force:

- **Emergence / complexity** (Conway's Life, Wolfram's automata, fractals, chaos): structure
  from simple rules. Here it is *directed* — a fitness steers emergence toward a solution.
- **Statistical mechanics → optimization**: the search is **simulated annealing** on
  connectivity (Kirkpatrick, Gelatt & Vecchi 1983, from the Metropolis algorithm; classic in
  VLSI placement). Pruning is a downhill move in energy; a *rare* added connection is an uphill
  thermal fluctuation that escapes local minima. Rare growth = low temperature = the system
  settles into a stable near-minimal structure — confirmed by the temperature sweep below.
- **Evolution**: a GA with selection and elitism; the no-selection control is literally
  **genetic drift**.
- **The locality principle**: information is often local (physics, biology). A method minimizing
  energy on a local task *should* avoid non-local connections — but "avoiding non-local" turns out
  to be weaker than "forming aligned local windows" (see results).
- **Network pruning**: "start dense, prune to the minimal structure that holds accuracy" is the
  pruning tradition, whose origin is **LeCun's own Optimal Brain Damage (1990)**. The pruning
  literature finds sparse *subnetworks*, not convolutions — which is exactly what we see.

## Setup

A one-hidden-layer net (H = N−K+1 = 10 units over N = 12 inputs) whose connectivity is a binary
mask, evolved by a GA from an all-ones (dense) seed. Fitness = validation accuracy on a **scarce**
train set (64 examples), objective = *minimize density subject to accuracy ≥ target*. Mutation is
**mostly-prune, rarely-grow** (the annealing schedule; prune 0.030, grow 0.006). Metrics:
**density** (energy), **RF-span** (mean receptive-field width / N; low = local *or* just few
connections), **on-relevant** (fraction of surviving connections on task-relevant inputs), vs a
drift (no-selection) control and a chance baseline.

## Results

### 1. What emerges matches the task — cleanly for feature selection, not for convolution

| task | density | RF-span | on-relevant (chance / drift) | test |
|---|---|---|---|---|
| LOCAL (contiguous windows) | 0.100 | 0.414 | 0.259 (0.25 / 0.272) | 0.876 |
| GLOBAL (linear, all inputs) | 0.087 | 0.403 | 1.000 (1.00 / 1.000) | 0.892 |
| SPARSE (few specific inputs) | **0.028** | 0.227 | **0.900** (0.33 / 0.342) | 0.920 |

**SPARSE is the clean positive:** the search prunes to ~3 connections and lands **90%** of them on
the informative inputs — far above chance (0.33) and drift (0.342). It genuinely *discovers which
inputs matter*. **LOCAL is the honest negative:** it also prunes hard, but its surviving connections
are **not** preferentially in-window (on-relevant 0.259 ≈ chance, and *below* drift), and its span
(0.414) is no lower than GLOBAL's (0.403). At ~0.1 density the small span is just "few connections,"
not the aligned local receptive fields convolution needs. Many sparse subnetworks solve a local task,
and selection finds *a* sparse one, not *the* convolutional one.

**Emergence in action.** On the SPARSE task the selection is visible generation by generation, as the
population anneals from the dense seed down to ~3 connections:

| gen | on-relevant | density |
|---|---|---|
| 0 | 0.333 (chance) | 1.000 |
| 25 | 0.379 | 0.246 |
| 50 | 0.700 | 0.044 |
| 100 | 0.904 | 0.028 |
| 150 | 0.905 | 0.028 |

As energy falls (density 1.0 → 0.028) the surviving connections migrate onto the informative inputs
(on-relevant 0.33 → 0.90). The structure is discovered *as the system cools*.

### 2. Annealing temperature controls how far it converges

| grow rate | RF-span | density | test |
|---|---|---|---|
| 0.000 | 0.426 | 0.086 | 0.860 |
| 0.006 | 0.414 | 0.100 | 0.876 |
| 0.030 | 0.422 | 0.189 | 0.881 |
| 0.100 | 0.819 | 0.506 | 0.902 |

Rarer growth (a cooler anneal) converges to a much sparser structure (density 0.086 vs 0.506); a hot
system stays broad. Simulated-annealing temperature made literal.

### 3. Energy–accuracy Pareto

| target acc | density | RF-span | feasible | test |
|---|---|---|---|---|
| 0.80 | 0.049 | 0.303 | 63% | 0.772 |
| 0.85 | 0.069 | 0.304 | 56% | 0.827 |
| 0.90 | 0.100 | 0.414 | 39% | 0.876 |
| 0.95 | 0.247 | 0.536 | 10% | 0.907 |

The energy a structure needs scales monotonically with the accuracy demanded — the method traces the
accuracy/energy Pareto front, from ~5% density at a modest target to ~25% at a stringent one.

### 4. Reuse

A structure emerged on one task (density 0.250) transfers to a **fresh task from the same family** at
test 0.889, versus a dense net's 0.909 — nearly matching at **25% of the energy**. Discover the
topology once, reuse it cheaply.

### 5. A weight-tying gene: sharing *does* emerge, but not the compact filter

`emerge_tie.c` adds convolution's other half as a gene: a `shared` bit that ties weights by offset
(one weight per relative position, reused everywhere — a convolution), with energy now counted as
free **parameters** (a shared filter of width K costs K; an unshared local receptive field costs
K·H = 30). From a dense *unshared* seed, sharing forced off vs allowed (16 seeds × 150 gens):

| arm | shared-frac | param-energy | RF-span | test |
|---|---|---|---|---|
| sharing forced off | 0.000 | 0.098 | 0.373 | 0.859 |
| sharing allowed | **0.893** | 0.112 | 0.611 | 0.900 |

**Weight sharing emerges** — 89% of the population turns it on — and it *improves generalization*
(0.900 vs 0.859): the search discovers that tying weights is the efficient way to exploit
translation-invariance. That is the half of convolution connectivity pruning alone could not produce,
and it is a cleaner positive than expected. **But** the emerged shared filter is **broad** (span 0.61,
~13 taps), not the compact K = 3 window: the parameter-energy pressure on filter *width* is too weak
to trim it, so a weight-shared *wide* filter emerges, not a tight convolution.

## Honest bottom line

Directed emergence under an energy budget **works as a sparsifier, a feature selector, and — with a
tying gene — a discoverer of weight sharing**: it finds sparse task-relevant connectivity, cleanly
picks the informative inputs on a sparse task, turns on weight sharing when translation-invariance
rewards it (improving generalization), obeys an annealing temperature, traces the accuracy/energy
Pareto, and its structures reuse across a task family. What still does **not** fall out is the
**compact aligned local receptive field**: connectivity pruning finds sparsity but not aligned
windows, and the tying gene finds sharing but not a tight filter. Convolution's two halves emerge
*separately and partially*; the *compact* convolution is not assembled by an energy budget alone.

This is consistent with the pruning literature (sparse subnetworks, not convolutions) and with the
crossover study in this repo: the general, useful thing here is **automatic structural discovery under
a resource budget** (sparsity, feature relevance, weight sharing); the last, specific piece — a tight
local filter — needs a pressure the current energy term does not supply.

## Next steps

- **A filter-width pressure**: penalize the *span* of a shared filter (not just its parameter count),
  to test whether the compact local window — the last missing piece — can then be selected for.
- **Larger N and 2-D inputs**, where greedy enumeration fails and the GA earns its keep; then sound
  and higher-dimensional tasks, to discover and reuse their topologies.
