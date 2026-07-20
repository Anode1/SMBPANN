# SMBPANN: Coding Style and Ideology

SMBPANN is written in C99 in the idiom of the **AIS** project. AIS's
`doc/dev/STYLE.md` is the parent contract; read it first. This file records only
where SMBPANN follows it and the **one place it must deviate**, with the reason.

## Inherited from AIS (non-negotiable)

- **C99**, K&R/Robbins lineage: clear, conventional, no cleverness. Build clean
  under `-std=c99 -W -Wall` and `make pedantic`. **A warning is a defect.**
- **One concept per `.c/.h`.** `net` (the model), `train` (backprop), `act`
  (activations), `rng` (determinism) are each isolatable frames of knowledge.
  Names are literal.
- **Return codes, not exits, in the modules.** Only the CLI (`main.c`) prints or
  exits; the engine returns `0`/`-1` or a value/`NULL` and lets the caller decide.
- **Bounded strings only** (`snprintf`, never `strcpy`/`sprintf`); `static` for
  everything module-private; `const`-correct; `size_t` for sizes; declare at
  point of use.
- **Single-exit cleanup** (`goto fail`) wherever a function holds a resource, so
  no path leaks. `net_new`/`trainer_new` free every partial allocation on failure.
- **Determinism.** No wall-clock in the engine: the same seed gives the same run
  on every machine (`rng.c`). Reproducibility is a requirement, not a nicety.
- **No framework, no dependency.** A plain `Makefile` and `cc`; sources are
  auto-globbed. Sanitizers (`make ut-asan`, `make ut-ubsan`) are the memory-safety
  gate, exactly as in AIS.

## The one sanctioned deviation: construction-time allocation

AIS's core allocates nothing on the record path -- peak footprint is a function
of struct sizes, never of the data. **SMBPANN cannot honour that literally**, and
pretending otherwise would misrepresent it: a neural network's **weights are its
data**. Their count *is* the topology; they must live in RAM to be trained.

So SMBPANN keeps the *guarantee* AIS's rule exists to provide -- no allocation on
the hot path, no leak/use-after-free class -- by moving all allocation to
**construction time** (NASA Power of Ten rule 3: *no dynamic allocation after
initialization*; MISRA C:2012 21.3 is relaxed only here, deliberately):

- `net_new` and `trainer_new` allocate **every** buffer **once**, up front.
- `net_forward` and `trainer_learn` -- the train/infer hot path -- allocate
  **nothing**. This is enforceable by reading: no `malloc` outside the two `*_new`
  functions.
- `net_free` and `trainer_free` release on every path; both are NULL-safe and
  safe on a partially-constructed object.
- The `Net`/`Trainer` **structs themselves stay fixed-size**: their pointer
  tables are bounded by `SMB_MAX_LAYERS` (network *depth*), so only the weight
  *data* (network *width x width*) is heap. Peak footprint is still computable by
  hand: `sum_l dim[l]*dim[l-1]` weights, times `sizeof(smb_real)`.

At GA scale this is what lets a worker process hold a multi-gigabyte network with
zero allocator traffic during training, and lets the coordinator size a 64 GB
population by counting weights. See `doc/dev/WHY-FLAT.md` (planned) for why flat
matrices, not the Java object graph, are mandatory at that scale.

## Numeric type

`smb_real` is `float` (`common.h`). Half the memory of `double` -- doubling the
population that fits a RAM budget -- with ample precision for backprop. One
typedef flips the engine to `double`.
