# AGENTS.md -- how to develop SMBPANN (for humans and AI agents)

SMBPANN is a self-modifying backpropagation feed-forward neural network, written
in **Ada 2012** (the sources live at the repo root), carrying the coding
discipline of the **AIS** project but with the compiler enforcing it. Two earlier
versions are preserved as references: `legacy/c/` (the C99 foundation, a
bit-for-bit oracle) and `legacy/java/` (the original object-graph design). This is
the operating manual.

## Why Ada

The project's real requirements each land on an Ada strength: **provable memory
safety** (SPARK proves absence of overflow / out-of-bounds / use-after-free), a
**marker/arena allocator** (Ada storage pools, native), and a **CPU-sized
parallel population** (language-level tasking, provably race-free under
Ravenscar/SPARK). Performance equals C (GNAT is a GCC front-end). The AIS
ideology — one concept per unit, bounded/stack-first, no framework, allocation
only at construction — is expressed here where the compiler checks it.

## The contract (read first)

- **`smbpann-nets.ads`**, **`smbpann-nets-training.ads`** -- the public API: the
  network model and the backpropagation trainer, with `Pre`/`Post` contracts.
  The engine implements them; the demo/tests exercise them.
- **The 1997 thesis** (`~/articles/BPFNN_Coursework`) -- the mathematics the
  engine executes (forward pass, generalized delta rule, momentum, sigmoid).
  `smbpann-nets.adb` / `smbpann-nets-training.adb` cite its sections.
- **`legacy/c/`** -- the reference oracle. Any Ada change to numeric behaviour
  should still match the C output on XOR (currently identical to the printed digit).

## Build and test

    make            # build the XOR demo: gnatmake -gnatwa -gnata -gnat2012
    make run        # build and run it
    make clean

`-gnatwa` makes every warning fatal-by-attention (a warning is a defect, as in
AIS); `-gnata` turns the `Pre`/`Post` contracts into runtime checks. Full SPARK
proof (`gnatprove`) is a later, opt-in step and needs the SPARK toolset installed
(user-space via Alire). The C reference builds under `cd legacy/c && make ut`.

## Module map (repo root)

    smbpann.ads               Real (= Float) -- the one scalar type
    smbpann-rng.{ads,adb}     deterministic xorshift PRNG (modular, no UB)
    smbpann-act.{ads,adb}     activation functions (sigmoid + derivative)
    smbpann-nets.{ads,adb}    the network: flat weight buffers, forward pass;
                              a controlled type -- Finalize frees automatically
    smbpann-nets-training.*   backprop (generalized delta rule + momentum), a
                              child package so it sees the private Layer rep
    xor_demo.adb              the XOR demonstration (the current gate)

Memory: `new` appears only in the two `Create` functions (network, trainer);
Forward/Learn allocate nothing; controlled `Finalize` releases everything. That
is the AIS "allocate once, hot path allocates nothing" invariant, made structural.

## Roadmap

The Java README's goal is *hyper-parameterization*: the network discovers its own
topology. In the AIS "lock the contract, implement, test" loop:

1. **Foundation (done):** flat-array FFNN, backprop + momentum, XOR regression,
   matching the C oracle.
2. **`smbpann-arena.*`:** a marker/arena storage pool (Mark/Release), the native
   Ada form of the manual heap the population will carve from.
3. **Tasking demo:** a fixed pool of `Number_Of_CPUs` worker tasks pulling
   candidates from a `protected` queue.
4. **`smbpann-data.*`:** plain-text datasets (train/test split); a real problem
   beyond XOR.
5. **`smbpann-genome.*` + `smbpann-evolve.*`:** the genetic architecture search --
   a population of dense-flat topologies, mutation (grow/prune nodes and layers)
   and crossover, selection on validation fitness.

Cross-machine scale, if ever needed, stays a plain-text handoff layer *outside*
the in-node task pool (hybrid: tasks within a node, plain text across nodes).

## Working with AI agents

As in AIS: spawn focused subagents for isolatable work; a separate tester with
fresh context writing the tests catches the implementer's assumptions; the main
session locks the contract and runs the build. GNAT's strictness is the safety
net for Ada written by a model less fluent in Ada than in C -- mistakes are
compile-time rejections with locations, not silent runtime bugs.
