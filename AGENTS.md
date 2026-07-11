# AGENTS.md -- how to develop SMBPANN (for humans and AI agents)

SMBPANN is a self-modifying backpropagation feed-forward neural network
(evolutionary Neural Architecture Search), written in **C99** in the coding
discipline of the **AIS** project. The original early-2000s Java prototype is
preserved in `legacy/java/`; an Ada 2012 rewrite was explored and lives in the
git history. This is the operating manual.

## The contract (read first)

- **`doc/dev/STYLE.md`** -- coding ideology: K&R/Robbins C99, one concept per
  `.c`/`.h`, stack-first, allocation only at construction, bounded strings,
  return codes, single-exit `goto` cleanup, sanitizer-gated. Non-negotiable.
- **`net.h`, `train.h`, `arena.h`, `data.h`** -- the public API. The engine
  implements them; `tests.c` tests them.
- **The 1997 thesis** (`~/articles/BPFNN_Coursework`) -- the mathematics the
  engine executes (forward pass, generalized delta rule, momentum, sigmoid);
  `net.c` / `train.c` cite it by section.

## Build and test

    make            # build ./smbpann
    make ut         # engine unit tests (tests.c, in-process) -- the commit gate
    make ut-asan    # tests under AddressSanitizer
    make ut-ubsan   # tests under UBSan
    make pedantic   # -std=c99 -pedantic + extra warnings; must be clean
    make clean

`make ut` is the commit gate; run `make ut-asan` and `make ut-ubsan` before
committing. A warning is a defect. Run the demo: `./smbpann -H 4 -e 20000`
(train a 2-4-1 net on XOR); `./smbpann -h` for options.

## Module map

    common.h   smb_real (=float), SMB_MAX_LAYERS, SMB_LINE_MAX
    rng        deterministic xorshift PRNG (reproducible init + evolution)
    act        activation functions (sigmoid + derivative), pure
    net        the network: flat weight matrices, forward pass, thesis init
    train      backpropagation: the generalized delta rule with momentum
    arena      marker/Mark-Release allocator (the population's heap)
    data       plain-text datasets + train/test split
    main.c     the CLI (getopt); the only layer that prints or exits
    tests.c    -DUNIT_TEST unit suite: rng act net xor arena data

Memory: allocation lives only in the `*_new` / `*_init` / load paths; the
train/infer hot path allocates nothing; every path frees on exit (`goto`
single-exit). Peak footprint is computable by hand from the struct sizes plus the
topology.

## Concurrency model

The evolutionary search is embarrassingly parallel: candidates are independent.
Each is evaluated as its own **worker process**, launched by a shell coordinator
sized to the CPU count, exchanging plain text with the parent. No shared memory,
so no data races and no synchronization code; a crashing candidate takes down only
its worker. Datasets stream one example at a time (bounded footprint; a read-only
file is shared across workers by the OS page cache).

## Roadmap

1. Foundation: flat-array FFNN, backprop + momentum, XOR. *(done)*
2. Arena allocator (Mark/Release). *(done)*
3. Datasets: plain-text train/test split. *(done)*
4. Parallel evaluation: a shell coordinator, one worker process per candidate.
5. The evolutionary search: population, mutation, selection on validation fitness,
   against a matched-compute random-search control.
6. A reproducible benchmark: the search vs its random-search control on a standard
   NAS benchmark (NAS-Bench-101 / 201).

## Working with AI agents

As in AIS: spawn focused subagents for isolatable work; a separate tester with
fresh context writing the tests catches the implementer's assumptions; the main
session locks the contract and runs `make ut` plus the sanitizers.
