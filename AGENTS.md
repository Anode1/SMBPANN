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
committing. A warning is a defect.

The worker: `./smbpann -t 2,4,1 -q` trains a topology and prints a machine-readable
`RESULT ... fitness=<x>` line (lower is better; a dataset via `-f file -i in -o out`,
else built-in XOR); `./smbpann -h` for options. The coordinator fans a population
of candidates (one topology per line) across `nproc` worker processes:

    printf '2,2,1\n2,4,1\n2,8,1\n' | scripts/evaluate.sh          # leaderboard
    COMMON="-f data.txt -i 2 -o 1" scripts/evaluate.sh pop.txt

## Module map

    common.h   smb_real (=float), SMB_MAX_LAYERS, SMB_LINE_MAX
    rng        deterministic xorshift PRNG (reproducible init + evolution)
    act        activation functions (sigmoid + derivative), pure
    net        the network: flat weight matrices, forward pass, thesis init
    train      backpropagation: the generalized delta rule with momentum
    arena      marker/Mark-Release allocator (the population's heap)
    data       plain-text datasets + train/test split
    main.c     the CLI / evaluation worker (getopt); emits a RESULT fitness line
    tests.c    -DUNIT_TEST unit suite: rng act net xor arena data
    scripts/evaluate.sh   the parallel coordinator (one worker process/candidate)

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
4. Parallel evaluation: `scripts/evaluate.sh` fans candidates to `nproc` worker
   processes and ranks them by fitness. *(done)*
5. The evolutionary search: population, mutation, selection on validation fitness,
   against a matched-compute random-search control.
6. A reproducible benchmark: the search vs its random-search control on a standard
   NAS benchmark (NAS-Bench-101 / 201).

## Working with AI agents

As in AIS: spawn focused subagents for isolatable work; a separate tester with
fresh context writing the tests catches the implementer's assumptions; the main
session locks the contract and runs `make ut` plus the sanitizers.
