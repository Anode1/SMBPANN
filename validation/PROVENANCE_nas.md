# NAS benchmark data: where it comes from and how to regenerate it

The archives and the large derived table are gitignored because they are big and re-fetchable. Every
number in the paper2 measurement work regenerates from these four steps.

## NAS-Bench-101 (423,624 architectures, CIFAR-10, 3 independent training runs each)

    curl -o nasbench_only108.tfrecord \
      https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord      # 499 MB, epoch-108 only
    curl -o nasbench_full.tfrecord \
      https://storage.googleapis.com/nasbench/nasbench_full.tfrecord         # 1989 MB, budgets 4/12/36/108

    cc -std=c99 -W -Wall -O2 -o nb101_trials validation/nb101_trials.c
    ./nb101_trials nasbench_only108.tfrecord validation/nasbench101_trials.txt

`validation/nb101_extract.c` is the ORIGINAL converter and averages an architecture's repeated runs into
one number. That is the right table for ranking architectures and the wrong one for studying an
estimator, because averaging destroys the replicate structure the measurement depends on.
`validation/nb101_trials.c` keeps every run.

**Extraction check:** the mean of the three recovered runs reproduces the averaged table
(`validation/nasbench101.txt`) to within 1e-4 percentage points on all 423,624 rows.

**Replicate noise, measured:** within-architecture SD of validation accuracy is 0.328 at the median,
0.786 at the 90th percentile, 42.68 at the 99th, and 48.0 at maximum. About one architecture in a
hundred sometimes trains and sometimes collapses to the accuracy of guessing.

## NAS-Bench-201 / NATS-Bench (15,625 architectures, 3 datasets, 3 seeds, 200 epochs)

The original release is distributed only through Google Drive and needs PyTorch to unpickle. Syne Tune
publishes a converted copy on Hugging Face that keeps the per-seed values and is a plain NumPy array,
so it needs neither:

    B=https://huggingface.co/datasets/synetune/blackbox-repository/resolve/main/nasbench201
    curl -L -o nb201_objectives.npy          $B/objectives_evaluations.npy    # 751 MB
    curl -L -o nb201_hyperparameters.parquet $B/hyperparameters.parquet
    curl -L -o nb201_metadata.json           $B/metadata.json

`objectives_evaluations.npy` is float32 with shape (3, 15625, 3, 200, 7) =
(task, architecture, seed, epoch, objective). Tasks are cifar10, cifar100, ImageNet16-120. Objectives are
valid_error, train_error, runtime, elapsed_time, latency, flops, params. The `.npy` header is plain text
followed by a raw float32 block, so it is parseable in C without NumPy.

Caveat carried from the converter: a few (architecture, seed) cells were never evaluated in the original
release, and those specific cells are filled with the mean of the remaining seeds. Real per-seed values
survive wherever they existed.
