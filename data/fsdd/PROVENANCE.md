# FSDD — Free Spoken Digit Dataset (parked for the real-audio direction)

Small, clean, labeled audio dataset for the **real-audio extension** of the emergence work
(a validation demo or the next paper). NOT used by this paper's results, which stay on the
synthetic pitch task. Parked here, ready.

## Source
- Repo: https://github.com/Jakobovski/free-spoken-digit-dataset
- Fetched: master branch tarball, 2026-07-20
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0)

## Re-fetch (the WAVs are gitignored, not committed)
    curl -sSL https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.tar.gz \
      | tar xz && mv free-spoken-digit-dataset-master/recordings ./recordings

## Characterization (verified on fetch)
- 3000 files, `recordings/<digit>_<speaker>_<index>.wav`
- Balanced: 300 clips per digit 0-9
- 6 speakers: george, jackson, lucas, nicolas, theo, yweweler
- Format: WAV, PCM signed 16-bit little-endian, **8 kHz, mono**
- Duration ~0.25-0.5 s per clip (~2000-4000 samples)
- ~27 MB extracted

## Why this shape (vs. the rejected music library)
Labeled (supervised task exists), mono (no channel confound), lossless PCM (no codec
artifacts), pure-C readable (44-byte header + int16 samples, no decoder), redistributable.
Speech carries the two structures we care about: **voiced segments = periodic (pitch)** and
**formants = local spectral peaks** -- so an emerged front-end can be checked against known
speech structure (Smith & Lewicki 2006 style), even without a planted ground-truth filter.

## Pipeline still needed before use (the "long" part)
Fixed-length framing (pad/window the variable-length clips), a task definition adapted to the
energy-budget GA, and a net scale larger than the synthetic N<=40. Deliberate next step.
