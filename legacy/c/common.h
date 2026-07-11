/* common.h -- shared numeric type and limits for SMBPANN.
 *
 * SMBPANN in C99. Coding ideology follows the AIS project's doc/dev/STYLE.md
 * (K&R/Robbins lineage, one concept per .c/.h, bounded strings, return codes,
 * single-exit cleanup). The one place SMBPANN must depart from AIS's strict
 * no-heap rule is documented in doc/dev/STYLE.md: a network's weights ARE the
 * model and scale with topology, so net_new/trainer_new allocate ONCE at
 * construction (Power of Ten rule 3) and the train/infer hot path allocates
 * nothing.
 *
 * Copyright (C) 2001 Vasili Gavrilov. GNU GPL v2 or later.
 */
#ifndef SMB_COMMON_H
#define SMB_COMMON_H

#include <stddef.h>

/* The scalar type for weights, activations, and error signals. `float` (32-bit)
 * halves the memory of `double` -- at a fixed RAM budget that doubles the GA
 * population of candidate networks that fits -- and carries ample precision for
 * backpropagation. One typedef to flip the whole engine to `double`. */
typedef float smb_real;

/* Cap on network DEPTH (number of layers, input included), NOT on width. It
 * bounds only the fixed pointer tables inside `Net`/`Trainer`, so the structs
 * are a fixed size computable by hand; per-layer node counts are unbounded and
 * live on the heap, sized by topology at construction. 32 is far beyond any
 * feed-forward net in the 1997 thesis (2-5 layers). */
#define SMB_MAX_LAYERS 32

#endif /* SMB_COMMON_H */
