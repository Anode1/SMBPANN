/* data.h -- plain-text datasets and a train/test split.
 *
 * A Dataset is loaded once from a whitespace-separated text file: each
 * non-blank, non-'#' line holds NINPUT input values then NOUTPUT target values,
 * kept in two flat arrays (all inputs, all targets). A Split is a shuffled
 * permutation of the sample indices partitioned into a train and a test part --
 * it holds only the ordering, never a copy of the data.
 *
 * Ported from the Ada smbpann-data package.
 */
#ifndef SMB_DATA_H
#define SMB_DATA_H

#include <stddef.h>

#include "common.h"
#include "rng.h"

typedef struct {
    size_t    ninput;
    size_t    noutput;
    size_t    nsamples;
    smb_real *x;   /* nsamples * ninput, row-major  */
    smb_real *y;   /* nsamples * noutput, row-major */
} Dataset;

/* Load PATH into DS. Returns 0, or -1 on an unreadable file, an over-long line,
 * or a line whose field count is not NINPUT + NOUTPUT (DS zeroed on failure). */
int dataset_load(Dataset *ds, const char *path, size_t ninput, size_t noutput);

/* Input / target row of sample I (0-based, I < nsamples). */
const smb_real *dataset_input(const Dataset *ds, size_t i);
const smb_real *dataset_target(const Dataset *ds, size_t i);

/* Release both flat arrays. NULL-safe. */
void dataset_free(Dataset *ds);

typedef struct {
    size_t *order;    /* a permutation of 0 .. n-1                      */
    size_t  n;
    size_t  ntrain;   /* the first ntrain of order[] are the train set */
} Split;

/* Shuffle the sample indices with RNG and take FRAC (0..1) as training.
 * Returns 0, or -1 on bad args / OOM (SP zeroed on failure). */
int split_make(Split *sp, const Dataset *ds, smb_real frac, Rng *rng);

size_t split_train_count(const Split *sp);
size_t split_test_count(const Split *sp);
size_t split_train_index(const Split *sp, size_t p);  /* p < train_count */
size_t split_test_index(const Split *sp, size_t p);   /* p < test_count  */

/* Release the permutation. NULL-safe. */
void split_free(Split *sp);

#endif /* SMB_DATA_H */
