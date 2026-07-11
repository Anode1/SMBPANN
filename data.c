/* data.c -- dataset loading and the train/test split. See data.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"

/* A line carries data unless it is blank or its first non-blank is '#'. */
static int row_is_data(const char *line)
{
    while (*line == ' ' || *line == '\t')
        line++;
    return (*line != '\0' && *line != '\n' && *line != '#');
}

/* True if fgets truncated LINE (buffer full, no newline, more file to come):
 * a bounded read must not silently split one record across two reads. */
static int line_overlong(const char *line, FILE *f)
{
    size_t len = strlen(line);

    return (len == SMB_LINE_MAX - 1 && line[len - 1] != '\n' && !feof(f));
}

/* Parse the whitespace-separated numbers on LINE into VALS (capacity CAP), and
 * report how many there were in *COUNT (which the caller checks). Returns 0, or
 * -1 on a token that is not a number. */
static int parse_row(const char *line, smb_real *vals, size_t cap, size_t *count)
{
    const char *p = line;
    size_t      n = 0;

    for (;;) {
        char  *end;
        double v;

        while (*p == ' ' || *p == '\t' || *p == '\n')
            p++;
        if (*p == '\0' || *p == '#')
            break;
        v = strtod(p, &end);
        if (end == p)                 /* nothing numeric consumed */
            return -1;
        if (n < cap)
            vals[n] = (smb_real)v;
        n++;
        p = end;
    }
    *count = n;
    return 0;
}

int dataset_load(Dataset *ds, const char *path, size_t ninput, size_t noutput)
{
    FILE     *f = NULL;
    smb_real *vals = NULL;
    char      line[SMB_LINE_MAX];
    size_t    nf = ninput + noutput;
    size_t    count = 0, s;
    int       rc = -1;

    if (ds == NULL)
        return -1;
    memset(ds, 0, sizeof *ds);
    if (path == NULL || ninput == 0 || noutput == 0)
        return -1;

    f = fopen(path, "r");
    if (f == NULL)
        return -1;

    /* pass 1: count data lines to size the allocation */
    while (fgets(line, sizeof line, f) != NULL) {
        if (line_overlong(line, f))
            goto done;
        if (row_is_data(line))
            count++;
    }
    if (count == 0)
        goto done;

    ds->ninput   = ninput;
    ds->noutput  = noutput;
    ds->nsamples = count;
    ds->x = malloc(count * ninput * sizeof *ds->x);
    ds->y = malloc(count * noutput * sizeof *ds->y);
    vals  = malloc(nf * sizeof *vals);
    if (ds->x == NULL || ds->y == NULL || vals == NULL)
        goto done;

    rewind(f);

    /* pass 2: parse each data line into its rows of x and y */
    s = 0;
    while (fgets(line, sizeof line, f) != NULL) {
        size_t nt;
        if (!row_is_data(line))
            continue;
        if (parse_row(line, vals, nf, &nt) != 0 || nt != nf)
            goto done;
        memcpy(ds->x + s * ninput,  vals,          ninput  * sizeof *vals);
        memcpy(ds->y + s * noutput, vals + ninput, noutput * sizeof *vals);
        s++;
    }
    rc = 0;

done:
    if (f != NULL)
        fclose(f);
    free(vals);
    if (rc != 0)
        dataset_free(ds);
    return rc;
}

const smb_real *dataset_input(const Dataset *ds, size_t i)
{
    return ds->x + i * ds->ninput;
}

const smb_real *dataset_target(const Dataset *ds, size_t i)
{
    return ds->y + i * ds->noutput;
}

void dataset_free(Dataset *ds)
{
    if (ds == NULL)
        return;
    free(ds->x);
    free(ds->y);
    ds->x = NULL;
    ds->y = NULL;
    ds->nsamples = 0;
}

int split_make(Split *sp, const Dataset *ds, smb_real frac, Rng *rng)
{
    size_t i, j, n, tmp;

    if (sp == NULL)
        return -1;
    memset(sp, 0, sizeof *sp);
    if (ds == NULL || ds->nsamples == 0 || rng == NULL)
        return -1;

    n = ds->nsamples;
    sp->order = malloc(n * sizeof *sp->order);
    if (sp->order == NULL)
        return -1;
    for (i = 0; i < n; i++)
        sp->order[i] = i;

    /* Fisher-Yates shuffle, so a fixed seed gives a fixed partition */
    for (i = n; i > 1; i--) {
        j = (size_t)(rng_u32(rng) % (uint32_t)i);
        tmp = sp->order[i - 1];
        sp->order[i - 1] = sp->order[j];
        sp->order[j] = tmp;
    }

    if (frac < 0.0f)
        frac = 0.0f;
    if (frac > 1.0f)
        frac = 1.0f;
    sp->n = n;
    sp->ntrain = (size_t)((double)frac * (double)n);
    return 0;
}

size_t split_train_count(const Split *sp) { return sp->ntrain; }
size_t split_test_count(const Split *sp)  { return sp->n - sp->ntrain; }

size_t split_train_index(const Split *sp, size_t p) { return sp->order[p]; }
size_t split_test_index(const Split *sp, size_t p)  { return sp->order[sp->ntrain + p]; }

void split_free(Split *sp)
{
    if (sp == NULL)
        return;
    free(sp->order);
    sp->order = NULL;
    sp->n = 0;
    sp->ntrain = 0;
}
