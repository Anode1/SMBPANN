/* nb101_extract.c -- one-shot converter: NAS-Bench-101 tfrecord -> plain text.
 *
 * Reads the official `nasbench_only108.tfrecord` (epoch-108 results) and writes a
 * flat table one architecture per line, so the C experiment needs no benchmark
 * library, no Python, no protobuf runtime -- just this table. Dependency-free C99.
 *
 * TFRecord framing per record:  uint64 length (LE), uint32 CRC, <length> bytes,
 * uint32 CRC. We skip both CRCs. Each payload is a UTF-8 JSON array
 *   ["<hash>", 108, "<N*N adjacency bits>", "<comma-sep ops>", "<base64 metrics>"]
 * where the base64 decodes to a ModelMetrics protobuf; we pull each
 * EvaluationData's validation_accuracy (field 4) and test_accuracy (field 5,
 * both IEEE-754 doubles) and average over that architecture's trials.
 *
 * Output line:  N  <N*N adjacency bits>  op0 op1 .. op{N-1}  val_acc test_acc
 *   op codes: 0=input 1=conv1x1 2=conv3x3 3=maxpool3x3 4=output
 *
 * Build: cc -std=c99 -O2 -o nb101_extract validation/nb101_extract.c
 * Run:   ./nb101_extract nasbench_only108.tfrecord nasbench101.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXN     7
#define MAXPAY   4096            /* a record payload is ~400 bytes; be generous */
#define TBITS    21             /* hash table: 2^21 slots for ~423k archs */
#define TSIZE    (1u << TBITS)

typedef struct {
    char   hash[33];
    int    n;
    char   adj[MAXN*MAXN + 1];
    signed char op[MAXN];
    double vsum, tsum;          /* accumulated over trials */
    int    cnt;
} Arch;

static Arch  *tab;              /* open-addressing table, TSIZE slots */
static int   *used;            /* 0/1 occupancy */
static long   nuniq = 0;

/* FNV-1a over the hex hash string */
static uint32_t hkey(const char *s)
{
    uint32_t h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h & (TSIZE - 1u);
}

static Arch *lookup(const char *hash)
{
    uint32_t i = hkey(hash);
    while (used[i]) {
        if (strcmp(tab[i].hash, hash) == 0) return &tab[i];
        i = (i + 1u) & (TSIZE - 1u);
    }
    used[i] = 1;
    snprintf(tab[i].hash, sizeof tab[i].hash, "%s", hash);
    tab[i].cnt = 0; tab[i].vsum = tab[i].tsum = 0.0; tab[i].n = 0;
    nuniq++;
    return &tab[i];
}

/* ---- base64 decode (whitespace-skipping), returns output length ---- */
static int b64dec(const char *in, size_t inlen, unsigned char *out)
{
    static signed char T[256];
    static int init = 0;
    const char *A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    unsigned int val = 0; int bits = 0, olen = 0; size_t i;
    if (!init) {
        int k;
        for (k = 0; k < 256; k++) T[k] = -1;
        for (k = 0; k < 64; k++) T[(unsigned char)A[k]] = (signed char)k;
        init = 1;
    }
    for (i = 0; i < inlen; i++) {
        signed char c = T[(unsigned char)in[i]];
        if (c < 0) continue;                 /* skip newlines, '=', whitespace */
        val = (val << 6) | (unsigned)c; bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[olen++] = (unsigned char)((val >> bits) & 0xffu);
            val &= (1u << bits) - 1u;        /* drop consumed high bits (no overflow) */
        }
    }
    return olen;
}

/* ---- protobuf helpers ---- */
static uint64_t readvarint(const unsigned char *b, size_t len, size_t *pos)
{
    uint64_t v = 0; int shift = 0;
    while (*pos < len) {
        unsigned char c = b[(*pos)++];
        v |= (uint64_t)(c & 0x7f) << shift;
        if (!(c & 0x80)) break;
        shift += 7;
    }
    return v;
}
static double readdouble(const unsigned char *b, size_t len, size_t *pos)
{
    double d;
    if (*pos + 8 > len) { *pos = len; return 0.0; }
    memcpy(&d, b + *pos, 8); *pos += 8; return d;            /* host is LE */
}
static void skipfield(const unsigned char *b, size_t len, size_t *pos, int wt)
{
    if (wt == 0)      (void)readvarint(b, len, pos);
    else if (wt == 1) *pos += 8;
    else if (wt == 5) *pos += 4;
    else if (wt == 2) { uint64_t l = readvarint(b, len, pos); *pos += (size_t)l; }
}

/* parse one EvaluationData submessage: pull field 1 (current_epoch), 4 (val) and
 * 5 (test), all doubles */
static void parse_eval(const unsigned char *b, size_t len,
                       double *ep, double *val, double *test)
{
    size_t pos = 0;
    *ep = *val = *test = 0.0;
    while (pos < len) {
        unsigned char tag = b[pos++];
        int field = tag >> 3, wt = tag & 7;
        if (field == 1 && wt == 1)      *ep   = readdouble(b, len, &pos);
        else if (field == 4 && wt == 1) *val  = readdouble(b, len, &pos);
        else if (field == 5 && wt == 1) *test = readdouble(b, len, &pos);
        else skipfield(b, len, &pos, wt);
    }
}

/* parse ModelMetrics: a record holds this trial's checkpoints at several epochs;
 * take the FINAL one (highest current_epoch) and accumulate it into the arch (so
 * the arch averages its final accuracy over its trials). */
static void parse_metrics(const unsigned char *b, size_t len, Arch *a)
{
    size_t pos = 0;
    double bestep = -1.0, bval = 0.0, btest = 0.0;
    while (pos < len) {
        unsigned char tag = b[pos++];
        int field = tag >> 3, wt = tag & 7;
        if (field == 1 && wt == 2) {
            uint64_t l = readvarint(b, len, &pos);
            double e, v, t;
            if (pos + (size_t)l > len) break;               /* corrupt/truncated */
            parse_eval(b + pos, (size_t)l, &e, &v, &t);
            if (e >= bestep) { bestep = e; bval = v; btest = t; }
            pos += (size_t)l;
        } else skipfield(b, len, &pos, wt);
    }
    if (bestep >= 0.0) { a->vsum += bval; a->tsum += btest; a->cnt++; }
}

static int opcode(const char *s, size_t len)
{
    if (len == 5  && !strncmp(s, "input", 5))               return 0;
    if (len == 15 && !strncmp(s, "conv1x1-bn-relu", 15))    return 1;
    if (len == 15 && !strncmp(s, "conv3x3-bn-relu", 15))    return 2;
    if (len == 10 && !strncmp(s, "maxpool3x3", 10))         return 3;
    if (len == 6  && !strncmp(s, "output", 6))              return 4;
    return -1;
}

/* copy the next JSON string (payload[*p] must be the opening quote) into out,
 * advancing *p past the closing quote. Strings here contain no escaped quotes. */
static int json_str(const char *pay, size_t plen, size_t *p, char *out, size_t outsz)
{
    size_t o = 0;
    if (*p >= plen || pay[*p] != '"') return -1;
    (*p)++;
    while (*p < plen && pay[*p] != '"') {
        char c = pay[*p];
        if (c == '\\' && *p + 1 < plen) {           /* JSON escape (e.g. "\n") */
            (*p)++;
            switch (pay[*p]) {
            case 'n': c = '\n'; break;  case 't': c = '\t'; break;
            case 'r': c = '\r'; break;  case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;  default:  c = pay[*p];
            }
        }
        if (o + 1 < outsz) out[o++] = c;
        (*p)++;
    }
    if (*p >= plen) return -1;
    (*p)++;                                  /* past closing quote */
    out[o] = '\0';
    return (int)o;
}
static void json_skip_to_str(const char *pay, size_t plen, size_t *p)
{
    while (*p < plen && pay[*p] != '"') (*p)++;
}

int main(int argc, char **argv)
{
    const char *inpath  = argc > 1 ? argv[1] : "nasbench_only108.tfrecord";
    const char *outpath = argc > 2 ? argv[2] : "nasbench101.txt";
    FILE *f = fopen(inpath, "rb");
    FILE *out;
    long  nrec = 0;
    static char pay[MAXPAY];
    static unsigned char metrics[MAXPAY];
    static char hash[33], adjs[128], ops[512];

    if (!f) { fprintf(stderr, "nb101: cannot open %s\n", inpath); return 1; }
    tab  = calloc(TSIZE, sizeof *tab);
    used = calloc(TSIZE, sizeof *used);
    if (!tab || !used) { fprintf(stderr, "nb101: out of memory\n"); return 1; }

    for (;;) {
        unsigned char lenbuf[8], crc[4];
        uint64_t len; size_t got, p, i;
        int n, dim;
        Arch *a;

        if (fread(lenbuf, 1, 8, f) != 8) break;                 /* clean EOF */
        len = 0; for (i = 0; i < 8; i++) len |= (uint64_t)lenbuf[i] << (8*i);
        if (fread(crc, 1, 4, f) != 4) break;
        if (len == 0 || len >= MAXPAY) {                        /* skip oversized */
            if (fseek(f, (long)len + 4, SEEK_CUR) != 0) break;
            continue;
        }
        got = fread(pay, 1, (size_t)len, f);
        if (got != len) break;                                  /* truncated tail */
        if (fread(crc, 1, 4, f) != 4) break;
        nrec++;

        /* JSON: [ "hash", 108, "adj", "ops", "b64" ] */
        p = 0;
        json_skip_to_str(pay, len, &p);                         /* over the '[' */
        if (json_str(pay, len, &p, hash, sizeof hash) < 0) continue;
        json_skip_to_str(pay, len, &p);                         /* over ", 108, " */
        if (json_str(pay, len, &p, adjs, sizeof adjs) < 0) continue;
        json_skip_to_str(pay, len, &p);
        if (json_str(pay, len, &p, ops, sizeof ops) < 0) continue;
        json_skip_to_str(pay, len, &p);
        /* remaining quoted string is the base64 metrics */
        {
            char b64[MAXPAY];
            int  bl = json_str(pay, len, &p, b64, sizeof b64), ml;
            if (bl < 0) continue;
            ml = b64dec(b64, (size_t)bl, metrics);

            a = lookup(hash);
            /* fill topology once */
            if (a->n == 0) {
                size_t L = strlen(adjs); const char *q = ops; int k = 0;
                dim = 1; while ((size_t)(dim*dim) < L) dim++;
                a->n = dim;
                memcpy(a->adj, adjs, L); a->adj[L] = '\0';
                for (k = 0; k < dim; k++) {
                    const char *e = strchr(q, ',');
                    size_t tl = e ? (size_t)(e - q) : strlen(q);
                    a->op[k] = (signed char)opcode(q, tl);
                    q = e ? e + 1 : q + tl;
                }
                (void)n;
            }
            parse_metrics(metrics, (size_t)ml, a);
        }
    }
    fclose(f);

    out = fopen(outpath, "w");
    if (!out) { fprintf(stderr, "nb101: cannot write %s\n", outpath); return 1; }
    fprintf(out, "# N  adjacency(N*N bits)  op0..op{N-1}  val_acc test_acc"
                 "  (op: 0=in 1=c1x1 2=c3x3 3=mp 4=out)\n");
    {
        uint32_t s; long written = 0;
        for (s = 0; s < TSIZE; s++) {
            if (!used[s] || tab[s].cnt == 0) continue;
            fprintf(out, "%d %s", tab[s].n, tab[s].adj);
            { int k; for (k = 0; k < tab[s].n; k++) fprintf(out, " %d", tab[s].op[k]); }
            fprintf(out, " %.4f %.4f\n",
                    100.0 * tab[s].vsum / tab[s].cnt, 100.0 * tab[s].tsum / tab[s].cnt);
            written++;
        }
        fclose(out);
        fprintf(stderr, "nb101: %ld records -> %ld unique architectures -> %s\n",
                nrec, written, outpath);
    }
    return 0;
}
