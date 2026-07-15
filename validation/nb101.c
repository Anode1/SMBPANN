/* nb101.c -- crossover on the REAL NAS-Bench-101 cell space, testing whether a
 * DIMENSION-AWARE crossover (recombining the two axes of the space -- wiring and
 * labeling -- with node alignment) beats random search and beats a naive flat
 * crossover that ignores the space's structure.
 *
 * The cell is a DAG on up to 7 nodes: node 0 = input, node n-1 = output, the
 * interior nodes are labeled conv1x1 / conv3x3 / maxpool3x3. An architecture thus
 * has two orthogonal dimensions -- its WIRING (adjacency) and its LABELING (ops) --
 * plus a graph/path view. Fitness is a table lookup into validation/nasbench101.txt
 * (extracted by nb101_extract.c), so no network is trained.
 *
 * The one hard piece is the fitness oracle: NAS-Bench-101 dedupes isomorphic
 * graphs, so a recombined child must be PRUNED (drop nodes off every input->output
 * path) and CANONICALIZED (relabel interior nodes to a canonical order) before
 * lookup. We canonicalize by brute force over the <=5! interior-node permutations
 * -- exact, dependency-free -- and that same alignment powers the aligned crossover.
 *
 * Self-contained C99.  Build: make nb101   Run: ./nb101 validation/nasbench101.txt
 * Self-test the oracle:      ./nb101 validation/nasbench101.txt --selftest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXN 7

/* ---- PRNG (xorshift) ---- */
static uint32_t rs = 1u;
static uint32_t r32(void){ uint32_t x=rs; x^=x<<13; x^=x>>17; x^=x<<5; rs=x; return x; }
static void rseed(uint32_t s){ rs = s ? s : 1u; }
static uint32_t rbelow(uint32_t m){ return m ? r32()%m : 0u; }

/* ---- a cell graph ---- */
typedef struct {
    int           n;              /* node count, 2..7 (input=0, output=n-1) */
    unsigned char adj[MAXN][MAXN];/* directed adjacency (edge a->b) */
    signed char   op[MAXN];       /* 0=input 1=c1x1 2=c3x3 3=maxpool 4=output */
} Graph;

/* ---- interior-node permutation tables (for canonicalization/alignment) ---- */
static int  perms[6][120][5];     /* perms[m] = the m! permutations of 0..m-1 */
static int  nperm[6];

static void gen_perms(void)
{
    int m;
    for (m = 0; m <= 5; m++) {
        int a[5], i, cnt = 0, c[5];
        for (i = 0; i < m; i++) a[i] = i;
        /* Heap's algorithm, iterative */
        for (i = 0; i < m; i++) c[i] = 0;
        { int k; for (k = 0; k < m; k++) perms[m][cnt][k] = a[k]; }
        cnt = 1;
        i = 0;
        while (i < m) {
            if (c[i] < i) {
                int sw = (i % 2 == 0) ? 0 : c[i];
                int t = a[sw]; a[sw] = a[i]; a[i] = t;
                { int k; for (k = 0; k < m; k++) perms[m][cnt][k] = a[k]; }
                cnt++;
                c[i]++;
                i = 0;
            } else { c[i] = 0; i++; }
        }
        nperm[m] = (m == 0) ? 1 : cnt;
    }
}

/* Pack a graph (with a GIVEN node labeling) into a comparable/uniqueness key:
 * low 3 bits = n, then the off-diagonal adjacency cells in fixed order, then the
 * interior op codes (2 bits each). Same n => same layout, so it is a total order. */
static uint64_t encode(const Graph *g)
{
    uint64_t key = (uint64_t)g->n;
    int shift = 3, a, b, k;
    for (a = 0; a < g->n; a++)
        for (b = 0; b < g->n; b++)
            if (a != b) { key |= (uint64_t)(g->adj[a][b] & 1u) << shift; shift++; }
    for (k = 1; k < g->n - 1; k++) {
        key |= (uint64_t)((g->op[k] - 1) & 3u) << shift; shift += 2;
    }
    return key;
}

/* Canonical key = min encode() over all relabelings that fix input(0) and
 * output(n-1) and permute the interior nodes. Isomorphic graphs => same key. */
static uint64_t canonical(const Graph *g)
{
    int m = g->n - 2;                 /* interior nodes at indices 1..n-2 */
    uint64_t best = ~(uint64_t)0;
    int pi;
    if (m < 0) m = 0;
    for (pi = 0; pi < nperm[m]; pi++) {
        Graph h; int a, b;
        int map[MAXN];                /* new index -> old index */
        map[0] = 0; map[g->n - 1] = g->n - 1;
        for (a = 0; a < m; a++) map[1 + a] = 1 + perms[m][pi][a];
        h.n = g->n;
        for (a = 0; a < g->n; a++) h.op[a] = g->op[map[a]];
        for (a = 0; a < g->n; a++)
            for (b = 0; b < g->n; b++)
                h.adj[a][b] = g->adj[map[a]][map[b]];
        { uint64_t e = encode(&h); if (e < best) best = e; }
    }
    return best;
}

/* edge count */
static int nedges(const Graph *g)
{
    int a, b, c = 0;
    for (a = 0; a < g->n; a++) for (b = 0; b < g->n; b++) c += g->adj[a][b] ? 1 : 0;
    return c;
}

/* Prune nodes that are not on any input->output path, compacting the survivors in
 * original index order. Returns 1 if the pruned graph is valid (input reaches
 * output, >=2 nodes), else 0. On success *out holds the reduced graph. */
static int prune(const Graph *g, Graph *out)
{
    int fromIn[MAXN] = {0}, toOut[MAXN] = {0}, keep[MAXN];
    int i, j, changed, cnt = 0, idx[MAXN];

    fromIn[0] = 1;
    do { changed = 0;
        for (i = 0; i < g->n; i++) if (fromIn[i])
            for (j = 0; j < g->n; j++) if (g->adj[i][j] && !fromIn[j]) { fromIn[j] = 1; changed = 1; }
    } while (changed);

    toOut[g->n - 1] = 1;
    do { changed = 0;
        for (i = 0; i < g->n; i++) if (toOut[i])
            for (j = 0; j < g->n; j++) if (g->adj[j][i] && !toOut[j]) { toOut[j] = 1; changed = 1; }
    } while (changed);

    for (i = 0; i < g->n; i++) {
        keep[i] = (fromIn[i] && toOut[i]);
        if (keep[i]) idx[cnt++] = i;
    }
    if (cnt < 2 || !keep[0] || !keep[g->n - 1]) return 0;   /* input must reach output */

    out->n = cnt;
    { int a, b, ra, rb;
      for (a = 0; a < cnt; a++) out->op[a] = g->op[idx[a]];
      for (ra = 0; ra < cnt; ra++) for (rb = 0; rb < cnt; rb++)
          out->adj[ra][rb] = 0;
      for (ra = 0; ra < cnt; ra++) for (rb = 0; rb < cnt; rb++)
          out->adj[ra][rb] = g->adj[idx[ra]][idx[rb]];
      (void)a; (void)b;
    }
    return 1;
}

/* ---- fitness table: canonical key -> (val, test) accuracy ---- */
#define TBITS 21
#define TSIZE (1u << TBITS)
typedef struct { uint64_t key; float val, test; } Slot;
static Slot *tab;
static long  ntab = 0;

static void tab_put(uint64_t key, float val, float test)
{
    uint32_t i = (uint32_t)((key * 11400714819323198485ull) >> (64 - TBITS));
    while (tab[i].key) {
        if (tab[i].key == key) return;             /* already present */
        i = (i + 1u) & (TSIZE - 1u);
    }
    tab[i].key = key; tab[i].val = val; tab[i].test = test; ntab++;
}
/* returns 1 and fills val/test if the key is found */
static int tab_get(uint64_t key, float *val, float *test)
{
    uint32_t i = (uint32_t)((key * 11400714819323198485ull) >> (64 - TBITS));
    while (tab[i].key) {
        if (tab[i].key == key) { if(val)*val=tab[i].val; if(test)*test=tab[i].test; return 1; }
        i = (i + 1u) & (TSIZE - 1u);
    }
    return 0;
}

/* fitness oracle: validate -> prune -> <=9 edges -> canonical -> lookup.
 * Returns 1 and fills val/test, or 0 if the child is not a valid catalogued arch. */
static int fitness(const Graph *g, float *val, float *test)
{
    Graph p;
    if (g->n < 2 || g->n > MAXN) return 0;
    if (!prune(g, &p)) return 0;
    if (nedges(&p) > 9) return 0;
    return tab_get(canonical(&p), val, test);
}

/* parse one table line: "N adjbits op0..op{N-1} val test" */
static int parse_line(const char *line, Graph *g, float *val, float *test)
{
    int n, k; const char *p = line; char *e;
    n = (int)strtol(p, &e, 10); if (e == p || n < 2 || n > MAXN) return 0;
    p = e; while (*p == ' ') p++;
    /* adjacency: n*n chars of 0/1 */
    { int a, b;
      for (a = 0; a < n; a++) for (b = 0; b < n; b++) {
          if (*p != '0' && *p != '1') return 0;
          g->adj[a][b] = (unsigned char)(*p - '0'); p++;
      }
    }
    g->n = n;
    for (k = 0; k < n; k++) {
        long o = strtol(p, &e, 10); if (e == p) return 0; g->op[k] = (signed char)o; p = e;
    }
    { double v = strtod(p, &e); if (e == p) return 0; *val = (float)v; p = e;
      { double t = strtod(p, &e); if (e == p) return 0; *test = (float)t; } }
    return 1;
}

static long load(const char *path)
{
    FILE *f = fopen(path, "r");
    char  line[512];
    long  nread = 0;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        Graph g, p; float val, test;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (!parse_line(line, &g, &val, &test)) continue;
        nread++;
        if (!prune(&g, &p)) { p = g; }            /* stored graphs are pruned already */
        tab_put(canonical(&p), val, test);
    }
    fclose(f);
    return nread;
}

/* ---- self-test: re-read some lines and confirm the oracle finds them ---- */
static int selftest(const char *path)
{
    FILE *f = fopen(path, "r");
    char  line[512];
    long  n = 0, ok = 0, iso_ok = 0;
    if (!f) return 1;
    while (fgets(line, sizeof line, f) && n < 5000) {
        Graph g; float val, test, gv, gt;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (!parse_line(line, &g, &val, &test)) continue;
        n++;
        /* (a) the exact graph is found with matching accuracy */
        if (fitness(&g, &gv, &gt) && fabsf(gt - test) < 0.001f) ok++;
        /* (b) a random interior relabeling (isomorph) maps to the same accuracy */
        { Graph h = g; int m = g.n - 2, i;
          if (m >= 2) {
            int pj = 1 + (int)rbelow((uint32_t)nperm[m] - 1); /* a non-identity perm */
            int map[MAXN]; float hv, ht; int a, b;
            map[0]=0; map[g.n-1]=g.n-1;
            for (i=0;i<m;i++) map[1+i]=1+perms[m][pj][i];
            h.n=g.n;
            for (a=0;a<g.n;a++) h.op[a]=g.op[map[a]];
            for (a=0;a<g.n;a++) for (b=0;b<g.n;b++) h.adj[a][b]=g.adj[map[a]][map[b]];
            if (fitness(&h,&hv,&ht) && fabsf(ht - test) < 0.001f) iso_ok++;
            else iso_ok += 0;
          } else iso_ok++;   /* nothing to permute */
        }
    }
    fclose(f);
    printf("selftest: %ld lines, exact-found %ld/%ld, isomorph-found %ld/%ld\n",
           n, ok, n, iso_ok, n);
    return (ok == n && iso_ok == n) ? 0 : 2;
}

/* ===================== the search: individuals, operators, GA ============== */

typedef struct { Graph g; float vf, tf; int valid; } Indiv;

static long   QB, QUSED;                  /* query budget and queries used */
static long   OFF_TOT, OFF_VAL;           /* offspring produced / of them valid */
static double DIVERSITY;                  /* final-population mean pairwise gene distance */

static int eval_indiv(Indiv *x)
{
    float v, t;
    QUSED++;
    if (fitness(&x->g, &v, &t)) { x->valid = 1; x->vf = v; x->tf = t; return 1; }
    x->valid = 0; x->vf = x->tf = 0.0f; return 0;
}

/* a raw random 7-node upper-triangular cell (input=0, output=6); ~PEDGE per edge.
 * No validity rejection -- an invalid draw is a real spent query (matched compute). */
#define PEDGE 38
static void random_raw(Graph *g)
{
    int i, j;
    g->n = MAXN; g->op[0] = 0; g->op[MAXN - 1] = 4;
    for (i = 1; i < MAXN - 1; i++) g->op[i] = (signed char)(1 + rbelow(3));
    for (i = 0; i < MAXN; i++) for (j = 0; j < MAXN; j++) g->adj[i][j] = 0;
    for (i = 0; i < MAXN; i++)
        for (j = i + 1; j < MAXN; j++)
            g->adj[i][j] = (unsigned char)(rbelow(100) < PEDGE);
}

/* one light mutation, shared by every GA arm (so arms differ ONLY in crossover):
 * flip a random edge, or re-roll a random interior op. */
static void mutate(Graph *g)
{
    if (rbelow(100) < 70u) {
        if (r32() & 1u) { int i = (int)rbelow(MAXN - 1); int j = i + 1 + (int)rbelow(MAXN - 1 - i);
                          g->adj[i][j] ^= 1u; }
        else            { int k = 1 + (int)rbelow(MAXN - 2); g->op[k] = (signed char)(1 + rbelow(3)); }
    }
}

/* in/out degree of interior node k in an upper-triangular cell */
static int indeg(const Graph *g, int k){ int i,d=0; for(i=0;i<g->n;i++) d+=g->adj[i][k]; return d; }
static int outdeg(const Graph *g,int k){ int j,d=0; for(j=0;j<g->n;j++) d+=g->adj[k][j]; return d; }

/* ---- crossover operators (all produce a 7-node upper-triangular child) ---- */

/* FLAT: per-cell/op uniform mix, NO alignment (the competing-conventions strawman) */
static void xover_flat(const Graph *a, const Graph *b, Graph *c)
{
    int i, j;
    c->n = MAXN; c->op[0] = 0; c->op[MAXN-1] = 4;
    for (i = 1; i < MAXN-1; i++) c->op[i] = (r32()&1u) ? a->op[i] : b->op[i];
    for (i = 0; i < MAXN; i++) for (j = 0; j < MAXN; j++) c->adj[i][j] = 0;
    for (i = 0; i < MAXN; i++) for (j = i+1; j < MAXN; j++)
        c->adj[i][j] = (r32()&1u) ? a->adj[i][j] : b->adj[i][j];
}

/* AXIS: recombine the two DIMENSIONS -- the whole wiring from one parent, the whole
 * labeling from the other (dimension-factored crossover). */
static void xover_axis(const Graph *a, const Graph *b, Graph *c)
{
    const Graph *wa = (r32()&1u) ? a : b;      /* wiring donor  */
    const Graph *la = (r32()&1u) ? a : b;      /* labeling donor */
    int i, j;
    c->n = MAXN; c->op[0] = 0; c->op[MAXN-1] = 4;
    for (i = 1; i < MAXN-1; i++) c->op[i] = la->op[i];
    for (i = 0; i < MAXN; i++) for (j = 0; j < MAXN; j++) c->adj[i][j] = 0;
    for (i = 0; i < MAXN; i++) for (j = i+1; j < MAXN; j++) c->adj[i][j] = wa->adj[i][j];
}

/* ALIGNED: first match B's interior nodes to A's by ROLE (op + in/out degree),
 * bringing B into A's coordinate frame; then recombine node-wise -- each node's
 * whole incoming-edge module and its op come together from one aligned parent.
 * This is "crossover through the dimensions with the competing conventions removed." */
static void xover_aligned(const Graph *a, const Graph *b, Graph *c)
{
    int m = a->n - 2, pi, bestpi = 0, bestcost = 1 << 30;
    Graph bp; int i, j;

    /* choose the interior permutation of B minimizing role-mismatch to A */
    for (pi = 0; pi < nperm[m]; pi++) {
        int cost = 0, k;
        for (k = 1; k <= m; k++) {
            int bk = 1 + perms[m][pi][k-1];
            cost += (a->op[k] != b->op[bk]) ? 2 : 0;
            cost += abs(indeg(a,k) - indeg(b,bk));
            cost += abs(outdeg(a,k) - outdeg(b,bk));
        }
        if (cost < bestcost) { bestcost = cost; bestpi = pi; }
    }
    /* B in A's frame */
    { int map[MAXN]; map[0]=0; map[a->n-1]=a->n-1;
      for (i=0;i<m;i++) map[1+i]=1+perms[m][bestpi][i];
      bp.n=a->n;
      for (i=0;i<a->n;i++) bp.op[i]=b->op[map[i]];
      for (i=0;i<a->n;i++) for (j=0;j<a->n;j++) bp.adj[i][j]=b->adj[map[i]][map[j]];
    }
    /* node-wise recombination in A's topological frame (upper-triangular) */
    c->n = MAXN; c->op[0]=0; c->op[MAXN-1]=4;
    for (i=0;i<MAXN;i++) for (j=0;j<MAXN;j++) c->adj[i][j]=0;
    for (j = 1; j < MAXN; j++) {
        const Graph *src = (r32()&1u) ? a : &bp;      /* whole node-j module from one parent */
        if (j < MAXN-1) c->op[j] = src->op[j];
        for (i = 0; i < j; i++) c->adj[i][j] = src->adj[i][j];
    }
}

/* enumerate input(0)->output(n-1) directed paths (node sequences), capped at MAXP */
#define MAXP 64
static int g_pc;
static void dfs_paths(const Graph *g, int u, int *path, int depth,
                      int paths[][MAXN], int *plen)
{
    int v;
    path[depth] = u;
    if (u == g->n - 1) {
        if (g_pc < MAXP) { int k; for (k = 0; k <= depth; k++) paths[g_pc][k] = path[k];
                           plen[g_pc] = depth + 1; g_pc++; }
        return;
    }
    for (v = u + 1; v < g->n; v++) if (g->adj[u][v] && g_pc < MAXP)
        dfs_paths(g, v, path, depth + 1, paths, plen);
}
static int enum_paths(const Graph *g, int paths[][MAXN], int *plen)
{ int path[MAXN]; g_pc = 0; dfs_paths(g, 0, path, 0, paths, plen); return g_pc; }

/* PATH/SUBGRAPH transplant: a cell is a set of input->output paths; recombine whole
 * paths from both parents (a coherent computational sub-structure at a time), each
 * path bringing its nodes' ops from its own parent, greedily under the <=9 edge cap.
 * Recombination along the GRAPH dimension. */
static void xover_path(const Graph *a, const Graph *b, Graph *c)
{
    int pa[MAXP][MAXN], la[MAXP], pb[MAXP][MAXN], lb[MAXP];
    int na = enum_paths(a, pa, la), nb = enum_paths(b, pb, lb);
    int total = na + nb, ord[2*MAXP], i, o;

    c->n = MAXN; c->op[0] = 0; c->op[MAXN-1] = 4;
    for (i = 1; i < MAXN-1; i++) c->op[i] = 1;                 /* valid default */
    for (i = 0; i < MAXN; i++) { int j; for (j = 0; j < MAXN; j++) c->adj[i][j] = 0; }

    for (i = 0; i < total; i++) ord[i] = i;
    for (i = total - 1; i > 0; i--) { int j = (int)rbelow((uint32_t)i + 1); int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }

    for (o = 0; o < total; o++) {
        int id = ord[o], fromA = (id < na);
        const Graph *P = fromA ? a : b;
        int (*pp)[MAXN] = fromA ? pa : pb;
        int idx = fromA ? id : id - na, len = fromA ? la[id] : lb[idx];
        Graph t = *c; int k;
        for (k = 0; k + 1 < len; k++) t.adj[pp[idx][k]][pp[idx][k+1]] = 1;
        for (k = 1; k + 1 < len; k++) { int nd = pp[idx][k]; t.op[nd] = P->op[nd]; }
        if (nedges(&t) <= 9) *c = t;                           /* keep if it fits */
    }
}

typedef enum { X_FLAT, X_AXIS, X_ALIGNED, X_PATH } Xkind;
static void crossover(Xkind k, const Graph *a, const Graph *b, Graph *c)
{
    if (k == X_FLAT) xover_flat(a, b, c);
    else if (k == X_AXIS) xover_axis(a, b, c);
    else if (k == X_ALIGNED) xover_aligned(a, b, c);
    else xover_path(a, b, c);
}

/* pure random search: best test-acc (by best val) among QB random raw draws */
static float rand_search(uint32_t seed)
{
    float bestv = -1.0f, bestt = 0.0f;
    rseed(seed); QUSED = 0;
    while (QUSED < QB) {
        Indiv x; random_raw(&x.g); eval_indiv(&x);
        if (x.valid && x.vf > bestv) { bestv = x.vf; bestt = x.tf; }
    }
    return bestt;
}

/* one GA run with crossover operator XK; returns best test-acc (selected by val).
 * Matched compute: at most QB oracle queries; elites are cached, not re-queried. */
static float ga(Xkind xk, int pop, int elite, uint32_t seed)
{
    Indiv P[64], Q[64];
    int   i, k;
    float bestv = -1.0f, bestt = 0.0f;

    rseed(seed); QUSED = 0; OFF_TOT = 0; OFF_VAL = 0;
    for (i = 0; i < pop; i++) { random_raw(&P[i].g); eval_indiv(&P[i]);
        if (P[i].valid && P[i].vf > bestv) { bestv = P[i].vf; bestt = P[i].tf; } }

    while (QUSED < QB) {
        int idx[64];
        for (i = 0; i < pop; i++) idx[i] = i;
        for (i = 0; i < pop; i++) for (k = i+1; k < pop; k++)
            if (P[idx[k]].vf > P[idx[i]].vf) { int t=idx[i]; idx[i]=idx[k]; idx[k]=t; }
        for (i = 0; i < elite; i++) Q[i] = P[idx[i]];
        for (i = elite; i < pop && QUSED < QB; i++) {
            int a = idx[(int)rbelow((uint32_t)elite)], b = idx[(int)rbelow((uint32_t)elite)];
            crossover(xk, &P[a].g, &P[b].g, &Q[i].g);
            mutate(&Q[i].g);
            eval_indiv(&Q[i]);
            OFF_TOT++; if (Q[i].valid) OFF_VAL++;
            if (Q[i].valid && Q[i].vf > bestv) { bestv = Q[i].vf; bestt = Q[i].tf; }
        }
        for (; i < pop; i++) Q[i] = P[i];         /* budget hit mid-fill: carry over */
        for (i = 0; i < pop; i++) P[i] = Q[i];
    }
    /* final-population genotypic diversity: mean pairwise gene distance (21 edge
     * bits + 5 interior ops), the direct measure of the offspring variety that
     * crossover-plus-selection depends on. */
    { int a, b, u, v; long sum = 0, cnt = 0;
      for (a = 0; a < pop; a++) for (b = a+1; b < pop; b++) {
          int d = 0;
          for (u = 0; u < MAXN; u++) for (v = u+1; v < MAXN; v++)
              d += (P[a].g.adj[u][v] != P[b].g.adj[u][v]);
          for (u = 1; u < MAXN-1; u++) d += (P[a].g.op[u] != P[b].g.op[u]);
          sum += d; cnt++;
      }
      DIVERSITY = cnt ? (double)sum / (double)cnt : 0.0;
    }
    return bestt;
}

/* ---- paired Wilcoxon signed-rank z (arm - baseline), normal approx ---- */
typedef struct { double a; int s; } Rk;
static int rk_cmp(const void *x, const void *y)
{ double d = ((const Rk*)x)->a - ((const Rk*)y)->a; return d<0?-1:(d>0?1:0); }
static double wilcoxon_z(const double *d, int n)
{
    Rk *r = malloc((size_t)n * sizeof *r);
    int i, m = 0; double Wp = 0, mean, var, z;
    for (i = 0; i < n; i++) if (d[i] != 0.0) { r[m].a = fabs(d[i]); r[m].s = d[i]>0?1:-1; m++; }
    if (m < 1) { free(r); return 0.0; }
    qsort(r, (size_t)m, sizeof *r, rk_cmp);
    for (i = 0; i < m; ) {                         /* average ranks over ties */
        int j = i; while (j < m && r[j].a == r[i].a) j++;
        double rank = (i + 1 + j) / 2.0;           /* mean of ranks i+1..j */
        int t; for (t = i; t < j; t++) if (r[t].s > 0) Wp += rank;
        i = j;
    }
    mean = (double)m * (m + 1) / 4.0;
    var  = (double)m * (m + 1) * (2*m + 1) / 24.0;
    z = (var > 0) ? (Wp - mean) / sqrt(var) : 0.0;
    free(r);
    return z;
}

/* read a comma-separated budget list from env into buf, return count (or defaults) */
static int parse_budgets(int *buf, int max)
{
    const char *e = getenv("BUDGETS");
    int n = 0;
    if (!e || !*e) {
        int def[4] = {50, 150, 300, 600}, i;
        for (i = 0; i < 4 && i < max; i++) buf[i] = def[i];
        return (max < 4) ? max : 4;
    }
    while (*e && n < max) {
        char *end; long v = strtol(e, &end, 10);
        if (end == e) break;
        if (v > 0) buf[n++] = (int)v;
        e = (*end) ? end + 1 : end;
    }
    return n ? n : (buf[0] = 300, 1);
}

static int envint(const char *k, int def){ const char *e = getenv(k); return e && *e ? atoi(e) : def; }

static void run_experiment(void)
{
    int budgets[16];
    int nb  = parse_budgets(budgets, 16);
    int pop = envint("POP", 20), elite = envint("ELITE", 5), S = envint("SEEDS", 800), bi;
    const char *names[4] = {"flat", "axis", "aligned", "path"};
    Xkind kinds[4] = {X_FLAT, X_AXIS, X_ALIGNED, X_PATH};
    enum { NARM = 4 };

    if (pop > 64) pop = 64;
    if (elite >= pop) elite = pop / 4;
    printf("\nNAS-Bench-101 crossover study: random vs flat vs axis vs aligned\n");
    printf("pop %d, elite %d, %d seeds/cell; mean best TEST-accuracy (selected by val)\n",
           pop, elite, S);
    printf("wins/losses vs random and vs flat; Wilcoxon signed-rank z (arm-baseline)\n");

    for (bi = 0; bi < nb; bi++) {
        int b = budgets[bi], s, ai;
        double *rnd = malloc((size_t)S*sizeof(double));
        double *arm[NARM];
        double vrate[NARM] = {0,0,0,0}, dvsty[NARM] = {0,0,0,0};
        double mr = 0;
        for (ai=0; ai<NARM; ai++) arm[ai] = malloc((size_t)S*sizeof(double));
        QB = b;
        for (s = 0; s < S; s++) {
            uint32_t sd = (uint32_t)(s + 1);      /* SAME seed => paired: same start */
            rnd[s] = rand_search(sd);
            for (ai=0; ai<NARM; ai++) {
                arm[ai][s] = ga(kinds[ai], pop, elite, sd);
                if (OFF_TOT) vrate[ai] += (double)OFF_VAL / OFF_TOT;
                dvsty[ai] += DIVERSITY;
            }
            mr += rnd[s];
        }
        printf("\nbudget %d evals   random mean %.3f\n", b, mr/S);
        printf("  %-8s %9s %14s %14s %9s %8s %8s\n",
               "arm","mean","win/loss vRND","win/loss vFLAT","z vRND","valid%","diversity");
        for (ai=0; ai<NARM; ai++) {
            double ma=0, *dv = malloc((size_t)S*sizeof(double)); int wv=0,lv=0,wf=0,lf=0;
            for (s=0;s<S;s++){ ma+=arm[ai][s];
                dv[s]=arm[ai][s]-rnd[s];
                if(arm[ai][s]>rnd[s])wv++; else if(arm[ai][s]<rnd[s])lv++;
                if(arm[ai][s]>arm[0][s])wf++; else if(arm[ai][s]<arm[0][s])lf++; }
            printf("  %-8s %9.3f %7d/%-6d %7d/%-6d %9.2f %7.0f%% %8.2f\n",
                   names[ai], ma/S, wv, lv, wf, lf, wilcoxon_z(dv,S),
                   100.0*vrate[ai]/S, dvsty[ai]/S);
            free(dv);
        }
        for (ai=0; ai<NARM; ai++) free(arm[ai]);
        free(rnd);
    }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "validation/nasbench101.txt";
    long n;
    gen_perms();
    tab = calloc(TSIZE, sizeof *tab);
    if (!tab) { fprintf(stderr, "nb101: out of memory\n"); return 1; }
    n = load(path);
    if (n <= 0) { fprintf(stderr, "nb101: cannot read %s\n", path); return 1; }
    printf("loaded %ld lines -> %ld unique canonical architectures\n", n, ntab);

    if (argc > 2 && !strcmp(argv[2], "--selftest"))
        return selftest(path);

    run_experiment();
    return 0;
}
