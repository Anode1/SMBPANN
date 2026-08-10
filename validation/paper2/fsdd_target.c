/* fsdd_target.c -- paper 2: is the compact convolution the OPTIMUM of the efficiency objective
 * on REAL data?  No search. Hand-build candidate filter supports, train each, score each.
 *
 * WHY. Every result in this line so far was measured on a synthetic generator, where nobody
 * independently knows the right answer, so a null could always be blamed on the operator, the seed,
 * the budget or the objective. On real perceptual data at small sample size, the expected answer IS
 * known: the compact convolution should be the efficient one. That makes this a POSITIVE CONTROL for
 * the whole apparatus, and the first check in the line that can come back "your machinery is wrong"
 * rather than "the search did not find it".
 *
 * SUBSTRATE. Inherited unchanged from emerge_fsdd.c: real spoken-digit audio, whole-clip mean-pool to
 * N=64 bins, per-seed 400/100/100 split, weights tied by OFFSET so the net is translation-invariant by
 * construction. The genome is therefore the filter SUPPORT, and "the convolution" means a compact
 * contiguous K-tap support.
 *
 * WHAT IS SCORED. Accuracy and energy are reported separately, plus the continuous objective
 * acc - LAMBDA*energy, because the step objective (acc>=target ? 2-energy : acc) that this substrate
 * inherited is provably blind above the target -- see ../../articles/smbpann2/FINDINGS.md.
 *
 * Build: make fsdd_target        env: SEEDS DIGA DIGB LAMBDA TEPOCHS
 */
/* emerge_fsdd.c -- the PROVE experiment (emerge_prove.c) on REAL audio instead of synthetic tasks.
 *
 * emerge_prove.c asked: under an energy budget, does the DIRECTED energy GA reach the compact, contiguous
 * offset filter where RANDOM search does not?  It answered that on a SYNTHETIC substrate (inputs drawn
 * uniformly, labels from a planted K-tap generative filter `wstar`).  The obvious objection is that the
 * result lives or dies with the synthetic generator.  This fork removes the generator entirely: the inputs
 * are real spoken-digit waveforms from the Free Spoken Digit Dataset (FSDD), and the label is which of two
 * digits was spoken.  Everything about the GA, the shared-weight net, the energy budget, the matched-budget
 * RANDOM arm, and the objective is IDENTICAL to emerge_prove -- only the data changed.
 *
 * Task: a BINARY digit pair (DIGA vs DIGB, from env; default 0 vs 1).  Each ~0.3 s clip is average-pooled
 * over its WHOLE length into N=64 coarse bins (a low-resolution picture of the entire digit), then
 * normalized to zero-mean / unit-max-abs.  600 clips (300 per digit) are loaded once; each seed is a fresh
 * shuffle + 400/100/100 train/val/test split plus a fresh net init, giving the paired-seed variance the
 * GA-vs-RANDOM comparison needs.
 *
 * Two search arms at MATCHED evaluation budget (unchanged from emerge_prove):
 *   GA     : the directed energy GA (selection, elitism, grouped per-offset mutation).
 *   RANDOM : draw the same number of random offset masks, train each, keep the best by the same objective.
 * Metrics: taps (-> K), contiguity (-> 1), test accuracy.  The on-relevance metric is N/A here -- there is
 * no ground-truth planted filter to be "on", so it is reported as 0.  The scaling-over-N sweep of the
 * original is dropped: N is fixed by the framing.
 *
 * Parameterization is unchanged: every knob is a --flag defaulting to its UPPERCASE env var then a
 * compile-time default (SEEDS, GENS, POP, ...).  Data digits: DIGA / DIGB.  Self-contained C99 + POSIX
 * dirent.  Build: cc -std=c99 -pedantic -Wall -Wextra -O2 emerge_fsdd.c -lm -o emerge_fsdd
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dirent.h>
#include <ctype.h>

/* ---- compile-time buffer capacities (memory footprint / task shape; not runtime knobs) ---- */
#define NMAX   64          /* problem size N (whole-clip pool resolution) */
#define K      3           /* filter width used for the compactness targets */
#define HMAX   (NMAX - K + 1)
#define NOFFMAX (NMAX + HMAX)
#define NTR    400         /* train set size */
#define NVAL   100         /* validation set size (search objective) */
#define NTE    100         /* held-out test set size (reported) */
#define POOLMAX 600        /* max clips loaded for the two digits (300 + 300) */

/* ---- compile-time DEFAULTS for the runtime hyperparameters below ---- */
#define DEF_TEPOCHS 50
#define DEF_LR      0.1
#define DEF_POP     24     /* also the population array capacity: runtime --pop must be <= DEF_POP */
#define DEF_ELITE   4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

/* ---- active problem dimensions (set by set_n) ---- */
static int    g_n=64, g_h=62, g_noff=126, g_off0=62, g_rlo=62, g_rhi=64;

/* ================= single source of truth for every runtime parameter =================
 * To add a parameter: add ONE line to PARAMS (scalar/string) or FLAGS (boolean mode). The same
 * table generates the global variable, its env default, its --flag parsing, and its --help row.
 *   PARAMS entry:  X(KIND, var, "flag", "ENV", default, "help")   KIND in {INT, DBL, STR}
 *   FLAGS  entry:  X(var, "flag", "ENV", "help")                  (boolean; presence => true)
 */
#define PARAMS(X) \
  X(INT, g_seeds,   "seeds",   "SEEDS",   40,               "seeds (shuffles/splits)") \
  X(INT, g_seed0,   "seed0",   "SEED0",   1,                "first seed / parallel-chunk offset") \
  X(DBL, g_target,  "target",  "TARGET",  0.90,             "accuracy gate") \
  X(DBL, g_width,   "width",   "WIDTH",   1.0,              "span weight in the energy cost") \
  X(INT, g_revals,  "revals",  "REVALS",  1,                "evals per random mask (denoise)") \
  X(INT, g_taptgt,  "taptgt",  "TAPTGT",  2*K,              "solved tap ceiling for evals->target") \
  X(INT, g_gens,    "gens",    "GENS",    150,              "GA generations") \
  X(INT, g_pop,     "pop",     "POP",     DEF_POP,          "GA population (<= capacity)") \
  X(INT, g_elite,   "elite",   "ELITE",   DEF_ELITE,        "GA elite carryover (< pop)") \
  X(DBL, g_flip,    "flip",    "FLIP",    0.06,             "per-offset mutation rate") \
  X(INT, g_tepochs, "tepochs", "TEPOCHS", DEF_TEPOCHS,      "inner training epochs") \
  X(DBL, g_lr,      "lr",      "LR",      DEF_LR,           "inner learning rate") \
  X(INT, g_diga,    "diga",    "DIGA",    0,                "positive-class digit (label 1)") \
  X(INT, g_digb,    "digb",    "DIGB",    1,                "negative-class digit (label 0)")

#define FLAGS(X) \
  X(g_raw,      "raw",      "RAW",      "one RAW line per seed (feed to --agg)") \
  X(g_agg,      "agg",      "AGG",      "read RAW lines on stdin, print aggregated table") \
  X(g_dbg,      "dbg",      "DBG",      "extra per-seed debug lines")

/* generate global declarations from the tables */
#define P_DECL_INT(v,def) static int    v = (def);
#define P_DECL_DBL(v,def) static double v = (def);
#define P_DECL_STR(v,def) static char   v[128] = def;
#define P_DECL(KIND,v,fl,ev,def,help) P_DECL_##KIND(v,def)
PARAMS(P_DECL)
#define F_DECL(v,fl,ev,help) static int v = 0;
FLAGS(F_DECL)

/* env-default loaders (used in main, before parse_args) */
#define P_LOAD_INT(v,ev,def) v = envint(ev,(def));
#define P_LOAD_DBL(v,ev,def) v = envdbl(ev,(def));
#define P_LOAD_STR(v,ev,def) do{ const char*e_=getenv(ev); if(e_&&*e_){ strncpy(v,e_,sizeof(v)-1); v[sizeof(v)-1]=0; } }while(0);
#define P_LOAD(KIND,v,fl,ev,def,help) P_LOAD_##KIND(v,ev,def)
#define F_LOAD(v,fl,ev,help) v = (getenv(ev)!=NULL);

/* argv matchers (used inside parse_args; `a` = flag name, `val` = its value string) */
#define P_TRY_INT(v,fl) else if(!strcmp(a,fl)){ v = atoi(val); }
#define P_TRY_DBL(v,fl) else if(!strcmp(a,fl)){ v = atof(val); }
#define P_TRY_STR(v,fl) else if(!strcmp(a,fl)){ strncpy(v,val,sizeof(v)-1); v[sizeof(v)-1]=0; }
#define P_TRY(KIND,v,fl,ev,def,help) P_TRY_##KIND(v,fl)
#define F_TRY(v,fl,ev,help) else if(!strcmp(a,fl)){ v = 1; }

/* --help rows (print the effective default, i.e. the variable value) */
#define P_HELP_INT(v,fl,ev,help) printf("    --%-8s %-46s (%s, default %d)\n", fl, help, ev, v);
#define P_HELP_DBL(v,fl,ev,help) printf("    --%-8s %-46s (%s, default %g)\n", fl, help, ev, v);
#define P_HELP_STR(v,fl,ev,help) printf("    --%-8s %-46s (%s, default %s)\n", fl, help, ev, v);
#define P_HELP(KIND,v,fl,ev,def,help) P_HELP_##KIND(v,fl,ev,help)
#define F_HELP(v,fl,ev,help) printf("    --%-8s %-46s (%s)\n", fl, help, ev);

static double Xtr[NTR][NMAX], Xval[NVAL][NMAX], Xte[NTE][NMAX];
static int    ytr[NTR], yval[NVAL], yte[NTE];

/* full loaded pool of the two digits (loaded once; new_task reshuffles/splits it per seed) */
static double POOL[POOLMAX][NMAX];
static int    POOLy[POOLMAX];
static int    NPOOL = 0;

static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static void set_n(int n){ g_n=n; g_h=n-K+1; g_noff=n+g_h; g_off0=g_h; g_rlo=g_off0; g_rhi=g_off0+K-1; }

/* ---------------- FSDD data path ---------------- */

/* Read a MONO 16-bit PCM WAV at `path` into out[] normalized to [-1,1]; returns #samples (<=maxn),
 * or -1 on error. Scans RIFF chunks for "fmt " and "data" instead of assuming a fixed 44-byte header,
 * so files with extra chunks still parse. Reports sample rate / channels / bits via the out-params.
 * (verbatim from fsdd_frame.c -- validated WAV reader) */
static int wav_read(const char *path, double *out, int maxn, int *rate, int *chans, int *bits)
{
    unsigned char hdr[12], ck[8];
    int c = 0, b = 0, sr = 0, n = 0, got_fmt = 0;
    long datasz = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) { fclose(f); return -1; }
    while (fread(ck, 1, 8, f) == 8) {
        uint32_t sz = (uint32_t)ck[4] | (uint32_t)ck[5] << 8 | (uint32_t)ck[6] << 16 | (uint32_t)ck[7] << 24;
        if (!memcmp(ck, "fmt ", 4)) {
            unsigned char fb[16];
            if (sz < 16 || fread(fb, 1, 16, f) != 16) { fclose(f); return -1; }
            c  = (int)(fb[2] | fb[3] << 8);
            sr = (int)((uint32_t)fb[4] | (uint32_t)fb[5] << 8 | (uint32_t)fb[6] << 16 | (uint32_t)fb[7] << 24);
            b  = (int)(fb[14] | fb[15] << 8);
            if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
            got_fmt = 1;
        } else if (!memcmp(ck, "data", 4)) {
            datasz = (long)sz;
            break;                                   /* PCM samples follow immediately */
        } else {
            fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);   /* skip unknown chunk (RIFF pads to even) */
        }
    }
    if (!got_fmt || datasz <= 0 || c != 1 || b != 16) { fclose(f); return -1; }
    { long ns = datasz / 2, i;
      for (i = 0; i < ns && n < maxn; i++) {
          unsigned char s2[2];
          if (fread(s2, 1, 2, f) != 2) break;
          out[n++] = (double)(int16_t)(s2[0] | s2[1] << 8) / 32768.0;
      } }
    fclose(f);
    if (rate)  *rate  = sr;
    if (chans) *chans = c;
    if (bits)  *bits  = b;
    return n;
}

/* Whole-clip mean-pool: average-pool the ENTIRE clip raw[len] into N bins (bin b = mean of the samples in
 * [b*len/N, (b+1)*len/N)), then normalize to zero-mean, unit-max-abs. This is a coarse picture of the whole
 * digit (unlike fsdd_frame's peak-energy window, which crops one loud region). Empty bins (len<N) -> 0. */
static void frame_pool(const double *raw, int len, double *x, int N)
{
    int b, i;
    for (b = 0; b < N; b++) {
        int lo = (int)((long)b * len / N), hi = (int)((long)(b + 1) * len / N);
        double s = 0.0; int c = 0;
        for (i = lo; i < hi; i++) { s += raw[i]; c++; }
        x[b] = c ? s / c : 0.0;
    }
    { double m = 0.0, mx = 1e-9;
      for (b = 0; b < N; b++) m += x[b];
      m /= N;
      for (b = 0; b < N; b++) { x[b] -= m; if (fabs(x[b]) > mx) mx = fabs(x[b]); }
      for (b = 0; b < N; b++) x[b] /= mx; }
}

/* Walk the recordings dir; for every clip whose leading digit == da or db, read + frame_pool into POOL,
 * with label da->1, db->0. Sets NPOOL. Guards against POOL overflow. */
static void fsdd_load(int da, int db)
{
    const char *dir = "/home/vas/smbpann/data/fsdd/recordings";
    double raw[8000];                                 /* 1 s at 8 kHz; clips are ~0.25-0.5 s */
    DIR *d;
    struct dirent *e;
    NPOOL = 0;
    d = opendir(dir);
    if (!d) { fprintf(stderr, "cannot open recordings dir: %s\n", dir); exit(1); }
    while ((e = readdir(d))) {
        const char *nm = e->d_name;
        size_t L = strlen(nm);
        int digit, len;
        char path[1024];
        if (L < 5 || strcmp(nm + L - 4, ".wav") || !isdigit((unsigned char)nm[0])) continue;
        digit = nm[0] - '0';
        if (digit != da && digit != db) continue;
        if (NPOOL >= POOLMAX) { fprintf(stderr, "POOL overflow (>%d clips)\n", POOLMAX); break; }
        snprintf(path, sizeof path, "%s/%s", dir, nm);
        len = wav_read(path, raw, 8000, NULL, NULL, NULL);
        if (len < 0) continue;
        frame_pool(raw, len, POOL[NPOOL], g_n);
        POOLy[NPOOL] = (digit == da) ? 1 : 0;
        NPOOL++;
    }
    closedir(d);
}

/* new_task(seed): a deterministic reshuffle + 400/100/100 split of the loaded POOL (NOT data regeneration).
 * Each seed = a different split + net init, giving the paired-seed variance the GA-vs-RANDOM comparison
 * needs. Requires NPOOL >= NTR+NVAL+NTE (checked in main after load). */
static void new_task(uint32_t seed)
{
    static int perm[POOLMAX];
    int i, s;
    rseed(seed);
    for (i = 0; i < NPOOL; i++) perm[i] = i;
    for (i = NPOOL - 1; i > 0; i--) {                 /* Fisher-Yates */
        int j = (int)(r32() % (uint32_t)(i + 1)), t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (s = 0; s < NTR;  s++) { int p = perm[s];               memcpy(Xtr[s],  POOL[p], sizeof(double) * g_n); ytr[s]  = POOLy[p]; }
    for (s = 0; s < NVAL; s++) { int p = perm[NTR + s];         memcpy(Xval[s], POOL[p], sizeof(double) * g_n); yval[s] = POOLy[p]; }
    for (s = 0; s < NTE;  s++) { int p = perm[NTR + NVAL + s];  memcpy(Xte[s],  POOL[p], sizeof(double) * g_n); yte[s]  = POOLy[p]; }
}

/* ---------------- GA / net (identical machinery to emerge_prove) ---------------- */

static void off_stats(char off[NOFFMAX], int *taps, int *span, int *onrel)
{ int o,d=0,lo=g_noff,hi=-1,rel=0; for(o=0;o<g_noff;o++) if(off[o]){ d++; if(o>=g_rlo&&o<=g_rhi) rel++; if(o<lo)lo=o; if(o>hi)hi=o; }
  *taps=d; *span=(hi>=lo)?(hi-lo+1):0; *onrel=rel; }
static double param_energy(char off[NOFFMAX])
{ int d,sp,rel; off_stats(off,&d,&sp,&rel); return (d + g_width*sp)/(g_h*g_n); }

static double run_net(char off[NOFFMAX], uint32_t seed, int on_test)
{
    static double woff[NOFFMAX], bh[HMAX], v[HMAX];
    double bo=0, h[HMAX]; int i,j,e,s,c=0, ns=on_test?NTE:NVAL;
    double (*Xe)[NMAX]=on_test?Xte:Xval; int *ye=on_test?yte:yval;
    wseed(seed);
    for(i=0;i<g_noff;i++) woff[i]=0.1*wunif();
    for(j=0;j<g_h;j++){ bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<g_tepochs;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout; double dwoff[NOFFMAX];
        for(j=0;j<g_h;j++){ double pre=bh[j];
            for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) pre+=woff[oo]*x[i]; }
            h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(i=0;i<g_noff;i++) dwoff[i]=0;
        for(j=0;j<g_h;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=g_lr*dout*h[j]; bh[j]-=g_lr*dpre;
            for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) dwoff[oo]+=dpre*x[i]; } }
        bo-=g_lr*dout;
        for(i=0;i<g_noff;i++) woff[i]-=g_lr*dwoff[i];
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<g_h;j++){ double pre=bh[j]; for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) pre+=woff[oo]*x[i]; } opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
/* The inherited objective(), tally(), ett_check(), run_ga() and run_random() are deliberately
 * NOT carried over: this probe runs no search. It hand-builds candidate supports and scores
 * them. The step objective in particular is the thing under review -- see FINDINGS.md. */

static void usage(const char*prog)
{
    printf("usage: %s [options]   (every option also settable via its UPPERCASE env var)\n", prog);
    printf("  parameters:\n");
    PARAMS(P_HELP)
    printf("  modes:\n");
    FLAGS(F_HELP)
    printf("  compile-time capacities (rebuild to change): NMAX=%d K=%d NTR=%d NVAL=%d NTE=%d POPcap=%d\n",
           NMAX,K,NTR,NVAL,NTE,DEF_POP);
}

/* argv overrides env-loaded defaults. Accepts --flag, --key value, and --key=value. */
static void parse_args(int argc, char**argv)
{
    int i;
    for(i=1;i<argc;i++){
        char *a=argv[i], *eq; const char *val=NULL;
        if(strncmp(a,"--",2)!=0){ fprintf(stderr,"ignoring non-option arg: %s\n", a); continue; }
        a+=2; eq=strchr(a,'='); if(eq){ *eq=0; val=eq+1; }
        if(!strcmp(a,"help")||!strcmp(a,"h")){ usage(argv[0]); exit(0); }
        else if(0){ /* anchor for FLAGS chain */ }
        FLAGS(F_TRY)
        else {
            if(!val){ if(i+1<argc) val=argv[++i]; else { fprintf(stderr,"missing value for --%s\n", a); exit(2); } }
            if(0){ /* anchor for PARAMS chain */ }
            PARAMS(P_TRY)
            else { fprintf(stderr,"unknown option: --%s\n", a); exit(2); }
        }
    }
    if(g_pop>DEF_POP) g_pop=DEF_POP;
    if(g_pop<2) g_pop=2;
    if(g_elite>=g_pop) g_elite=g_pop-1;
    if(g_elite<1) g_elite=1;
}


/* ---- the candidate supports, hand-built ---- */
#define NCAND 7
static const char *cand_name[NCAND] = {
    "conv-K (compact)", "conv-1 (too narrow)", "conv-5", "conv-9", "conv-21",
    "scattered-K", "full support"
};
static void build_cand(char off[NOFFMAX], int c)
{
    int o, lo, hi, w;
    for(o=0;o<g_noff;o++) off[o]=0;
    switch(c){
      case 0: w=K;  break;
      case 1: w=1;  break;
      case 2: w=5;  break;
      case 3: w=9;  break;
      case 4: w=21; break;
      case 5:                                  /* K taps, deliberately non-contiguous */
        off[g_off0]=1;
        if(g_off0-10 > 0)        off[g_off0-10]=1;
        if(g_off0+10 < g_noff)   off[g_off0+10]=1;
        return;
      default:                                 /* every offset except the phantom at 0 */
        for(o=1;o<g_noff;o++) off[o]=1;
        return;
    }
    lo = g_off0 - (w-K)/2; hi = lo + w - 1;    /* centred on the natural K-tap band */
    if(lo<1) lo=1;
    if(hi>=g_noff) hi=g_noff-1;
    for(o=lo;o<=hi;o++) off[o]=1;
}

int main(int argc, char**argv)
{
    static char off[NOFFMAX];
    double acc[NCAND], en[NCAND], tst[NCAND];
    double lambda; int sd, c, n1=0, n0=0, i, best=0;

    PARAMS(P_LOAD)
    FLAGS(F_LOAD)
    parse_args(argc, argv);
    lambda = getenv("LAMBDA") ? atof(getenv("LAMBDA")) : 1.0;
    set_n(NMAX);

    fsdd_load(g_diga, g_digb);
    for(i=0;i<NPOOL;i++){ if(POOLy[i]) n1++; else n0++; }
    if(NPOOL < NTR+NVAL+NTE){
        fprintf(stderr,"FSDD load: got %d clips for digits %d/%d, need >=%d\n",
                NPOOL, g_diga, g_digb, NTR+NVAL+NTE);
        return 1;
    }

    printf("FSDD TARGET CHECK -- no search. digits %d vs %d, N=%d, %d clips (%d/%d), %d seeds,\n",
           g_diga, g_digb, g_n, NPOOL, n1, n0, g_seeds);
    printf("per-seed 400/100/100 split, %d train epochs. objective = acc - %.2f*energy.\n\n",
           g_tepochs, lambda);

    for(c=0;c<NCAND;c++){ acc[c]=0; en[c]=0; tst[c]=0; }
    for(sd=1; sd<=g_seeds; sd++){
        new_task((uint32_t)(sd*911u+1u));
        for(c=0;c<NCAND;c++){
            build_cand(off, c);
            acc[c] += run_net(off,(uint32_t)(sd*7u+c*131u+1u),0);   /* validation: what selection sees */
            tst[c] += run_net(off,(uint32_t)(sd*7u+c*131u+1u),1);   /* held out                        */
            en[c]  += param_energy(off);
        }
    }

    printf("  %-20s %6s %8s %9s %9s %11s\n", "support", "taps", "contig", "val acc", "test acc", "objective");
    for(c=0;c<NCAND;c++){
        int taps,span,rel; double obj;
        build_cand(off,c); off_stats(off,&taps,&span,&rel);
        acc[c]/=g_seeds; tst[c]/=g_seeds; en[c]/=g_seeds;
        obj = acc[c] - lambda*en[c];
        printf("  %-20s %6d %8.2f %9.3f %9.3f %11.4f\n", cand_name[c], taps,
               span?(double)taps/span:0.0, acc[c], tst[c], obj);
        if(obj > acc[best]-lambda*en[best]) best=c;
    }
    printf("\nbest by objective: %s\n", cand_name[best]);
    printf("If the compact convolution wins, the apparatus is calibrated and the SYNTHETIC task was the\n");
    printf("weak link. If it loses on real data too, the objective or the net is miscalibrated -- that is\n");
    printf("a bug to find, not a result to report.\n");
    return 0;
}
