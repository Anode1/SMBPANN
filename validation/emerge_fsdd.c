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
static double objective(char off[NOFFMAX], double acc)
{ int d,sp,rel; off_stats(off,&d,&sp,&rel); if(d==0) return acc*0.5;
  return (acc>=g_target) ? (2.0 - param_energy(off)) : acc; }

/* collect metrics of the best offset mask into out += {taps,contig,onrel,test}.
 * on-rel (out[2]) is N/A for real data: there is no ground-truth planted filter, so it is fixed at 0. */
static void tally(char best[NOFFMAX], uint32_t seed, double out[])
{ int taps,span,rel; off_stats(best,&taps,&span,&rel);
  out[0]+=taps; out[1]+= span?(double)taps/span:0.0; out[2]+=0.0; /* on-rel N/A for real data */
  out[3]+=run_net(best,seed+999u,1); }

/* evals-to-target: record the (1-based) eval index ec at which a mask is first BOTH accurate and compact
 * (acc>=g_target AND taps<=g_taptgt). *ett stays -1 if the budget is exhausted (right-censored). */
static void ett_check(char off[NOFFMAX], double acc, long ec, double *ett)
{ if(*ett<0.0 && acc>=g_target){ int t,sp,rl; off_stats(off,&t,&sp,&rl); if(t<=g_taptgt) *ett=(double)ec; } }

/* GA arm: directed energy GA with grouped mutation. out[0..3]=metrics, out[4]=evals-to-target (-1 if none). */
static void run_ga(uint32_t seed, double out[])
{
    static char pop[DEF_POP][NOFFMAX], nxt[DEF_POP][NOFFMAX], best[NOFFMAX];
    double fit[DEF_POP], facc[DEF_POP]; int idx[DEF_POP]; int g,p,q,o; long ec=0;
    out[4]=-1.0; rseed(seed);
    for(p=0;p<g_pop;p++) for(o=0;o<g_noff;o++) pop[p][o]=(o!=0);   /* offset 0 is a phantom (no edge maps to it); never activate it, else off_stats pins span's lower bound to 0 and kills the contiguity gradient */
    for(p=0;p<g_pop;p++){ facc[p]=run_net(pop[p],(uint32_t)(seed+p*2654435761u+1u),0); fit[p]=objective(pop[p],facc[p]);
        ec++; ett_check(pop[p],facc[p],ec,&out[4]); }
    for(g=0;g<g_gens;g++){
        for(p=0;p<g_pop;p++) idx[p]=p;
        for(p=0;p<g_pop;p++) for(q=p+1;q<g_pop;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<g_elite;p++) memcpy(nxt[p],pop[idx[p]],sizeof nxt[p]);
        for(p=g_elite;p<g_pop;p++){ int a=idx[(int)(r32()%(uint32_t)g_elite)]; memcpy(nxt[p],pop[a],sizeof nxt[p]);
            for(o=1;o<g_noff;o++) if((double)r32()/4294967296.0<g_flip) nxt[p][o]^=1; }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<g_pop;p++){ facc[p]=run_net(pop[p],(uint32_t)(seed+(uint32_t)(g*g_pop+p)+7u),0); fit[p]=objective(pop[p],facc[p]);
            ec++; ett_check(pop[p],facc[p],ec,&out[4]); }
    }
    { int bi=0; double bf=-1; for(p=0;p<g_pop;p++) if(fit[p]>bf){bf=fit[p];bi=p;} memcpy(best,pop[bi],sizeof best); }
    tally(best,seed,out);
}
/* RANDOM arm: draw random masks at matched total budget, keep the best by the same objective.
   revals>1 gives each mask r independent evaluations (best-of-r objective) so it gets the same
   denoising against the hard accuracy gate that the GA gets by re-evaluating survivors each
   generation; total evaluations stay = evals, so the budget is still matched (evals/r distinct masks). */
static void run_random(uint32_t seed, int evals, int revals, double out[])
{
    static char cand[NOFFMAX], best[NOFFMAX]; double bestfit=-1; int e,o,rr; long ec=0; double bacc;
    int nmask = (revals>1) ? evals/revals : evals;
    out[4]=-1.0; rseed(seed^0x5bd1e995u);
    for(e=0;e<nmask;e++){
        int k = 1 + (int)(r32()%(uint32_t)(g_noff-1));       /* random density: sparse..dense over the real offsets */
        double f=-1;
        for(o=0;o<g_noff;o++) cand[o]=0;                     /* offset 0 phantom, left off (matches the GA) */
        { int placed=0; while(placed<k){ int oo=1+(int)(r32()%(uint32_t)(g_noff-1)); if(!cand[oo]){cand[oo]=1;placed++;} } }
        bacc=-1.0;
        for(rr=0;rr<revals;rr++){                            /* best-of-r: did this mask ever clear the gate / score well */
            double acc=run_net(cand,(uint32_t)(seed+((uint32_t)e*7u+(uint32_t)rr)*2654435761u+3u),0), fo=objective(cand,acc);
            ec++; if(fo>f) f=fo; if(acc>bacc) bacc=acc;
        }
        ett_check(cand,bacc,ec,&out[4]);
        if(f>bestfit){ bestfit=f; memcpy(best,cand,sizeof best); }
    }
    tally(best,seed,out);
}

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

int main(int argc, char**argv)
{
    int sd, k, evals, n1=0, n0=0, i;

    /* defaults <- environment (argv overrides in parse_args) */
    PARAMS(P_LOAD)
    FLAGS(F_LOAD)

    parse_args(argc, argv);   /* argv overrides env; may exit on --help/error; clamps pop/elite */

    evals = g_pop*(g_gens+1);
    set_n(NMAX);              /* g_n=64: whole-clip pool resolution (N is fixed by framing) */

    if(g_agg){   /* read RAW lines from stdin (from parallel chunks), print the aggregated table -- pure C, no external tools */
        static double GA[NMAX+1][4], GQ[NMAX+1][4], RN[NMAX+1][4], RQ[NMAX+1][4];
        static double DC[NMAX+1], DC2[NMAX+1], DO[NMAX+1], DO2[NMAX+1], DT[NMAX+1], DT2[NMAX+1];
        static double EGA[NMAX+1], ERN[NMAX+1]; static int NEG[NMAX+1], NER[NMAX+1];
        static int CNT[NMAX+1], WC[NMAX+1], TC[NMAX+1], LC[NMAX+1], WA[NMAX+1], TA[NMAX+1], LA[NMAX+1];
        int nn, n; double a[4], b[4], ega, ern;
        while(scanf(" RAW %d %*d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &nn,&a[0],&a[1],&a[2],&a[3],&b[0],&b[1],&b[2],&b[3],&ega,&ern)==11){
            if(nn<0||nn>NMAX) continue;
            CNT[nn]++;
            for(k=0;k<4;k++){ GA[nn][k]+=a[k]; GQ[nn][k]+=a[k]*a[k]; RN[nn][k]+=b[k]; RQ[nn][k]+=b[k]*b[k]; }
            { double d=a[1]-b[1]; DC[nn]+=d; DC2[nn]+=d*d; if(d>0.02) WC[nn]++; else if(d<-0.02) LC[nn]++; else TC[nn]++;
              d=a[2]-b[2]; DO[nn]+=d; DO2[nn]+=d*d;
              d=a[3]-b[3]; DT[nn]+=d; DT2[nn]+=d*d; if(d>0.005) WA[nn]++; else if(d<-0.005) LA[nn]++; else TA[nn]++; }
            if(ega>=0){ EGA[nn]+=ega; NEG[nn]++; }
            if(ern>=0){ ERN[nn]+=ern; NER[nn]++; }
        }
        printf("PROVE-FSDD (aggregated): directed energy GA vs RANDOM at matched evals, grouped mutation.\n");
        printf("compact = taps->K(=%d), contig->1. on-rel N/A (real data, no planted filter). paired GA-RANDOM per seed.\n", K);
        printf("evals->target = # evals until a mask is first accurate (>=%.2f) AND compact (taps<=%d).\n\n", g_target, g_taptgt);
        printf("  N   arm     taps          contig        on-rel        test        | paired GA-RANDOM (mean±SEM, win/tie/loss)\n");
        for(n=0;n<=NMAX;n++){ int c=CNT[n]; if(c<1) continue;
            double gm[4],gs[4],rm[4],rsd[4]; for(k=0;k<4;k++){ gm[k]=GA[n][k]/c; rm[k]=RN[n][k]/c;
                gs[k]=c>1?sqrt((GQ[n][k]-GA[n][k]*GA[n][k]/c)/(c-1)):0; rsd[k]=c>1?sqrt((RQ[n][k]-RN[n][k]*RN[n][k]/c)/(c-1)):0; }
            double mdc=DC[n]/c, sdc=c>1?sqrt((DC2[n]-DC[n]*DC[n]/c)/(c-1)):0, semc=sdc/sqrt((double)c);
            double mdo=DO[n]/c, sdo=c>1?sqrt((DO2[n]-DO[n]*DO[n]/c)/(c-1)):0, semo=sdo/sqrt((double)c);
            double mdt=DT[n]/c, sdt=c>1?sqrt((DT2[n]-DT[n]*DT[n]/c)/(c-1)):0, semt=sdt/sqrt((double)c);
            printf("  %-3d GA      %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  | contig %+.2f±%.2f  onrel %+.2f±%.2f  test %+.4f±%.4f  (contig %d/%d/%d, acc %d/%d/%d, n=%d)\n",
                   n, gm[0],gs[0], gm[1],gs[1], gm[2],gs[2], gm[3],gs[3], mdc,semc, mdo,semo, mdt,semt, WC[n],TC[n],LC[n], WA[n],TA[n],LA[n], c);
            printf("  %-3d RANDOM  %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  |\n",
                   n, rm[0],rsd[0], rm[1],rsd[1], rm[2],rsd[2], rm[3],rsd[3]);
            { double eg=NEG[n]?EGA[n]/NEG[n]:-1.0, er=NER[n]?ERN[n]/NER[n]:-1.0;
              printf("  %-3d evals->target: GA %.0f (%d/%d solved)  RANDOM %.0f (%d/%d solved)  ratio RND/GA %.1fx%s\n",
                     n, eg, NEG[n], c, er, NER[n], c, (eg>0&&er>0)?er/eg:0.0, (NER[n]<NEG[n]?"  (random censored: lower bound)":"")); }
        }
        return 0; }

    /* ---- load the two digits ONCE, then run the paired GA-vs-RANDOM comparison over seeds ---- */
    fsdd_load(g_diga, g_digb);
    for(i=0;i<NPOOL;i++){ if(POOLy[i]) n1++; else n0++; }
    if(NPOOL < NTR+NVAL+NTE){
        fprintf(stderr,"FSDD load: got %d clips for digits %d/%d, need >=%d (%d/%d/%d split)\n",
                NPOOL, g_diga, g_digb, NTR+NVAL+NTE, NTR, NVAL, NTE);
        return 1;
    }

    if(!g_raw){
        printf("PROVE-FSDD: real spoken-digit pair (DIGA=%d [label 1] vs DIGB=%d [label 0]), whole-clip mean-pool to N=%d.\n", g_diga, g_digb, g_n);
        printf("directed energy GA vs RANDOM at matched evals (%d), grouped mutation, width=%.1f, target=%.2f.\n", evals, g_width, g_target);
        printf("loaded %d clips (%d pos / %d neg); per-seed shuffle+split %d/%d/%d; %d seeds, %d gens, pop %d.\n",
               NPOOL, n1, n0, NTR, NVAL, NTE, g_seeds, g_gens, g_pop);
        printf("metrics: taps->K(=%d), contig->1, test accuracy. on-rel N/A (no ground-truth filter, reported 0).\n\n", K);
        printf("  N   arm     taps          contig        on-rel        test        | paired GA-RANDOM (mean±SEM, win/tie/loss)\n");
    }

    { double ga[4]={0,0,0,0}, gq[4]={0,0,0,0}, rn[4]={0,0,0,0}, rq[4]={0,0,0,0};
      double dc=0,dc2=0,doo=0,doo2=0,dt=0,dt2=0; int wc=0,tc=0,lc=0,wa=0,ta=0,la=0;
      double sega=0,sern=0; int nsg=0,nsr=0;      /* evals-to-target: sum + solved-count per arm */
      for(sd=g_seed0;sd<g_seed0+g_seeds;sd++){ new_task((uint32_t)(sd*131+1));
          double a[5]={0,0,0,0,0}, b[5]={0,0,0,0,0};
          run_ga((uint32_t)(sd*7+1),a);
          run_random((uint32_t)(sd*7+1),evals,g_revals,b);
          if(g_raw){ printf("RAW %d %d %.0f %.4f %.4f %.4f %.0f %.4f %.4f %.4f %.0f %.0f\n",
                          g_n, sd, a[0],a[1],a[2],a[3], b[0],b[1],b[2],b[3], a[4], b[4]); continue; }
          for(k=0;k<4;k++){ ga[k]+=a[k]; gq[k]+=a[k]*a[k]; }
          for(k=0;k<4;k++){ rn[k]+=b[k]; rq[k]+=b[k]*b[k]; }
          { double d=a[1]-b[1]; dc+=d; dc2+=d*d; if(d>0.02) wc++; else if(d<-0.02) lc++; else tc++;
            d=a[2]-b[2]; doo+=d; doo2+=d*d;
            d=a[3]-b[3]; dt+=d; dt2+=d*d; if(d>0.005) wa++; else if(d<-0.005) la++; else ta++; }
          if(a[4]>=0){ sega+=a[4]; nsg++; }  if(b[4]>=0){ sern+=b[4]; nsr++; }
          if(g_dbg) printf("  [dbg seed=%d GA taps=%.0f contig=%.2f test=%.3f | RND taps=%.0f contig=%.2f test=%.3f]\n",
                           sd, a[0],a[1],a[3], b[0],b[1],b[3]);
      }
      if(!g_raw){
        double gm[4],gs[4],rm[4],rsd[4]; for(k=0;k<4;k++){ gm[k]=ga[k]/g_seeds; rm[k]=rn[k]/g_seeds;
            gs[k]=g_seeds>1?sqrt((gq[k]-ga[k]*ga[k]/g_seeds)/(g_seeds-1)):0; rsd[k]=g_seeds>1?sqrt((rq[k]-rn[k]*rn[k]/g_seeds)/(g_seeds-1)):0; }
        double mdc=dc/g_seeds, sdc=g_seeds>1?sqrt((dc2-dc*dc/g_seeds)/(g_seeds-1)):0, semc=sdc/sqrt((double)g_seeds);
        double mdo=doo/g_seeds, sdo=g_seeds>1?sqrt((doo2-doo*doo/g_seeds)/(g_seeds-1)):0, semo=sdo/sqrt((double)g_seeds);
        double mdt=dt/g_seeds, sdt=g_seeds>1?sqrt((dt2-dt*dt/g_seeds)/(g_seeds-1)):0, semt=sdt/sqrt((double)g_seeds);
        printf("  %-3d GA      %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  | contig %+.2f±%.2f  onrel %+.2f±%.2f  test %+.4f±%.4f  (contig %d/%d/%d, acc %d/%d/%d)\n",
               g_n, gm[0],gs[0], gm[1],gs[1], gm[2],gs[2], gm[3],gs[3], mdc,semc, mdo,semo, mdt,semt, wc,tc,lc, wa,ta,la);
        printf("  %-3d RANDOM  %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  |\n",
               g_n, rm[0],rsd[0], rm[1],rsd[1], rm[2],rsd[2], rm[3],rsd[3]);
        { double eg=nsg?sega/nsg:-1.0, er=nsr?sern/nsr:-1.0;
          printf("  %-3d evals->target: GA %.0f (%d/%d solved)  RANDOM %.0f (%d/%d solved)  ratio RND/GA %.1fx%s\n",
                 g_n, eg, nsg, g_seeds, er, nsr, g_seeds, (eg>0&&er>0)?er/eg:0.0, (nsr<nsg?"  (random censored: lower bound)":"")); }
      }
    }
    if(!g_raw){
        printf("\ndirected search does real work if the paired GA-RANDOM contig/test gap is positive by several SEM.\n");
        printf("emergence transfers to real audio if the GA reaches a compact contiguous filter (taps->K, contig->1)\n");
        printf("at competitive test accuracy where matched-budget random search does not.\n"); }
    return 0;
}
