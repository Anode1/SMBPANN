/* pairstat.c -- paired statistics with NAMED tests, for the per-seed output of the emerge_* probes.
 *
 * WHY THIS FILE EXISTS. The paper reports paired GA-vs-RANDOM differences as mean +- SEM plus
 * win/tie/loss counts, and names no statistical test anywhere. submission_plan.md and
 * revision_notes_gpem.md both predict that as the most likely referee request. The paired seeds are
 * already archived and emerge_prove already has a --raw mode, so the test is a computation over
 * existing data, not a new experiment.
 *
 * No Python in this repo, so this is C99 like everything else, and it is gated by a self-test
 * against hand-computable textbook cases (--selftest), the same smallest-case-first discipline as
 * tests.c. A tool that reports p-values must itself be checked against values you can derive by
 * hand, or it is one more apparatus artifact waiting to happen.
 *
 * WHAT IT COMPUTES, per group and per metric:
 *   Wilcoxon signed-rank  PRIMARY. Exact null by dynamic programming when no |d| tie forces average
 *                         ranks; otherwise the tie-corrected normal approximation with a continuity
 *                         correction. The output always says WHICH was used. Zero differences are
 *                         dropped (Wilcoxon's own convention) and the dropped count is reported,
 *                         because dropping them changes n and a reader must see that.
 *   sign test             exact binomial. Assumes only symmetry of sign under the null, so it
 *                         survives any doubt about the signed-rank's symmetry assumption.
 *   Hodges-Lehmann        median of WALSH AVERAGES (d_i+d_j)/2 over i<=j -- NOT the median of the
 *                         differences -- with the distribution-free rank-based confidence interval.
 *   paired t              SECONDARY, via the incomplete beta. Never a normal standing in for a t.
 *   Holm                  across the whole declared family in a single invocation, so the family is
 *                         what was actually declared rather than what survived.
 *   TOST                  equivalence against a margin, when MARGIN is set. Only meaningful if the
 *                         margin was fixed BEFORE seeing these numbers (PROTOCOL.md item 8).
 *   MDE                   minimum detectable effect at the given alpha and power, so that a null
 *                         result is bounded rather than merely asserted.
 *
 * Three cautions from PROTOCOL.md are honoured here by construction:
 *   - an approximation is never quoted without being labelled one;
 *   - Hodges-Lehmann is the median of Walsh averages;
 *   - the paired t uses a t distribution, not a normal approximation to it.
 *
 * Build: make pairstat
 * Use:   cat scratch_prove_raw_*.out | GROUP=2 METRICS=contig:5:9,onrel:6:10,acc:7:11 ./pairstat
 *        (emerge_prove --raw emits: RAW N seed ga0 ga1 ga2 ga3 rnd0 rnd1 rnd2 rnd3 gaEv rndEv
 *         so column 2 is N, 5/9 are contiguity, 6/10 on-relevance, 7/11 test accuracy.)
 * Env:   PREFIX GROUP METRICS ALPHA POWER MARGIN ZERO SELFTEST
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXPAIRS   512               /* per group and metric; exact-null cap lives here too */
#define WMAX       ((MAXPAIRS*(MAXPAIRS+1))/2)
#define MAXGROUPS  16
#define MAXMETRICS 8
#define MAXCOLS    32
#define LINEMAX    1024

static double g_dp[WMAX+1];          /* exact signed-rank null, as probabilities */
static double g_walsh[WMAX+1];       /* Walsh averages for Hodges-Lehmann */

static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static int cmp_dbl(const void*a,const void*b)
{
    double x=*(const double*)a, y=*(const double*)b;
    return (x>y)-(x<y);
}

/* ---------------------------------------------------------------- distributions */

/* upper tail of the standard normal; erfc is accurate far into the tail, unlike 1-Phi(z) */
static double norm_sf(double z){ return 0.5*erfc(z/sqrt(2.0)); }

/* continued fraction for the incomplete beta (Lentz); the engine behind the t distribution */
static double betacf(double a,double b,double x)
{
    const double TINY=1e-300;
    double qab=a+b, qap=a+1.0, qam=a-1.0;
    double c=1.0, d=1.0-qab*x/qap, h;
    int m;
    if(fabs(d)<TINY) d=TINY;
    d=1.0/d; h=d;
    for(m=1;m<=300;m++){
        int m2=2*m;
        double aa=m*(b-m)*x/((qam+m2)*(a+m2)), del;
        d=1.0+aa*d; if(fabs(d)<TINY) d=TINY;
        c=1.0+aa/c; if(fabs(c)<TINY) c=TINY;
        d=1.0/d; h*=d*c;
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
        d=1.0+aa*d; if(fabs(d)<TINY) d=TINY;
        c=1.0+aa/c; if(fabs(c)<TINY) c=TINY;
        d=1.0/d; del=d*c; h*=del;
        if(fabs(del-1.0)<3e-16) break;
    }
    return h;
}

static double betai(double a,double b,double x)
{
    double bt;
    if(x<=0.0) return 0.0;
    if(x>=1.0) return 1.0;
    bt = exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log(1.0-x));
    if(x < (a+1.0)/(a+b+2.0)) return bt*betacf(a,b,x)/a;
    return 1.0-bt*betacf(b,a,1.0-x)/b;
}

/* two-sided p for Student's t; df real so this stays honest for small samples */
static double t_sf2(double t,double df)
{
    if(df<=0.0) return 1.0;
    return betai(0.5*df,0.5,df/(df+t*t));
}

/* one-sided quantile: the t with upper-tail area p (p<0.5), by bisection on the CDF */
static double t_quantile(double p,double df)
{
    double lo=0.0, hi=200.0, mid;
    int i;
    for(i=0;i<200;i++){
        mid=0.5*(lo+hi);
        if(0.5*t_sf2(mid,df) > p) lo=mid; else hi=mid;
    }
    return 0.5*(lo+hi);
}

/* exact two-sided binomial sign test at q=0.5 */
static double sign_p(int pos,int n)
{
    double lo=0.0, hi=0.0, ln2=log(0.5)*n;
    int i, k = pos<n-pos ? pos : n-pos;
    if(n<=0) return 1.0;
    for(i=0;i<=k;i++)
        lo += exp(lgamma(n+1.0)-lgamma(i+1.0)-lgamma(n-i+1.0)+ln2);
    hi = lo;
    { double p = lo+hi; return p>1.0?1.0:p; }
}

/* inverse of the regularised incomplete beta, by bisection: needed for Clopper-Pearson bounds */
static double betainv(double p,double a,double b)
{
    double lo=0.0, hi=1.0, mid;
    int i;
    for(i=0;i<200;i++){
        mid=0.5*(lo+hi);
        if(betai(a,b,mid) < p) lo=mid; else hi=mid;
    }
    return 0.5*(lo+hi);
}

/* exact signed-rank null over ranks 1..n, in place, as a probability distribution */
static void wsr_null(int n)
{
    int r,w,maxw=n*(n+1)/2;
    for(w=0;w<=maxw;w++) g_dp[w]=0.0;
    g_dp[0]=1.0;
    for(r=1;r<=n;r++){
        int cur=r*(r+1)/2;
        for(w=cur;w>=r;w--)  g_dp[w]=0.5*(g_dp[w]+g_dp[w-r]);
        for(w=r-1;w>=0;w--)  g_dp[w]=0.5*g_dp[w];
    }
}

/* ---------------------------------------------------------------- one paired comparison */

typedef struct {
    char   name[32];
    int    grp;
    int    npairs;                /* all paired seeds: the estimator's n */
    int    n, nzero, win, tie, loss;  /* n = non-zero pairs: the rank test's n */
    double mean, sd, sem;
    double hl, hl_lo, hl_hi;
    double p_wsr, p_sign, p_t, p_holm;
    double mde, p_tost;
    int    exact;                 /* 1 = exact signed-rank null, 0 = tie-corrected normal */
    int    binary;                /* 1 = paired 0/1 outcomes: McNemar replaces Wilcoxon */
    int    b, c, nconc;           /* McNemar: A-only wins, B-only wins, concordant pairs */
    double pi, pi_lo, pi_hi;      /* b/(b+c) with an exact Clopper-Pearson interval */
    double pdiff;                 /* marginal proportion difference (b-c)/n */
} Result;

/* Paired BINARY outcomes (solved / not solved per seed). Wilcoxon is the wrong test here: every
   |d| is 1, so the signed-rank reduces to the sign test anyway but arrives via a tie-corrected
   approximation. McNemar's exact test IS the sign test on the discordant pairs, so we name it
   correctly and report the discordant counts a referee will want to see. */
static void analyse_binary(const double *d,int n,double alpha,Result *R)
{
    int i;
    R->binary=1; R->n=n; R->npairs=n; R->b=0; R->c=0; R->nconc=0; R->nzero=0;
    for(i=0;i<n;i++){
        if(d[i]>0.5)       R->b++;
        else if(d[i]<-0.5) R->c++;
        else             { R->nconc++; R->nzero++; }
    }
    R->win=R->b; R->loss=R->c; R->tie=R->nconc;
    R->mean  = n? (double)(R->b-R->c)/n : 0.0;
    R->pdiff = R->mean;
    R->p_wsr = sign_p(R->b, R->b+R->c);      /* McNemar exact; the primary for binary data */
    R->p_sign= R->p_wsr;
    R->p_t   = -1.0;
    R->exact = 1;
    R->sd=R->sem=0.0; R->mde=0.0; R->p_tost=-1.0;
    { int m=R->b+R->c;
      R->pi    = m? (double)R->b/m : 0.0;
      R->pi_lo = (m&&R->b>0)   ? betainv(alpha/2.0,(double)R->b,(double)(m-R->b+1)) : 0.0;
      R->pi_hi = (m&&R->b<m)   ? betainv(1.0-alpha/2.0,(double)(R->b+1),(double)(m-R->b)) : 1.0;
      R->hl=R->pi; R->hl_lo=R->pi_lo; R->hl_hi=R->pi_hi; }
}

/* Zero differences are dropped by the Wilcoxon TEST (its own convention) but must NOT be dropped
   from the ESTIMATOR: the Hodges-Lehmann location and the mean describe the distribution of the
   difference, and an exact zero is a real observation of it. Dropping them from the estimator
   inflates the effect -- at N=12 here it turns a +0.02 contiguity gap over 200 seeds into +0.04
   over the 112 non-zero ones, which is a different and larger claim. So: estimator over ALL pairs,
   rank test over the non-zero ones, and both counts reported. */
static void analyse(const double *d,int n0,double alpha,double power,double margin,int drop_zero,Result *R)
{
    static double all[MAXPAIRS];
    static double nz[MAXPAIRS];
    static double ad[MAXPAIRS];
    static int    idx[MAXPAIRS];
    static double rank[MAXPAIRS];
    double wplus=0.0, mu, var, z, tiesum=0.0, s=0.0, s2=0.0;
    int n=0, i, j, m, ties=0;

    R->nzero=0; R->win=0; R->tie=0; R->loss=0; R->binary=0;
    R->b=R->c=R->nconc=0; R->pi=R->pi_lo=R->pi_hi=R->pdiff=0.0;
    R->npairs=n0;

    for(i=0;i<n0;i++){
        all[i]=d[i];
        if(d[i]==0.0){ R->nzero++; R->tie++; if(!drop_zero) nz[n++]=d[i]; continue; }
        if(d[i]>0.0) R->win++; else R->loss++;
        nz[n++]=d[i];
    }
    R->n=n;
    if(n<2||n0<2){ R->p_wsr=R->p_sign=R->p_t=R->p_holm=1.0; R->mean=R->sd=R->sem=0.0;
                   R->hl=R->hl_lo=R->hl_hi=0.0; R->mde=0.0; R->p_tost=1.0; R->exact=0; return; }

    /* mean/sd/sem over ALL pairs, which is what the paper's tables report */
    for(i=0;i<n0;i++){ s+=all[i]; s2+=all[i]*all[i]; }
    R->mean=s/n0;
    R->sd  = sqrt((s2-s*s/n0)/(n0-1));
    R->sem = R->sd/sqrt((double)n0);

    /* average ranks over |d| of the NON-ZERO differences, recording tie-group sizes */
    for(i=0;i<n;i++){ ad[i]=fabs(nz[i]); idx[i]=i; }
    for(i=0;i<n;i++)                              /* insertion sort of idx by ad: n<=512 */
        for(j=i+1;j<n;j++)
            if(ad[idx[j]]<ad[idx[i]]){ int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    for(i=0;i<n;){
        double avg;
        for(j=i; j+1<n && ad[idx[j+1]]==ad[idx[i]]; j++) ;
        m=j-i+1;
        avg=0.5*((i+1)+(j+1));
        for(; i<=j; i++) rank[idx[i]]=avg;
        if(m>1){ ties=1; tiesum += (double)m*m*m - (double)m; }
    }
    for(i=0;i<n;i++) if(nz[i]>0.0) wplus+=rank[i];

    mu  = 0.25*n*(n+1.0);
    var = n*(n+1.0)*(2.0*n+1.0)/24.0 - tiesum/48.0;

    if(!ties && n<=MAXPAIRS){
        /* exact: two-sided p = 2 * min(lower tail, upper tail), capped at 1 */
        int maxw=n*(n+1)/2, w=(int)(wplus+0.5);
        double lo=0.0, hi=0.0, p;
        wsr_null(n);
        for(i=0;i<=w && i<=maxw;i++) lo+=g_dp[i];
        for(i=w;i<=maxw;i++)         hi+=g_dp[i];
        p=2.0*(lo<hi?lo:hi);
        R->p_wsr = p>1.0?1.0:p;
        R->exact = 1;
    }else{
        double c = wplus>mu ? -0.5 : 0.5;         /* continuity correction toward the mean */
        z = var>0.0 ? (wplus-mu+c)/sqrt(var) : 0.0;
        R->p_wsr = 2.0*norm_sf(fabs(z));
        if(R->p_wsr>1.0) R->p_wsr=1.0;
        R->exact = 0;
    }

    R->p_sign = sign_p(R->win, R->win+R->loss);
    /* the paired t uses ALL pairs, so df = n0-1. sd == 0 means every difference is identical: the
       t statistic is undefined, not infinite, so flag it rather than dividing by an epsilon. */
    R->p_t    = (R->sd>0.0) ? t_sf2(R->mean/R->sem,(double)(n0-1)) : -1.0;

    /* Hodges-Lehmann over ALL n0 pairs (zeros included: they are real observations of the
       difference), with the rank-based interval computed for that same n0. */
    {
        int M=0, k;
        double tie_all=0.0, mu_all, var_all;
        for(i=0;i<n0;i++) for(j=i;j<n0;j++) g_walsh[M++]=0.5*(all[i]+all[j]);
        qsort(g_walsh,(size_t)M,sizeof(double),cmp_dbl);
        R->hl = (M%2) ? g_walsh[M/2] : 0.5*(g_walsh[M/2-1]+g_walsh[M/2]);

        /* tie-group sizes among |all|, for the interval's variance correction */
        for(i=0;i<n0;i++) ad[i]=fabs(all[i]);
        qsort(ad,(size_t)n0,sizeof(double),cmp_dbl);
        for(i=0;i<n0;){
            for(j=i; j+1<n0 && ad[j+1]==ad[i]; j++) ;
            m=j-i+1;
            if(m>1) tie_all += (double)m*m*m - (double)m;
            i=j+1;
        }
        mu_all  = 0.25*n0*(n0+1.0);
        var_all = n0*(n0+1.0)*(2.0*n0+1.0)/24.0 - tie_all/48.0;

        if(tie_all==0.0 && n0<=MAXPAIRS){
            double cum=0.0;
            wsr_null(n0);
            for(k=0;k<=M && cum+g_dp[k]<=alpha/2.0;k++) cum+=g_dp[k];
            if(k<1) k=1;
            R->hl_lo=g_walsh[k-1];
            R->hl_hi=g_walsh[M-k];
        }else{
            double zc=1.959963985, kk = mu_all - zc*sqrt(var_all>0.0?var_all:0.0);
            int k2=(int)floor(kk);
            if(k2<1) k2=1;
            if(k2>M) k2=M;
            R->hl_lo=g_walsh[k2-1];
            R->hl_hi=g_walsh[M-k2];
        }
    }

    /* MDE over ALL pairs: the effect this design could have detected at (alpha, power) */
    {
        double ta=t_quantile(alpha/2.0,(double)(n0-1));
        double tb=t_quantile(1.0-power,(double)(n0-1));
        R->mde=(ta+tb)*R->sd/sqrt((double)n0);
    }

    /* TOST: equivalent iff both one-sided tests reject at alpha */
    if(margin>0.0 && R->sem>0.0){
        double t1=(R->mean+margin)/R->sem, t2=(R->mean-margin)/R->sem;
        double p1=0.5*t_sf2(t1,(double)(n0-1)), p2=0.5*t_sf2(t2,(double)(n0-1));
        if(t1<0.0) p1=1.0-p1;
        if(t2>0.0) p2=1.0-p2;
        R->p_tost = p1>p2?p1:p2;
    }else R->p_tost=-1.0;
}

/* ---------------------------------------------------------------- self-test */

/* Format a p-value honestly. PROTOCOL.md: never quote an approximation far into a tail -- a
   tie-corrected normal is trustworthy near alpha and meaningless at 1e-30, so it is floored and
   shown as an inequality. Exact values are floored only at the limit of double precision. */
static void fmt_p(char*buf,size_t sz,double p,int exact)
{
    if(p<0.0)            { snprintf(buf,sz,"n/a");      return; }
    if(p<=0.0)           { snprintf(buf,sz,"<1e-300");  return; }
    if(!exact && p<1e-12){ snprintf(buf,sz,"<1e-12~");  return; }
    snprintf(buf,sz,"%.3g%s",p,exact?"":"~");
}

static int approx(double got,double want,double tol,const char*what)
{
    int ok = fabs(got-want)<=tol;
    printf("  %-46s got %-12.6g want %-12.6g %s\n", what, got, want, ok?"PASS":"** FAIL **");
    return ok;
}

static int selftest(void)
{
    Result R;
    int ok=1;
    printf("pairstat self-test (values derivable by hand)\n\n");

    /* n=5, all positive, ranks 1..5: W+=15 is the unique maximum of 2^5 sign assignments,
       so the two-sided exact p is 2*(1/32) = 0.0625. The sign test agrees at 2*(1/32). */
    { double d[5]={1,2,3,4,5};
      analyse(d,5,0.05,0.80,0.0,1,&R);
      ok &= approx(R.p_wsr,0.0625,1e-9,"exact Wilcoxon, d={1,2,3,4,5}");
      ok &= approx(R.p_sign,0.0625,1e-9,"exact sign test, 5/5 positive");
      ok &= approx(R.hl,3.0,1e-9,"Hodges-Lehmann (median of Walsh avgs)");
      ok &= approx(R.mean,3.0,1e-9,"mean difference");
      ok &= approx(R.p_t,0.013228,1e-5,"paired t, t=4.2426 df=4");
      ok &= (R.exact==1); }

    /* n=5 with one negative: W+ = 2+3+4+5 = 14. Subsets of {1..5} summing >=14 are {all}=15
       and {2,3,4,5}=14, so the upper tail is 2/32 and the two-sided p is 4/32 = 0.125.
       The 15 Walsh averages sort to -1,0.5,1,1.5,2,2,2.5,3,3,3.5,3.5,4,4,4.5,5 and the
       8th is 3 -- which is NOT the median of the differences (that is 3 as well here only
       by coincidence of this example; the two estimators differ in general). */
    { double d[5]={-1,2,3,4,5};
      analyse(d,5,0.05,0.80,0.0,1,&R);
      ok &= approx(R.p_wsr,0.125,1e-9,"exact Wilcoxon, d={-1,2,3,4,5}");
      ok &= approx(R.hl,3.0,1e-9,"Hodges-Lehmann with one negative"); }

    /* n=6, all positive: 2*(1/64) = 0.03125 */
    { double d[6]={1,2,3,4,5,6};
      analyse(d,6,0.05,0.80,0.0,1,&R);
      ok &= approx(R.p_wsr,0.03125,1e-9,"exact Wilcoxon, n=6 all positive"); }

    /* zeros are dropped and n falls from 7 to 5, reproducing the first case exactly */
    { double d[7]={1,2,0,3,4,0,5};
      analyse(d,7,0.05,0.80,0.0,1,&R);
      ok &= approx(R.p_wsr,0.0625,1e-9,"zeros dropped: n 7 -> 5, p unchanged");
      ok &= (R.n==5 && R.nzero==2); }

    /* ties in |d| force the approximation, and the tool must SAY so rather than claim exactness */
    { double d[6]={2,2,2,-2,3,4};
      analyse(d,6,0.05,0.80,0.0,1,&R);
      printf("  %-46s exact=%d (expect 0: |d| ties force the approximation)\n","tie handling",R.exact);
      ok &= (R.exact==0); }

    /* McNemar exact. b=10, c=2: only the 12 discordant pairs count, so the two-sided p is
       2*P(X<=2) for X~Bin(12,1/2) = 2*(1+12+66)/4096 = 158/4096 = 0.0385742. */
    { double d[14]; int i;
      for(i=0;i<10;i++) d[i]=1.0;      /* A solved, B did not */
      for(i=10;i<12;i++) d[i]=-1.0;    /* B solved, A did not */
      d[12]=0.0; d[13]=0.0;            /* concordant: carry no information */
      analyse_binary(d,14,0.05,&R);
      ok &= approx(R.p_wsr,0.0385742,1e-6,"McNemar exact, b=10 c=2 (+2 concordant)");
      ok &= (R.b==10 && R.c==2 && R.nconc==2);
      ok &= approx(R.pdiff,(10.0-2.0)/14.0,1e-9,"marginal solve-rate difference (b-c)/n"); }

    /* Clopper-Pearson sanity: b=m means the lower bound is alpha^(1/m) and the upper is 1 */
    { double d[5]; int i;
      for(i=0;i<5;i++) d[i]=1.0;
      analyse_binary(d,5,0.05,&R);
      ok &= approx(R.p_wsr,0.0625,1e-9,"McNemar exact, b=5 c=0");
      ok &= approx(R.pi_lo,pow(0.025,1.0/5.0),1e-6,"Clopper-Pearson lower at b=m=5"); }

    /* the incomplete beta against a known t: t=2.776445 at df=4 is the 5%% two-sided point */
    ok &= approx(t_sf2(2.776445,4.0),0.05,1e-6,"t_sf2(2.776445, df=4)");
    ok &= approx(t_sf2(1.959964,1e7),0.05,1e-4,"t_sf2(1.959964, df->inf) ~ normal");

    printf("\n%s\n", ok?"self-test PASSED":"self-test FAILED");
    return ok?0:1;
}

/* ---------------------------------------------------------------- main */

int main(int argc,char**argv)
{
    static double diff[MAXGROUPS][MAXMETRICS][MAXPAIRS];
    static int    ndiff[MAXGROUPS][MAXMETRICS];
    static double gkey[MAXGROUPS];
    static Result res[MAXGROUPS*MAXMETRICS];
    static char   mname[MAXMETRICS][32];
    static int    mca[MAXMETRICS], mcb[MAXMETRICS];
    char   line[LINEMAX], mspec[256], prefix[32]="RAW";
    int    ngroups=0, nmetrics=0, nres=0, gcol, i, g, m, over=0;
    double alpha, power, margin;
    int    drop_zero, binary;

    for(i=1;i<argc;i++)
        if(!strcmp(argv[i],"--selftest")) return selftest();
        else if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){
            printf("usage: <per-seed lines> | ./pairstat        (env-configured)\n");
            printf("  PREFIX   only use lines whose first token is this (default RAW; empty = all)\n");
            printf("  GROUP    1-based column to group by, 0 = one group (default 2)\n");
            printf("  METRICS  name:colA:colB[,...]  1-based columns of the paired arms\n");
            printf("  ALPHA    significance level (default 0.05)\n");
            printf("  POWER    power used for the MDE (default 0.8)\n");
            printf("  MARGIN   TOST equivalence margin, 0 = skip (default 0)\n");
            printf("  ZERO     1 = drop zero differences (Wilcoxon, default), 0 = keep them\n");
            printf("  --selftest   check the tests against hand-computable cases and exit\n");
            return 0;
        }

    alpha  = envdbl("ALPHA",0.05);
    power  = envdbl("POWER",0.80);
    margin = envdbl("MARGIN",0.0);
    drop_zero = envint("ZERO",1);
    binary = envint("BINARY",0);
    gcol   = envint("GROUP",2);
    { const char*e=getenv("PREFIX"); if(e){ strncpy(prefix,e,sizeof(prefix)-1); prefix[sizeof(prefix)-1]=0; } }
    { const char*e=getenv("METRICS");
      strncpy(mspec, e&&*e ? e : "contig:5:9,onrel:6:10,acc:7:11", sizeof(mspec)-1);
      mspec[sizeof(mspec)-1]=0; }

    { char *tok=strtok(mspec,",");
      while(tok && nmetrics<MAXMETRICS){
          char *c1=strchr(tok,':'), *c2;
          if(!c1){ fprintf(stderr,"pairstat: bad METRICS entry '%s'\n",tok); return 2; }
          *c1=0; c2=strchr(c1+1,':');
          if(!c2){ fprintf(stderr,"pairstat: bad METRICS entry '%s'\n",tok); return 2; }
          *c2=0;
          { size_t L=strlen(tok); if(L>sizeof(mname[0])-1) L=sizeof(mname[0])-1;
            memcpy(mname[nmetrics],tok,L); mname[nmetrics][L]=0; }
          mca[nmetrics]=atoi(c1+1); mcb[nmetrics]=atoi(c2+1);
          nmetrics++;
          tok=strtok(NULL,",");
      } }

    for(g=0;g<MAXGROUPS;g++) for(m=0;m<MAXMETRICS;m++) ndiff[g][m]=0;

    while(fgets(line,sizeof(line),stdin)){
        char *tokv[MAXCOLS]; int ntok=0;
        char *p=strtok(line," \t\n\r");
        while(p&&ntok<MAXCOLS){ tokv[ntok++]=p; p=strtok(NULL," \t\n\r"); }
        if(ntok==0) continue;
        if(prefix[0] && strcmp(tokv[0],prefix)) continue;
        { double key = (gcol>=1&&gcol<=ntok) ? atof(tokv[gcol-1]) : 0.0;
          g=-1;
          for(i=0;i<ngroups;i++) if(gkey[i]==key){ g=i; break; }
          if(g<0){ if(ngroups>=MAXGROUPS) continue; g=ngroups++; gkey[g]=key; }
          for(m=0;m<nmetrics;m++){
              if(mca[m]<1||mca[m]>ntok||mcb[m]<1||mcb[m]>ntok) continue;
              if(ndiff[g][m]>=MAXPAIRS){ over=1; continue; }
              diff[g][m][ndiff[g][m]++] = atof(tokv[mca[m]-1]) - atof(tokv[mcb[m]-1]);
          } }
    }

    if(ngroups==0){ fprintf(stderr,"pairstat: no input lines matched PREFIX='%s'\n",prefix); return 2; }
    if(over) fprintf(stderr,"pairstat: WARNING more than %d pairs in a cell; extra pairs IGNORED\n",MAXPAIRS);

    /* sort groups by key so the table reads in order */
    for(i=0;i<ngroups;i++) for(g=i+1;g<ngroups;g++)
        if(gkey[g]<gkey[i]){
            double tk=gkey[i]; gkey[i]=gkey[g]; gkey[g]=tk;
            for(m=0;m<nmetrics;m++){
                double tmp[MAXPAIRS]; int tn;
                memcpy(tmp,diff[i][m],sizeof(tmp));
                memcpy(diff[i][m],diff[g][m],sizeof(tmp));
                memcpy(diff[g][m],tmp,sizeof(tmp));
                tn=ndiff[i][m]; ndiff[i][m]=ndiff[g][m]; ndiff[g][m]=tn;
            } }

    for(g=0;g<ngroups;g++)
        for(m=0;m<nmetrics;m++){
            if(ndiff[g][m]<2) continue;
            if(binary) analyse_binary(diff[g][m],ndiff[g][m],alpha,&res[nres]);
            else       analyse(diff[g][m],ndiff[g][m],alpha,power,margin,drop_zero,&res[nres]);
            memcpy(res[nres].name,mname[m],sizeof(res[nres].name));
            res[nres].name[sizeof(res[nres].name)-1]=0;
            res[nres].grp=(int)gkey[g];
            nres++;
        }

    /* Holm across the DECLARED family: every test computed in this invocation */
    { int order[MAXGROUPS*MAXMETRICS];
      double run=0.0;
      for(i=0;i<nres;i++) order[i]=i;
      for(i=0;i<nres;i++) for(g=i+1;g<nres;g++)
          if(res[order[g]].p_wsr<res[order[i]].p_wsr){ int t=order[i]; order[i]=order[g]; order[g]=t; }
      for(i=0;i<nres;i++){
          double adj=(nres-i)*res[order[i]].p_wsr;
          if(adj>1.0) adj=1.0;
          if(adj<run) adj=run; else run=adj;      /* enforce monotonicity */
          res[order[i]].p_holm=adj;
      } }

    if(binary){
        printf("pairstat: paired BINARY outcomes, alpha=%.3g, family size %d (Holm)\n", alpha, nres);
        printf("primary = McNemar exact (the sign test on discordant pairs). Wilcoxon is not used:\n");
        printf("every |d| is 1, so it would reduce to this test but arrive via an approximation.\n");
        printf("pi = b/(b+c), the share of DISCORDANT seeds favouring arm A, with an exact\n");
        printf("Clopper-Pearson interval. diff = (b-c)/n is the marginal difference in solve rate.\n\n");
        printf("%-4s %-8s %5s %5s %5s %6s %9s %24s %12s %10s\n",
               "grp","metric","n","b(A)","c(B)","conc","diff","pi [95% CI]","p(McNemar)","p(Holm)");
        for(i=0;i<nres;i++){
            Result *R=&res[i];
            char ci[40], pw[24], ph[24];
            snprintf(ci,sizeof(ci),"%.3f [%.3f,%.3f]",R->pi,R->pi_lo,R->pi_hi);
            fmt_p(pw,sizeof(pw),R->p_wsr,1);
            fmt_p(ph,sizeof(ph),R->p_holm,1);
            printf("%-4d %-8s %5d %5d %5d %6d %+9.4f %24s %12s %10s\n",
                   R->grp,R->name,R->n,R->b,R->c,R->nconc,R->pdiff,ci,pw,ph);
        }
        printf("\n  b = seeds where arm A solved and B did not; c = the reverse; conc = both or neither.\n");
        printf("  Only the b+c discordant seeds carry information: that is McNemar's whole point.\n");
        return 0;
    }
    printf("pairstat: paired tests, alpha=%.3g, power=%.2f, family size %d (Holm)\n", alpha, power, nres);
    printf("primary = Wilcoxon signed-rank; sign test exact; HL = median of Walsh averages; t secondary.\n");
    printf("zero differences %s.\n\n", drop_zero?"DROPPED (Wilcoxon convention)":"kept");
    printf("n = all paired seeds (the estimator's n); ntest = non-zero pairs the rank test uses.\n");
    printf("mean and HL are over ALL pairs; zeros are real observations of the difference.\n\n");
    printf("%-4s %-8s %5s %6s %5s %11s %9s %19s %11s %10s %10s %10s\n",
           "grp","metric","n","ntest","zero","w/t/l","mean","HL [95% CI]","p(Wilcox)","p(Holm)","p(sign)","p(t)");
    for(i=0;i<nres;i++){
        Result *R=&res[i];
        char ci[40], pw[24], ph[16], ps[16], pt[16];
        if(R->n<2){
            /* every difference was an exact zero: no test was run, and saying so beats printing 1 */
            printf("%-4d %-8s %5d %6d %5d %5d/%d/%-4d %+9.5f %19s %11s %10s %10s %10s\n",
                   R->grp,R->name,R->npairs,R->n,R->nzero,R->win,R->tie,R->loss,
                   R->mean,"(no non-zero diffs)","n/a","n/a","n/a","n/a");
            continue;
        }
        snprintf(ci,sizeof(ci),"%+.4f [%+.4f,%+.4f]",R->hl,R->hl_lo,R->hl_hi);
        fmt_p(pw,sizeof(pw),R->p_wsr,R->exact);
        fmt_p(ph,sizeof(ph),R->p_holm,R->exact);
        fmt_p(ps,sizeof(ps),R->p_sign,1);       /* sign test is exact binomial */
        fmt_p(pt,sizeof(pt),R->p_t,1);          /* t via incomplete beta, not an approximation to it */
        printf("%-4d %-8s %5d %6d %5d %5d/%d/%-4d %+9.5f %19s %11s %10s %10s %10s\n",
               R->grp,R->name,R->npairs,R->n,R->nzero,R->win,R->tie,R->loss,
               R->mean,ci,pw,ph,ps,pt);
    }
    printf("\n  ~ = tie-corrected normal approximation (|d| ties made the exact null invalid);\n");
    printf("    no suffix = EXACT signed-rank null by dynamic programming.\n");
    printf("  w/t/l counts ties as exact zero differences, which the test then drops.\n\n");
    printf("%-4s %-8s %12s %12s %s\n","grp","metric","MDE","SEM","note");
    for(i=0;i<nres;i++){
        Result *R=&res[i];
        printf("%-4d %-8s %12.5f %12.5f %s\n", R->grp,R->name,R->mde,R->sem,
               (R->p_wsr>alpha) ? "null: effects below MDE were undetectable at this n" : "");
    }
    if(margin>0.0){
        printf("\nTOST against margin %+.4f (only valid if the margin was fixed BEFORE these runs):\n",margin);
        for(i=0;i<nres;i++)
            printf("  %-4d %-8s p=%.3g  %s\n",res[i].grp,res[i].name,res[i].p_tost,
                   res[i].p_tost<alpha?"EQUIVALENT":"not shown equivalent");
    }
    return 0;
}
