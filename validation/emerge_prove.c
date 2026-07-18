/* emerge_prove.c -- does the DIRECTED energy GA reach the compact filter where RANDOM search does not?
 *
 * The grouped-mutation result (emerge_offset.c) showed the compact filter emerges under an energy budget.
 * Two questions remain to PROVE the mechanism rather than illustrate it: (1) is the emergence due to the
 * DIRECTED search, or would random sampling of offset masks find a compact filter just as well? The whole
 * project is haunted by random search being a strong baseline (the crossover study lost to it), so this is
 * the sharp test. (2) Does it hold as the problem SCALES (larger N, more offsets to prune)?
 *
 * Same setup as emerge_offset: shared-weight net, genome = offset mask, grouped per-offset mutation,
 * energy = (#taps + width*span)/(H*N), objective = minimize energy subject to accuracy >= target. Two
 * search arms at MATCHED evaluation budget:
 *   GA     : the directed energy GA (selection, elitism, grouped mutation, annealing).
 *   RANDOM : draw the same number of random offset masks (each with a random number of active offsets,
 *            so sparse and dense are both sampled), train each, keep the best by the same objective.
 * Reports taps / contiguity / on-relevant / test, mean +/- SD over many seeds, at two problem sizes.
 * The compact filter is "proven emergent" if GA reaches it (taps -> K, contiguity high, on-rel high) and
 * beats random by more than the noise; scaling holds if that survives the larger N. Self-contained C99.
 * Build: make emerge_prove
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NMAX   32
#define K      3
#define HMAX   (NMAX - K + 1)
#define NOFFMAX (NMAX + HMAX)
#define MAXN   8           /* max number of problem sizes in a sweep */
#define NTR    64
#define NVAL   300
#define NTE    1000
#define TEPOCHS 50
#define LR     0.1
#define POP    24
#define ELITE  4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int    g_n=12, g_h=10, g_noff=22, g_off0=10, g_rlo=10, g_rhi=12;   /* runtime dims */
static double wstar[K];
static double Xtr[NTR][NMAX], Xval[NVAL][NMAX], Xte[NTE][NMAX];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_target=0.90, g_width=1.0, g_flip=0.06;
static int    g_gens=150;
static int    g_trace=0;   /* TRACE mode: accumulate per-generation population offset-activation frequencies */
#define TRACEG 1024
static double g_tracebuf[TRACEG][NOFFMAX];   /* [gen][offset] summed activation freq across trace runs */
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static void set_n(int n){ g_n=n; g_h=n-K+1; g_noff=n+g_h; g_off0=g_h; g_rlo=g_off0; g_rhi=g_off0+K-1; }

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=g_n;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][NMAX],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<g_n;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

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
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout; double dwoff[NOFFMAX];
        for(j=0;j<g_h;j++){ double pre=bh[j];
            for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) pre+=woff[oo]*x[i]; }
            h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        for(i=0;i<g_noff;i++) dwoff[i]=0;
        for(j=0;j<g_h;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
            for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) dwoff[oo]+=dpre*x[i]; } }
        bo-=LR*dout;
        for(i=0;i<g_noff;i++) woff[i]-=LR*dwoff[i];
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<g_h;j++){ double pre=bh[j]; for(i=0;i<g_n;i++){ int oo=i-j+g_off0; if(oo>=0&&oo<g_noff&&off[oo]) pre+=woff[oo]*x[i]; } opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
static double objective(char off[NOFFMAX], double acc)
{ int d,sp,rel; off_stats(off,&d,&sp,&rel); if(d==0) return acc*0.5;
  return (acc>=g_target) ? (2.0 - param_energy(off)) : acc; }

/* collect metrics of the best offset mask into out += {taps,contig,onrel,test} */
static void tally(char best[NOFFMAX], uint32_t seed, double out[4])
{ int taps,span,rel; off_stats(best,&taps,&span,&rel);
  out[0]+=taps; out[1]+= span?(double)taps/span:0.0; out[2]+= taps?(double)rel/taps:0.0;
  out[3]+=run_net(best,seed+999u,1); }

/* GA arm: directed energy GA with grouped mutation. */
static void run_ga(uint32_t seed, double out[4])
{
    static char pop[POP][NOFFMAX], nxt[POP][NOFFMAX], best[NOFFMAX];
    double fit[POP], facc[POP]; int idx[POP]; int g,p,q,o;
    rseed(seed);
    for(p=0;p<POP;p++) for(o=0;o<g_noff;o++) pop[p][o]=(o!=0);   /* offset 0 is a phantom (no edge maps to it); never activate it, else off_stats pins span's lower bound to 0 and kills the contiguity gradient */
    for(p=0;p<POP;p++){ facc[p]=run_net(pop[p],(uint32_t)(seed+p*2654435761u+1u),0); fit[p]=objective(pop[p],facc[p]); }
    for(g=0;g<g_gens;g++){
        if(g_trace && g<TRACEG){ int oo,pp;   /* accumulate population offset-activation freq entering gen g (g=0 = dense seed) */
            for(oo=0;oo<g_noff;oo++){ int cnt=0; for(pp=0;pp<POP;pp++) cnt+=pop[pp][oo]; g_tracebuf[g][oo]+=(double)cnt/POP; } }
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++) memcpy(nxt[p],pop[idx[p]],sizeof nxt[p]);
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)]; memcpy(nxt[p],pop[a],sizeof nxt[p]);
            for(o=1;o<g_noff;o++) if((double)r32()/4294967296.0<g_flip) nxt[p][o]^=1; }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++){ facc[p]=run_net(pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0); fit[p]=objective(pop[p],facc[p]); }
    }
    { int bi=0; double bf=-1; for(p=0;p<POP;p++) if(fit[p]>bf){bf=fit[p];bi=p;} memcpy(best,pop[bi],sizeof best); }
    tally(best,seed,out);
}
/* RANDOM arm: draw random masks at matched total budget, keep the best by the same objective.
   revals>1 gives each mask r independent evaluations (best-of-r objective) so it gets the same
   denoising against the hard accuracy gate that the GA gets by re-evaluating survivors each
   generation; total evaluations stay = evals, so the budget is still matched (evals/r distinct masks). */
static void run_random(uint32_t seed, int evals, int revals, double out[4])
{
    static char cand[NOFFMAX], best[NOFFMAX]; double bestfit=-1; int e,o,rr;
    int nmask = (revals>1) ? evals/revals : evals;
    rseed(seed^0x5bd1e995u);
    for(e=0;e<nmask;e++){
        int k = 1 + (int)(r32()%(uint32_t)(g_noff-1));       /* random density: sparse..dense over the real offsets */
        double f=-1;
        for(o=0;o<g_noff;o++) cand[o]=0;                     /* offset 0 phantom, left off (matches the GA) */
        { int placed=0; while(placed<k){ int oo=1+(int)(r32()%(uint32_t)(g_noff-1)); if(!cand[oo]){cand[oo]=1;placed++;} } }
        for(rr=0;rr<revals;rr++){                            /* best-of-r: did this mask ever clear the gate / score well */
            double acc=run_net(cand,(uint32_t)(seed+((uint32_t)e*7u+(uint32_t)rr)*2654435761u+3u),0), fo=objective(cand,acc);
            if(fo>f) f=fo;
        }
        if(f>bestfit){ bestfit=f; memcpy(best,cand,sizeof best); }
    }
    tally(best,seed,out);
}

int main(void)
{
    int seeds=envint("SEEDS",40), sd, ni, k, evals;
    int seed0=envint("SEED0",1), raw=getenv("RAW")!=NULL;   /* SEED0 offsets the chunk; RAW emits one line per seed for parallel aggregation */
    int revals=envint("REVALS",1);                          /* r evaluations per random mask (denoising confound control); 1 = original */
    int Ns[MAXN], nN=0;
    g_gens=envint("GENS",150); g_target=envdbl("TARGET",0.90); g_width=envdbl("WIDTH",1.0);
    evals = POP*(g_gens+1);
    if(getenv("PROBE")){ char t[NOFFMAX]; int q; set_n(12); new_task(132u);
        for(q=0;q<NOFFMAX;q++) t[q]=0;
        t[10]=1; t[11]=1; t[12]=1;
        printf("PROBE prove: acc(off 10,11,12)=%.4f  acc2=%.4f\n", run_net(t,42u,0), run_net(t,7u,0)); return 0; }
    if(getenv("TRACE")){   /* GA runs over SEEDS seeds; emit the MEAN population offset-activation per generation */
        int n0=envint("N",12), ts, gg, oo, ng=(g_gens<TRACEG?g_gens:TRACEG);
        set_n(n0); g_trace=1;
        for(gg=0;gg<TRACEG;gg++) for(oo=0;oo<NOFFMAX;oo++) g_tracebuf[gg][oo]=0;
        for(ts=seed0;ts<seed0+seeds;ts++){ double out[4]={0,0,0,0};
            new_task((uint32_t)(ts*131+1)); run_ga((uint32_t)(ts*7+1), out); }
        printf("# TRACE N=%d noff=%d off0=%d gen_offsets=%d..%d gens=%d seeds=%d (mean activation)\n",
               g_n,g_noff,g_off0,g_rlo,g_rhi,ng,seeds);
        for(gg=0;gg<ng;gg++){ printf("TRACE %d", gg);
            for(oo=0;oo<g_noff;oo++) printf(" %.4f", g_tracebuf[gg][oo]/seeds);
            printf("\n"); }
        return 0; }
    if(getenv("BOUNDARY")){   /* phase boundary: N* (largest N whose GA still lands on the generative offsets) vs budget */
        int Bl[8], nb=0, Nl[MAXN], nNl=0, bi2, sd2; double thr=envdbl("NSTAR",0.40);
        { const char*e=getenv("BUDGETS"); char buf[128]; strncpy(buf,e&&*e?e:"50,150,400",127); buf[127]=0;
          { char*p=strtok(buf,","); while(p&&nb<8){ int v=atoi(p); if(v>0) Bl[nb++]=v; p=strtok(NULL,","); } } }
        { const char*e=getenv("NLIST"); char buf[128]; strncpy(buf,e&&*e?e:"12,16,20,24,28,32",127); buf[127]=0;
          { char*p=strtok(buf,","); while(p&&nNl<MAXN){ int v=atoi(p); if(v>=K+2&&v<=NMAX) Nl[nNl++]=v; p=strtok(NULL,","); } } }
        printf("PHASE BOUNDARY: does emergence (GA landing on the K=%d generative offsets) survive to larger N\n", K);
        printf("as the search budget grows?  order parameter = GA on-relevance; N* = largest N with on-rel >= %.2f.\n", thr);
        printf("%d seeds/cell, width=%.1f, target=%.2f.\n\n", seeds, g_width, g_target);
        printf("  budget(gens,evals) |");
        for(ni=0;ni<nNl;ni++) printf("  N=%d", Nl[ni]);
        printf("   | N*\n");
        for(bi2=0;bi2<nb;bi2++){ int nstar=-1; g_gens=Bl[bi2]; evals=POP*(g_gens+1);
            printf("  gens=%-4d ev=%-6d |", g_gens, evals);
            for(ni=0;ni<nNl;ni++){ set_n(Nl[ni]); double onrel=0;
                for(sd2=seed0;sd2<seed0+seeds;sd2++){ double a[4]={0,0,0,0}; new_task((uint32_t)(sd2*131+1));
                    run_ga((uint32_t)(sd2*7+1),a); onrel+=a[2]; }
                onrel/=seeds; printf("  %.2f", onrel); if(onrel>=thr) nstar=Nl[ni]; }
            if(nstar>0) printf("   | %d\n", nstar); else printf("   | -\n"); }
        printf("\nphase boundary confirmed if N* GROWS with budget: directed emergence has an upper critical size\n");
        printf("that a larger search budget pushes outward (more evaluations reach the generative offsets at larger N).\n");
        return 0; }
    if(getenv("AGG")){   /* read RAW lines from stdin (from parallel chunks), print the aggregated scaling table -- pure C, no external tools */
        static double GA[NMAX+1][4], GQ[NMAX+1][4], RN[NMAX+1][4], RQ[NMAX+1][4];
        static double DC[NMAX+1], DC2[NMAX+1], DO[NMAX+1], DO2[NMAX+1];
        static int CNT[NMAX+1], WC[NMAX+1], TC[NMAX+1], LC[NMAX+1];
        int nn, ss, n; double a[4], b[4];
        while(scanf(" RAW %d %d %lf %lf %lf %lf %lf %lf %lf %lf", &nn,&ss,&a[0],&a[1],&a[2],&a[3],&b[0],&b[1],&b[2],&b[3])==10){
            if(nn<0||nn>NMAX) continue;
            CNT[nn]++;
            for(k=0;k<4;k++){ GA[nn][k]+=a[k]; GQ[nn][k]+=a[k]*a[k]; RN[nn][k]+=b[k]; RQ[nn][k]+=b[k]*b[k]; }
            { double d=a[1]-b[1]; DC[nn]+=d; DC2[nn]+=d*d; if(d>0.02) WC[nn]++; else if(d<-0.02) LC[nn]++; else TC[nn]++;
              d=a[2]-b[2]; DO[nn]+=d; DO2[nn]+=d*d; }
        }
        printf("PROVE (scaling, aggregated): directed energy GA vs RANDOM at matched evals, grouped mutation.\n");
        printf("compact = taps->K(=%d), contig->1, on-rel->1. paired GA-RANDOM per seed (same task).\n\n", K);
        printf("  N   arm     taps          contig        on-rel        test        | paired GA-RANDOM (mean±SEM, win/tie/loss)\n");
        for(n=0;n<=NMAX;n++){ int c=CNT[n]; if(c<1) continue;
            double gm[4],gs[4],rm[4],rsd[4]; for(k=0;k<4;k++){ gm[k]=GA[n][k]/c; rm[k]=RN[n][k]/c;
                gs[k]=c>1?sqrt((GQ[n][k]-GA[n][k]*GA[n][k]/c)/(c-1)):0; rsd[k]=c>1?sqrt((RQ[n][k]-RN[n][k]*RN[n][k]/c)/(c-1)):0; }
            double mdc=DC[n]/c, sdc=c>1?sqrt((DC2[n]-DC[n]*DC[n]/c)/(c-1)):0, semc=sdc/sqrt((double)c);
            double mdo=DO[n]/c, sdo=c>1?sqrt((DO2[n]-DO[n]*DO[n]/c)/(c-1)):0, semo=sdo/sqrt((double)c);
            printf("  %-3d GA      %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  | contig %+.2f±%.2f  onrel %+.2f±%.2f  (%d/%d/%d, n=%d)\n",
                   n, gm[0],gs[0], gm[1],gs[1], gm[2],gs[2], gm[3],gs[3], mdc,semc, mdo,semo, WC[n],TC[n],LC[n], c);
            printf("  %-3d RANDOM  %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  |\n",
                   n, rm[0],rsd[0], rm[1],rsd[1], rm[2],rsd[2], rm[3],rsd[3]); }
        return 0; }
    { const char *e=getenv("NLIST"); if(e&&*e){ char buf[128]; strncpy(buf,e,127); buf[127]=0;
          char *p=strtok(buf,","); while(p&&nN<MAXN){ int v=atoi(p); if(v>=K+2&&v<=NMAX) Ns[nN++]=v; p=strtok(NULL,","); } }
      if(nN==0){ int def[]={12,16,20,24,28}; for(k=0;k<5;k++) Ns[nN++]=def[k]; } }
    if(!raw){
    printf("PROVE (scaling): directed energy GA vs RANDOM search at matched evals (%d), grouped mutation, width=%.1f\n", evals, g_width);
    printf("local K=%d generative filter, %d seeds, %d gens. paired per seed (same task). compact = taps->K(=%d), contig->1, on-rel->1\n\n", K, seeds, g_gens, K);
    printf("  N   arm     taps          contig        on-rel        test        | paired GA-RANDOM (mean±SEM, win/tie/loss)\n"); }
    for(ni=0;ni<nN;ni++){ set_n(Ns[ni]);
        double ga[4]={0,0,0,0}, gq[4]={0,0,0,0}, rn[4]={0,0,0,0}, rq[4]={0,0,0,0};
        double dc=0,dc2=0,doo=0,doo2=0; int wc=0,tc=0,lc=0;     /* paired contig diff + win/tie/loss */
        for(sd=seed0;sd<seed0+seeds;sd++){ new_task((uint32_t)(sd*131+1));
            double a[4]={0,0,0,0}, b[4]={0,0,0,0};
            run_ga((uint32_t)(sd*7+1),a);
            run_random((uint32_t)(sd*7+1),evals,revals,b);
            if(raw){ printf("RAW %d %d %.0f %.4f %.4f %.4f %.0f %.4f %.4f %.4f\n",
                            Ns[ni], sd, a[0],a[1],a[2],a[3], b[0],b[1],b[2],b[3]); continue; }
            for(k=0;k<4;k++){ ga[k]+=a[k]; gq[k]+=a[k]*a[k]; }
            for(k=0;k<4;k++){ rn[k]+=b[k]; rq[k]+=b[k]*b[k]; }
            { double d=a[1]-b[1]; dc+=d; dc2+=d*d; if(d>0.02) wc++; else if(d<-0.02) lc++; else tc++;
              d=a[2]-b[2]; doo+=d; doo2+=d*d; }
            if(getenv("DBG")) printf("  [dbg N=%d seed=%d GA c=%.2f o=%.2f | RND c=%.2f o=%.2f]\n", Ns[ni], sd, a[1],a[2], b[1],b[2]);
        }
        if(raw) continue;
        { double gm[4],gs[4],rm[4],rsd[4]; for(k=0;k<4;k++){ gm[k]=ga[k]/seeds; rm[k]=rn[k]/seeds;
            gs[k]=seeds>1?sqrt((gq[k]-ga[k]*ga[k]/seeds)/(seeds-1)):0; rsd[k]=seeds>1?sqrt((rq[k]-rn[k]*rn[k]/seeds)/(seeds-1)):0; }
          double mdc=dc/seeds, sdc=seeds>1?sqrt((dc2-dc*dc/seeds)/(seeds-1)):0, semc=sdc/sqrt((double)seeds);
          double mdo=doo/seeds, sdo=seeds>1?sqrt((doo2-doo*doo/seeds)/(seeds-1)):0, semo=sdo/sqrt((double)seeds);
          printf("  %-3d GA      %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  | contig %+.2f±%.2f  onrel %+.2f±%.2f  (%d/%d/%d)\n",
                 Ns[ni], gm[0],gs[0], gm[1],gs[1], gm[2],gs[2], gm[3],gs[3], mdc,semc, mdo,semo, wc,tc,lc);
          printf("  %-3d RANDOM  %.1f±%.1f     %.2f±%.2f    %.2f±%.2f    %.3f±%.3f  |\n",
                 Ns[ni], rm[0],rsd[0], rm[1],rsd[1], rm[2],rsd[2], rm[3],rsd[3]); }
    }
    if(!raw){
    printf("\ndirected search does real work if the paired GA-RANDOM contig/on-rel gap is positive by several SEM,\n");
    printf("and the scaling claim holds if that gap GROWS with N (tie at small N, widening advantage as the space grows).\n"); }
    return 0;
}
