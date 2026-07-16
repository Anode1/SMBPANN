/* emerge_tie.c -- can the FULL convolution emerge if the genome can share weights?
 *
 * conv_emerge.c showed that energy-constrained pruning discovers sparse, task-relevant
 * connectivity, but NOT convolution's structured topology: it prunes to few connections,
 * yet on a translation-invariant task it does not form aligned local windows, and a
 * connectivity mask cannot express weight sharing at all. Weight sharing is convolution's
 * other half. This probe adds it as a gene and asks whether the whole of convolution then
 * emerges under an energy budget.
 *
 * Genome = a connectivity mask M[H][N] PLUS one bit `shared`. If shared, connection (j,i)
 * uses a weight tied by OFFSET d = i - j (one weight per offset, reused at every position) --
 * a convolution. If not, every connection has its own weight. Crucially, ENERGY is now the
 * number of free PARAMETERS: for a shared net that is the number of distinct offsets used (a
 * local filter of width K costs K), for an unshared net it is the number of connections (a
 * local receptive field per unit costs K*H). So on a task where the same filter works at
 * every position, sharing is far cheaper AND just as accurate -- if the search can find it.
 *
 * Objective: minimize parameter-energy subject to accuracy >= target, from a dense unshared
 * seed, mostly-prune rare-grow mutation (annealing). We add a FILTER-WIDTH pressure: the shared
 * energy also charges g_width * (offset span), so a broad sparse kernel costs more than a
 * contiguous compact one, and we sweep that pressure to ask whether a COMPACT convolution (a
 * tight ~K-tap kernel) can be driven to emerge.
 *
 * Finding: weight sharing emerges (shared-frac ~0.9) and helps generalization, but the compact
 * filter does NOT -- it stays ~12 taps at every width pressure. Once weights are shared, pruning
 * a connection no longer reduces parameter-energy, so the connectivity never tightens; the width
 * penalty only adds cost, and at high pressure sharing reverts to unshared rather than becoming
 * compact. Convolution's pieces emerge separately; the compact kernel resists an energy budget on
 * a general connectivity genome. Self-contained C99.  Build: make emerge_tie
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N   12
#define K   3
#define H   (N - K + 1)     /* = 10 */
#define NTR   64
#define NVAL  300
#define NTE   1000
#define TEPOCHS 50
#define LR    0.1
#define POP   24
#define ELITE 4
#define OFF0  (H)           /* offset index base: d=i-j in [-(H-1),N-1] -> +OFF0 in [1, N-1+H] */
#define NOFF  (N + H)       /* number of offset slots */

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static double wstar[K];
static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_target = 0.90, g_padd = 0.006, g_prem = 0.030, g_pshare = 0.05;
static double g_width = 0.0;          /* filter-width pressure: penalize offset SPAN of a shared filter */
static int    g_gens = 150;
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

/* distinct offsets used (=filter taps) and their span (max-min+1) for a shared filter */
static void offset_extent(const char m[H][N], int *distinct, int *span)
{
    char seen[NOFF]; int j,i,lo=NOFF,hi=-1,d=0; memset(seen,0,sizeof seen);
    for(j=0;j<H;j++)for(i=0;i<N;i++) if(m[j][i]){ int o=i-j+OFF0; if(!seen[o]){seen[o]=1;d++;} if(o<lo)lo=o; if(o>hi)hi=o; }
    *distinct=d; *span=(hi>=lo)?(hi-lo+1):0;
}
/* free-parameter energy, normalized by H*N. For a shared filter, add g_width * offset SPAN so a
 * broad sparse kernel costs more than a contiguous compact one -- the filter-width pressure. */
static double param_energy(const char m[H][N], int shared)
{
    int j,i;
    if (!shared){ long a=0; for(j=0;j<H;j++)for(i=0;i<N;i++) a+=m[j][i]; return (double)a/(H*N); }
    { int d,sp; offset_extent(m,&d,&sp); return (d + g_width*sp)/(H*N); }
}
static double rf_span(const char m[H][N])
{ int j,i,cnt=0; double tot=0; for(j=0;j<H;j++){ int lo=-1,hi=-1; for(i=0;i<N;i++) if(m[j][i]){if(lo<0)lo=i;hi=i;}
      if(lo>=0){tot+=(double)(hi-lo+1)/N; cnt++;} } return cnt?tot/cnt:0.0; }

/* train (shared or independent weights); return val accuracy (train=1) or test acc (train=0) */
static double run_net(const char m[H][N], int shared, uint32_t seed, int on_test)
{
    static double W[H][N], woff[NOFF], bh[H], v[H];
    double bo=0, h[H]; int i,j,e,s,c=0, ns = on_test?NTE:NVAL;
    double (*Xe)[N] = on_test?Xte:Xval; int *ye = on_test?yte:yval;
    wseed(seed);
    for(i=0;i<NOFF;i++) woff[i]=0.1*wunif();
    for(j=0;j<H;j++){ for(i=0;i<N;i++) W[j][i]=m[j][i]?0.1*wunif():0.0; bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout; double dwoff[NOFF];
        for(j=0;j<H;j++){ double pre=bh[j];
            for(i=0;i<N;i++) if(m[j][i]) pre += (shared?woff[i-j+OFF0]:W[j][i])*x[i];
            h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        if(shared) memset(dwoff,0,sizeof dwoff);
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
            for(i=0;i<N;i++) if(m[j][i]){ if(shared) dwoff[i-j+OFF0]+=dpre*x[i]; else W[j][i]-=LR*dpre*x[i]; } }
        bo-=LR*dout;
        if(shared) for(i=0;i<NOFF;i++) woff[i]-=LR*dwoff[i];
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=bh[j]; for(i=0;i<N;i++) if(m[j][i]) pre+=(shared?woff[i-j+OFF0]:W[j][i])*x[i]; opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
static double objective(const char m[H][N], int shared, double acc)
{ return (acc>=g_target) ? (2.0 - param_energy(m,shared)) : acc; }

typedef struct { char m[H][N]; int shared; } Indiv;

/* one GA run. allow_share=0 forces unshared.
 * out = {shared_frac, param_energy, mean_rf_span, best_test, best_filter_span}. */
static void run_ga(int allow_share, uint32_t seed, double out[5])
{
    static Indiv pop[POP], nxt[POP]; double fit[POP], facc[POP]; int idx[POP]; int g,p,q,i,j;
    rseed(seed);
    for(p=0;p<POP;p++){ for(j=0;j<H;j++)for(i=0;i<N;i++) pop[p].m[j][i]=1; pop[p].shared=0; }
    for(p=0;p<POP;p++){ facc[p]=run_net(pop[p].m,pop[p].shared,(uint32_t)(seed+p*2654435761u+1u),0);
                        fit[p]=objective(pop[p].m,pop[p].shared,facc[p]); }
    for(g=0;g<g_gens;g++){
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++) nxt[p]=pop[idx[p]];
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)]; nxt[p]=pop[a];
            for(j=0;j<H;j++) for(i=0;i<N;i++){
                if(nxt[p].m[j][i]){ if((double)r32()/4294967296.0<g_prem) nxt[p].m[j][i]=0; }
                else              { if((double)r32()/4294967296.0<g_padd) nxt[p].m[j][i]=1; } }
            if(allow_share && (double)r32()/4294967296.0<g_pshare) nxt[p].shared^=1; }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++){ facc[p]=run_net(pop[p].m,pop[p].shared,(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0);
                            fit[p]=objective(pop[p].m,pop[p].shared,facc[p]); }
    }
    { double sh=0,en=0,sp=0; int bi=0; double bf=-1;
      for(p=0;p<POP;p++){ sh+=pop[p].shared; en+=param_energy(pop[p].m,pop[p].shared); sp+=rf_span(pop[p].m);
                          if(fit[p]>bf){bf=fit[p];bi=p;} }
      out[0]+=sh/POP; out[1]+=en/POP; out[2]+=sp/POP;
      out[3]+=run_net(pop[bi].m,pop[bi].shared,seed+999u,1);
      { int d,spn; offset_extent(pop[bi].m,&d,&spn);
        out[4]+= pop[bi].shared ? (double)spn : rf_span(pop[bi].m)*N; } }   /* filter width in taps */
}

int main(void)
{
    int seeds=envint("SEEDS",16), sd, r, k;
    double widths[5] = {0.0, 0.25, 0.5, 1.0, 2.0};
    g_gens=envint("GENS",150); g_target=envdbl("TARGET",0.90);
    printf("FILTER-WIDTH PRESSURE: can a COMPACT convolution emerge? (weight-tying + span penalty)\n");
    printf("dense unshared seed, N=%d hidden=%d, target acc >= %.2f, %d seeds x %d gens\n", N, H, g_target, seeds, g_gens);
    printf("shared energy = (#taps + width*offset-span) / (H*N).  compact conv = K=%d taps, span K/N=%.2f\n\n",
           K, (double)K/N);

    printf("  %-16s  shared-frac  param-energy  filter-taps  test\n", "width-pressure");
    { double base[5]={0,0,0,0,0};
      for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga(0,(uint32_t)(sd*7+1),base); }
      for(k=0;k<5;k++) base[k]/=seeds;
      printf("  %-16s   %.3f        %.3f        %.1f(RF)   %.3f\n", "OFF (unshared)", base[0], base[1], base[4], base[3]); }
    for(r=0;r<5;r++){ double o[5]={0,0,0,0,0}; g_width=widths[r];
      for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga(1,(uint32_t)(sd*7+1),o); }
      for(k=0;k<5;k++) o[k]/=seeds;
      printf("  %-16.2f   %.3f        %.3f        %.1f       %.3f\n", widths[r], o[0], o[1], o[4], o[3]); }

    printf("\ncompact convolution has emerged when: shared-frac ~1, filter-taps -> K (=%d), test high.\n", K);
    printf("(filter-taps = the shared kernel's offset extent in taps; RF-span*N for the unshared baseline.)\n");
    return 0;
}
