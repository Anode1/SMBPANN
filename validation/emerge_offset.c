/* emerge_offset.c -- GROUPED (GLOBAL) MUTATION: does the compact filter emerge once mutation acts on
 *                    the shared mechanism, as real genetics does, instead of on one connection?
 *
 * emerge_tie.c found the shared filter stalls at ~12 taps, and Section 6 diagnosed why: once weights
 * are shared, removing ONE connection does not reduce the parameter count (that offset is still used by
 * other units), so per-connection pruning has no energy gradient to tighten the kernel. Section 6 read
 * that as a near-fundamental barrier. It is not: it is an artifact of the MUTATION MODEL. Biological
 * mutation is GLOBAL, not local -- a change to a gene acts on every instance of the mechanism it builds
 * (all "similar blocks and duplicated edges"). For a weight-shared filter the natural unit of mutation
 * is therefore the shared OFFSET (all the edges that reuse that weight), not a single connection.
 *
 * So this probe uses grouped mutation: genome = an offset mask off[NOFF]; connection (j,i) exists iff
 * offset (i-j) is active; a mutation flips a WHOLE OFFSET, adding or removing all its duplicated edges
 * at once. Now removing an offset DOES reduce the parameter count by one, so the energy gradient Section
 * 6 said was missing is present, and the kernel can contract. (Caveat: making the genome an
 * offset mask also makes the connectivity translation-invariant by construction; that is the natural
 * consequence of the feature being shared, but it does bake in more than weight-tying alone.)
 *
 *   (1) OPERATOR at width=0: per-connection mutation (the Section 6 model, reproduces the ~12-tap
 *       plateau) vs grouped per-offset mutation. Does grouped mutation let the tap count fall toward K?
 *   (2) WIDTH PRESSURE under grouped mutation: does the kernel become CONTIGUOUS and land on the
 *       generative offsets (span -> taps -> K, on-rel -> 1)?
 *
 * FINDING (16 seeds): grouped mutation OVERTURNS the Section 6 negative. Taps collapse from 12.7
 * (per-connection) to 2.6 (grouped, ~K); with a width penalty the kernel sharpens to contiguity 0.80 and
 * lands 0.69 on the generative offsets (per-connection: 0.21). Not a perfect K=3 kernel (span ~4, contig
 * ~0.8), but the compact filter LARGELY emerges -- it just needed the biologically-correct global
 * mutation on the shared mechanism, not per-connection pruning. Self-contained C99. Build: make emerge_offset
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
#define NOFF  (N + H)       /* number of offset slots = 22 */
#define RLO   (OFF0)        /* generative (relevant) offsets d=0..K-1  ->  [OFF0, OFF0+K-1] */
#define RHI   (OFF0 + K - 1)

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static double wstar[K];
static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_target = 0.90, g_padd = 0.006, g_prem = 0.030;
static double g_width = 0.0;          /* filter-width pressure: penalize offset SPAN */
static int    g_gens = 150;
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

/* which offsets carry >=1 connection, plus span and how many are generative-relevant */
static void offset_stats(char m[H][N], int *taps, int *span, int *onrel)
{
    char seen[NOFF]; int j,i,lo=NOFF,hi=-1,d=0,rel=0; memset(seen,0,sizeof seen);
    for(j=0;j<H;j++)for(i=0;i<N;i++) if(m[j][i]){ int o=i-j+OFF0;
        if(!seen[o]){seen[o]=1;d++; if(o>=RLO&&o<=RHI) rel++;} if(o<lo)lo=o; if(o>hi)hi=o; }
    *taps=d; *span=(hi>=lo)?(hi-lo+1):0; *onrel=rel;
}
/* free-parameter energy for a shared net = (#taps + width*span)/(H*N) */
static double param_energy(char m[H][N])
{ int d,sp,rel; offset_stats(m,&d,&sp,&rel); return (d + g_width*sp)/(H*N); }

/* build connectivity from an offset mask: (j,i) active iff its offset is active */
static void mask_from_off(char off[NOFF], char m[H][N])
{ int j,i; for(j=0;j<H;j++)for(i=0;i<N;i++){ int o=i-j+OFF0; m[j][i]=(o>=0&&o<NOFF)?off[o]:0; } }

/* train a shared-weight net over mask m; on_test picks val (train) or test set */
static double run_net(char m[H][N], uint32_t seed, int on_test)
{
    static double woff[NOFF], bh[H], v[H];
    double bo=0, h[H]; int i,j,e,s,c=0, ns = on_test?NTE:NVAL;
    double (*Xe)[N] = on_test?Xte:Xval; int *ye = on_test?yte:yval;
    wseed(seed);
    for(i=0;i<NOFF;i++) woff[i]=0.1*wunif();
    for(j=0;j<H;j++){ bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout; double dwoff[NOFF];
        for(j=0;j<H;j++){ double pre=bh[j];
            for(i=0;i<N;i++) if(m[j][i]) pre += woff[i-j+OFF0]*x[i];
            h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        memset(dwoff,0,sizeof dwoff);
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
            for(i=0;i<N;i++) if(m[j][i]) dwoff[i-j+OFF0]+=dpre*x[i]; }
        bo-=LR*dout;
        for(i=0;i<NOFF;i++) woff[i]-=LR*dwoff[i];
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=bh[j]; for(i=0;i<N;i++) if(m[j][i]) pre+=woff[i-j+OFF0]*x[i]; opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
static double objective(char m[H][N], double acc)
{ return (acc>=g_target) ? (2.0 - param_energy(m)) : acc; }

/* out += {taps, span, onrel_frac, test, contiguity} for the best individual, averaged by caller.
 * mode 0 = per-connection mutation (control), mode 1 = coordinated per-offset mutation. */
static void run_ga(int mode, uint32_t seed, double out[5])
{
    static char mpop[POP][H][N], mnxt[POP][H][N];
    static char opop[POP][NOFF], onxt[POP][NOFF];
    double fit[POP], facc[POP]; int idx[POP]; int g,p,q,i,j,o;
    rseed(seed);
    /* dense seed: all connections (mode 0) / all offsets (mode 1) active */
    for(p=0;p<POP;p++){
        if(mode==0){ for(j=0;j<H;j++)for(i=0;i<N;i++) mpop[p][j][i]=1; }
        else       { for(o=0;o<NOFF;o++) opop[p][o]=1; mask_from_off(opop[p],mpop[p]); }
    }
    for(p=0;p<POP;p++){ facc[p]=run_net(mpop[p],(uint32_t)(seed+p*2654435761u+1u),0);
                        fit[p]=objective(mpop[p],facc[p]); }
    for(g=0;g<g_gens;g++){
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++){ memcpy(mnxt[p],mpop[idx[p]],sizeof mnxt[p]); if(mode) memcpy(onxt[p],opop[idx[p]],sizeof onxt[p]); }
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)];
            if(mode==0){
                memcpy(mnxt[p],mpop[a],sizeof mnxt[p]);
                for(j=0;j<H;j++) for(i=0;i<N;i++){
                    if(mnxt[p][j][i]){ if((double)r32()/4294967296.0<g_prem) mnxt[p][j][i]=0; }
                    else             { if((double)r32()/4294967296.0<g_padd) mnxt[p][j][i]=1; } }
            } else {
                memcpy(onxt[p],opop[a],sizeof onxt[p]);
                for(o=1;o<NOFF;o++){   /* per-offset: coordinated add/remove of a whole tap */
                    if(onxt[p][o]){ if((double)r32()/4294967296.0<g_prem) onxt[p][o]=0; }
                    else          { if((double)r32()/4294967296.0<g_padd) onxt[p][o]=1; } }
                mask_from_off(onxt[p],mnxt[p]);
            } }
        memcpy(mpop,mnxt,sizeof mpop); if(mode) memcpy(opop,onxt,sizeof opop);
        for(p=0;p<POP;p++){ facc[p]=run_net(mpop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0);
                            fit[p]=objective(mpop[p],facc[p]); }
    }
    { int bi=0; double bf=-1; for(p=0;p<POP;p++) if(fit[p]>bf){bf=fit[p];bi=p;}
      { int taps,span,rel; offset_stats(mpop[bi],&taps,&span,&rel);
        out[0]+=taps; out[1]+=span; out[2]+= taps?(double)rel/taps:0.0;
        out[3]+=run_net(mpop[bi],seed+999u,1);
        out[4]+= span?(double)taps/span:0.0; } }   /* contiguity: 1.0 = solid window */
}

int main(void)
{
    int seeds=envint("SEEDS",16), sd, r, k;
    double widths[4] = {0.0, 0.5, 1.0, 2.0};
    g_gens=envint("GENS",150); g_target=envdbl("TARGET",0.90);
    printf("COORDINATED OPERATOR: can the COMPACT convolution emerge if mutation acts on whole offsets?\n");
    printf("dense shared seed, N=%d hidden=%d, target acc >= %.2f, %d seeds x %d gens\n", N, H, g_target, seeds, g_gens);
    printf("energy = (#taps + width*span)/(H*N).  compact conv = K=%d contiguous taps on offsets d=0..%d\n\n", K, K-1);

    printf("(1) OPERATOR GRANULARITY at width=0  (per-connection reproduces the ~12-tap plateau)\n");
    printf("  %-22s  taps  span  contig  on-rel  test\n", "operator");
    g_width=0.0;
    { const char *nm[2]={"per-connection","coordinated-offset"}; int mode;
      for(mode=0;mode<2;mode++){ double o[5]={0,0,0,0,0};
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga(mode,(uint32_t)(sd*7+1),o); }
        for(k=0;k<5;k++) o[k]/=seeds;
        printf("  %-22s  %.1f  %.1f   %.2f    %.2f    %.3f\n", nm[mode], o[0], o[1], o[4], o[2], o[3]); } }

    printf("\n(2) WIDTH PRESSURE under the coordinated operator  (does the kernel become CONTIGUOUS?)\n");
    printf("  %-8s  taps  span  contig  on-rel  test\n", "width");
    for(r=0;r<4;r++){ double o[5]={0,0,0,0,0}; g_width=widths[r];
      for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga(1,(uint32_t)(sd*7+1),o); }
      for(k=0;k<5;k++) o[k]/=seeds;
      printf("  %-8.2f  %.1f  %.1f   %.2f    %.2f    %.3f\n", widths[r], o[0], o[1], o[4], o[2], o[3]); }

    printf("\ncompact convolution has emerged when: taps -> K(=%d), span -> K, contig -> 1, on-rel -> 1, test high.\n", K);
    printf("(contig = taps/span; on-rel = fraction of active offsets that are generative, d=0..%d.)\n", K-1);
    return 0;
}
