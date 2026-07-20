/* emerge_local.c -- locality IMPOSED (necessary prior); does the convolution emerge ON TOP?
 *
 * The fair experiment. emerge_tie.c showed weight-sharing is adopted but on a general genome the
 * filter stays broad (~12 taps); emerge_offset.c would tighten it only by hardwiring translation
 * invariance -- which over-imposes (it installs the very abstraction we want to watch emerge).
 *
 * Here we impose ONLY locality, which is a necessary prior (a bounded receptive field is a law of
 * the problem, not something to wait out): each hidden unit sees a CONTIGUOUS window [start, start+w)
 * of the inputs. That is the one imposed abstraction layer. Everything above it is left free and must
 * emerge under an energy (free-parameter) budget:
 *   - the window WIDTH w_j (compactness) -- free per unit, seeded wide, annealed;
 *   - WEIGHT SHARING across positions (a `shared` gene, offered not imposed): if on, weights are
 *     tied by within-window offset and reused by every unit -- i.e. TRANSLATION INVARIANCE, the
 *     convolution operator itself. Energy for shared = the number of distinct tied weights (the
 *     kernel width); for unshared = the total connection count. So sharing is cheap exactly when the
 *     units settle on a common compact window -- a convolution -- and expensive otherwise.
 *
 * Two abstraction layers, one imposed (local field) and one that must emerge (shared compact filter).
 * The question: given the necessary locality prior, does an energy budget assemble the rest of
 * the convolution -- adopt sharing, shrink to a compact kernel, and tile the input -- or not?
 *
 * Reports, per arm: shared-frac, mean/max width, coverage (do the windows tile all N inputs?),
 * energy, test. "Convolution emerged on top of locality" = shared-frac ~1, width -> K, coverage ~1.
 * Self-contained C99.  Build: make emerge_local
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

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static double rprob(void){return (double)r32()/4294967296.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static double wstar[K];
static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_target = 0.90, g_pgrow = 0.10, g_pshare = 0.05;
static int    g_gens = 150;
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

typedef struct { int start[H], w[H]; int shared; } Indiv;

static int maxwidth(const Indiv*g){ int j,m=0; for(j=0;j<H;j++) if(g->w[j]>m) m=g->w[j]; return m; }
/* free-parameter energy, normalized by H*N. shared => #distinct tied weights (=max width);
 * unshared => total connections (sum of widths). */
static double param_energy(const Indiv*g)
{ int j; if(g->shared) return (double)maxwidth(g)/(H*N);
  { long a=0; for(j=0;j<H;j++) a+=g->w[j]; return (double)a/(H*N); } }
static double meanwidth(const Indiv*g){ int j; double s=0; for(j=0;j<H;j++) s+=g->w[j]; return s/H; }
/* coverage: fraction of the N inputs covered by the union of the windows (a convolution tiles all) */
static double coverage(const Indiv*g)
{ char c[N]; int j,t,i,cov=0; memset(c,0,sizeof c);
  for(j=0;j<H;j++) for(t=0;t<g->w[j];t++){ int i2=g->start[j]+t; if(i2>=0&&i2<N) c[i2]=1; }
  for(i=0;i<N;i++) cov+=c[i];
  return (double)cov/N; }

/* train the net; shared ties weight by within-window offset t=i-start. on_test: val vs test set */
static double run_net(const Indiv*g, uint32_t seed, int on_test)
{
    static double W[H][N], wsh[N], bh[H], v[H];
    double bo=0, h[H]; int j,t,e,s,c=0, ns = on_test?NTE:NVAL, mw=maxwidth(g);
    double (*Xe)[N] = on_test?Xte:Xval; int *ye = on_test?yte:yval;
    wseed(seed);
    for(t=0;t<N;t++) wsh[t]=0.1*wunif();
    for(j=0;j<H;j++){ for(t=0;t<N;t++) W[j][t]=0.1*wunif(); bh[j]=0; v[j]=0.1*wunif(); }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout, dwsh[N];
        for(j=0;j<H;j++){ double pre=bh[j];
            for(t=0;t<g->w[j];t++) pre += (g->shared?wsh[t]:W[j][t])*x[g->start[j]+t];
            h[j]=tanh(pre); opre+=v[j]*h[j]; }
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        if(g->shared){ for(t=0;t<mw;t++) dwsh[t]=0; }
        for(j=0;j<H;j++){ double dpre=dout*v[j]*(1.0-h[j]*h[j]); v[j]-=LR*dout*h[j]; bh[j]-=LR*dpre;
            for(t=0;t<g->w[j];t++){ double xx=x[g->start[j]+t];
                if(g->shared) dwsh[t]+=dpre*xx; else W[j][t]-=LR*dpre*xx; } }
        bo-=LR*dout;
        if(g->shared) for(t=0;t<mw;t++) wsh[t]-=LR*dwsh[t];
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        for(j=0;j<H;j++){ double pre=bh[j];
            for(t=0;t<g->w[j];t++) pre+=(g->shared?wsh[t]:W[j][t])*x[g->start[j]+t];
            opre+=v[j]*tanh(pre); }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
static double objective(const Indiv*g, double acc)
{ return (acc>=g_target) ? (2.0 - param_energy(g)) : acc; }

static void clampw(Indiv*g){ int j; for(j=0;j<H;j++){ if(g->w[j]<1) g->w[j]=1; if(g->w[j]>N) g->w[j]=N;
    if(g->start[j]<0) g->start[j]=0;
    if(g->start[j]+g->w[j]>N) g->start[j]=N-g->w[j]; } }

/* out += {shared_frac, mean_width, max_width, coverage, test}. allow_share=0 forces unshared. */
static void run_ga(int allow_share, uint32_t seed, double out[5])
{
    static Indiv pop[POP], nxt[POP]; double fit[POP], facc[POP]; int idx[POP]; int g,p,q,j;
    rseed(seed);
    for(p=0;p<POP;p++){ for(j=0;j<H;j++){ pop[p].start[j]=0; pop[p].w[j]=N; } pop[p].shared=0; }  /* dense contiguous seed */
    for(p=0;p<POP;p++){ facc[p]=run_net(&pop[p],(uint32_t)(seed+p*2654435761u+1u),0);
                        fit[p]=objective(&pop[p],facc[p]); }
    for(g=0;g<g_gens;g++){
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++) nxt[p]=pop[idx[p]];
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)]; nxt[p]=pop[a];
            for(j=0;j<H;j++){                     /* annealing on width: mostly shrink, rarely grow */
                if(rprob()<g_pgrow) nxt[p].w[j]+=1; else if(rprob()<0.5) nxt[p].w[j]-=1;
                if(rprob()<0.15) nxt[p].start[j]+=(rprob()<0.5)?-1:1;   /* window can slide */
            }
            if(allow_share && rprob()<g_pshare) nxt[p].shared^=1;
            clampw(&nxt[p]); }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++){ facc[p]=run_net(&pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0);
                            fit[p]=objective(&pop[p],facc[p]); }
    }
    { double sh=0,mw=0,mx=0,cv=0; int bi=0; double bf=-1;
      for(p=0;p<POP;p++){ sh+=pop[p].shared; mw+=meanwidth(&pop[p]); mx+=maxwidth(&pop[p]); cv+=coverage(&pop[p]);
                          if(fit[p]>bf){bf=fit[p];bi=p;} }
      out[0]+=sh/POP; out[1]+=mw/POP; out[2]+=mx/POP; out[3]+=cv/POP;
      out[4]+=run_net(&pop[bi],seed+999u,1); }
}

int main(void)
{
    int seeds=envint("SEEDS",16), sd, k;
    g_gens=envint("GENS",150); g_target=envdbl("TARGET",0.90);
    printf("LOCALITY IMPOSED, convolution emergent? contiguous windows given; sharing+width must emerge\n");
    printf("N=%d hidden=%d, target acc >= %.2f, %d seeds x %d gens.  compact conv = width K=%d, coverage 1\n\n",
           N, H, g_target, seeds, g_gens, K);
    printf("  %-18s  shared-frac  mean-w  max-w  coverage  test\n", "arm");
    { const char *nm[2]={"sharing forced off","sharing allowed"}; int as;
      for(as=0;as<2;as++){ double o[5]={0,0,0,0,0};
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga(as,(uint32_t)(sd*7+1),o); }
        for(k=0;k<5;k++) o[k]/=seeds;
        printf("  %-18s   %.3f       %.2f    %.2f   %.3f    %.3f\n", nm[as], o[0], o[1], o[2], o[3], o[4]); } }
    printf("\nconvolution emerged on top of imposed locality when: shared-frac ~1, width -> K(=%d), coverage ~1.\n", K);
    printf("(only locality is imposed; weight sharing = translation invariance and compact width must emerge.)\n");
    return 0;
}
