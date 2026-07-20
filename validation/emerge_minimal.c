/* emerge_minimal.c -- does the compact filter emerge from a MINIMAL seed too, not only from dense?
 *
 * emerge_offset.c grew the compact filter by pruning a DENSE seed under grouped (per-offset) mutation.
 * NEAT and its kin instead start MINIMAL and augment. This asks whether the same energy GA reaches the
 * same emergent structure from the opposite direction: start with NO offsets active and let grouped
 * mutation add them under the energy budget, versus start dense and prune. If both converge on the same
 * compact, generative kernel, the emergence is robust to the starting point; if only one direction
 * works, that itself is the finding.
 *
 * Same setup as emerge_offset: shared-weight net, genome = offset mask, grouped mutation flips a whole
 * offset (all its duplicated edges) at once, energy = (#taps + width*span)/(H*N), objective = minimize
 * energy subject to accuracy >= target. The only variable is the seed: DENSE (all offsets on, prune down)
 * vs MINIMAL (all offsets off, grow up). Symmetric add/remove mutation so selection, not the seed, drives
 * the outcome. Reports taps/span/contiguity/on-relevant/test for each seed.
 *
 * FINDING (24 seeds, width penalty 1.0): the emergence is ROBUST TO THE STARTING POINT. Dense-prune and
 * minimal-grow reach an essentially identical kernel: taps 3.5 vs 3.8 (~K), span 6.9 vs 7.0, contiguity
 * 0.77 vs 0.80, on-relevant 0.61 vs 0.65, test 0.877 vs 0.877. So with grouped mutation the filter
 * emerges the same way whether you prune from dense or grow from minimal (NEAT-style) -- the direction
 * does not matter. (A lucky 16-seed run gave a tighter 2.8/0.90; 24 seeds is the looser number,
 * and the robustness across direction is what holds firmly.) Self-contained C99. Build: make emerge_minimal
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define N   12
#define K   3
#define H   (N - K + 1)
#define NTR   64
#define NVAL  300
#define NTE   1000
#define TEPOCHS 50
#define LR    0.1
#define POP   24
#define ELITE 4
#define OFF0  (H)
#define NOFF  (N + H)
#define RLO   (OFF0)
#define RHI   (OFF0 + K - 1)

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static double wstar[K];
static double Xtr[NTR][N], Xval[NVAL][N], Xte[NTE][N];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_target = 0.90, g_flip = 0.06;   /* symmetric per-offset flip probability */
static double g_width = 0.0;
static int    g_gens = 200;
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

static int label_of(const double *x)
{ int p,k; double s=0; for(p=0;p+K<=N;p++){ double a=0; for(k=0;k<K;k++) a+=wstar[k]*x[p+k]; s+=tanh(a);} return s>0; }
static void gen(double X[][N],int*y,int n){ int s,i; for(s=0;s<n;s++){ for(i=0;i<N;i++) X[s][i]=runif(); y[s]=label_of(X[s]); } }
static void new_task(uint32_t seed){ int k; rseed(seed); for(k=0;k<K;k++) wstar[k]=runif(); gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE); }

static void offset_stats(char m[H][N], int *taps, int *span, int *onrel)
{
    char seen[NOFF]; int j,i,lo=NOFF,hi=-1,d=0,rel=0; memset(seen,0,sizeof seen);
    for(j=0;j<H;j++)for(i=0;i<N;i++) if(m[j][i]){ int o=i-j+OFF0;
        if(!seen[o]){seen[o]=1;d++; if(o>=RLO&&o<=RHI) rel++;} if(o<lo)lo=o; if(o>hi)hi=o; }
    *taps=d; *span=(hi>=lo)?(hi-lo+1):0; *onrel=rel;
}
static double param_energy(char m[H][N])
{ int d,sp,rel; offset_stats(m,&d,&sp,&rel); return (d + g_width*sp)/(H*N); }
static void mask_from_off(char off[NOFF], char m[H][N])
{ int j,i; for(j=0;j<H;j++)for(i=0;i<N;i++){ int o=i-j+OFF0; m[j][i]=(o>=0&&o<NOFF)?off[o]:0; } }

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
/* a minimal net (no offsets) is degenerate; force at least accuracy of a constant predictor via objective */
static double objective(char m[H][N], double acc)
{ int d,sp,rel; offset_stats(m,&d,&sp,&rel); if(d==0) return acc*0.5;   /* empty net: no credit */
  return (acc>=g_target) ? (2.0 - param_energy(m)) : acc; }

/* seed_mode 0 = DENSE (all offsets on, prune), 1 = MINIMAL (all off, grow). out += {taps,span,onrel,test,contig}. */
static void run_ga(int minimal, uint32_t seed, double out[5])
{
    static char opop[POP][NOFF], onxt[POP][NOFF], m[H][N];
    double fit[POP], facc[POP]; int idx[POP]; int g,p,q,o;
    rseed(seed);
    for(p=0;p<POP;p++) for(o=0;o<NOFF;o++) opop[p][o] = minimal ? 0 : 1;
    for(p=0;p<POP;p++){ mask_from_off(opop[p],m); facc[p]=run_net(m,(uint32_t)(seed+p*2654435761u+1u),0);
                        fit[p]=objective(m,facc[p]); }
    for(g=0;g<g_gens;g++){
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++) memcpy(onxt[p],opop[idx[p]],sizeof onxt[p]);
        for(p=ELITE;p<POP;p++){ int a=idx[(int)(r32()%ELITE)]; memcpy(onxt[p],opop[a],sizeof onxt[p]);
            for(o=1;o<NOFF;o++) if((double)r32()/4294967296.0<g_flip) onxt[p][o]^=1;   /* symmetric grouped flip */
        }
        memcpy(opop,onxt,sizeof opop);
        for(p=0;p<POP;p++){ mask_from_off(opop[p],m); facc[p]=run_net(m,(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0);
                            fit[p]=objective(m,facc[p]); }
    }
    { int bi=0; double bf=-1; for(p=0;p<POP;p++) if(fit[p]>bf){bf=fit[p];bi=p;}
      mask_from_off(opop[bi],m);
      { int taps,span,rel; offset_stats(m,&taps,&span,&rel);
        out[0]+=taps; out[1]+=span; out[2]+= taps?(double)rel/taps:0.0;
        out[3]+=run_net(m,seed+999u,1);
        out[4]+= span?(double)taps/span:0.0;
        if(getenv("DBG")) printf("  [dbg minimal=%d seed=%u taps=%d contig=%.2f onrel=%.2f test=%.3f]\n", minimal, seed, taps, span?(double)taps/span:0.0, taps?(double)rel/taps:0.0, run_net(m,seed+999u,1)); } }
}

int main(void)
{
    int seeds=envint("SEEDS",16), sd, k;
    g_gens=envint("GENS",200); g_target=envdbl("TARGET",0.90); g_width=envdbl("WIDTH",1.0);
    if(getenv("PROBE")){ char off[NOFF]; char m[H][N]; int q; new_task(132u);
        for(q=0;q<NOFF;q++) off[q]=0; off[10]=1; off[11]=1; off[12]=1; mask_from_off(off,m);
        printf("PROBE minimal: acc(off 10,11,12)=%.4f  acc2=%.4f\n", run_net(m,42u,0), run_net(m,7u,0)); return 0; }
    printf("MINIMAL vs DENSE seed under the SAME energy GA (grouped per-offset mutation, width penalty=%.1f)\n", g_width);
    printf("local K=%d generative filter, N=%d hidden=%d, %d seeds x %d gens, target %.2f\n", K, N, H, seeds, g_gens, g_target);
    printf("does the compact filter emerge from EITHER direction? compact = taps->K, span->K, contig->1, on-rel->1\n\n");
    printf("  seed        taps  span  contig  on-rel  test\n");
    { const char *nm[2]={"DENSE (prune)","MINIMAL (grow)"}; int mode;
      for(mode=0;mode<2;mode++){ double o[5]={0,0,0,0,0};
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga(mode,(uint32_t)(sd*7+1),o); }
        for(k=0;k<5;k++) o[k]/=seeds;
        printf("  %-12s %.1f  %.1f   %.2f    %.2f    %.3f\n", nm[mode], o[0], o[1], o[4], o[2], o[3]); } }
    printf("\nemergence is robust to the starting point if DENSE-prune and MINIMAL-grow reach the same compact,\n");
    printf("generative kernel; if only one direction gets there, the seed matters and that is the result.\n");
    return 0;
}
