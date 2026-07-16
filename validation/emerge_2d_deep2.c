/* emerge_2d_deep2.c -- the closer: does a deeper FEATURE EXTRACTOR (2 conv layers) recover K=3?
 *
 * emerge_2d_deep showed the K=3 failure is not the readout (an MLP head recovers K=2 but not K=3), so
 * the limit is the conv channels: a single conv layer cannot pack three clean orientation detectors
 * into three channels. This asks the last question in that corner -- is the limit specifically "ONE
 * conv layer cannot specialize", which a deeper extractor would fix, or a harder wall? We add a second
 * conv layer: 1 -> C1 intermediate channels -> C final channels (per-channel global max-pool -> linear
 * head, held fixed so only the extractor depth changes). Same K-operation task (K oriented bars all
 * present vs one dropped, balanced).
 *
 * Compares a 1-layer extractor (conv 1->C) against a 2-layer extractor (conv 1->C1, conv C1->C) at K=2
 * and K=3 across the final channel count C. The 2-layer extractor recovers K=3 if its K=3 row crosses
 * target at C=3, where the 1-layer row (and the MLP head) could not.
 *
 * FINDING (8 seeds) -- a firm NEGATIVE: extractor depth does NOT recover K=3; it makes it WORSE. At K=3
 * the 2-layer extractor is stuck at ~0.63 across all C (vs the 1-layer's up to 0.76), and K=2 is also
 * slightly worse. The deeper net adds optimization difficulty without cleaner orientation specialization.
 * So the K=3 limit is ROBUST -- not the readout (Sec 15, MLP head), not extractor depth (here) -- a wall
 * of this conv+max-pool+conjunction setup, not a single-layer-specialization artifact. Honest caveat:
 * the deeper net is plausibly under-trained, so this shows adding depth did not help (and hurt), not that
 * depth fundamentally cannot. Self-contained C99. make emerge_2d_deep2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G     12
#define KK    3
#define SP1   (G-KK+1)       /* = 10 after layer 1 */
#define SP2   (SP1-KK+1)     /* = 8  after layer 2 */
#define CMAX  4
#define C1FIX 4              /* intermediate channels in the 2-layer extractor */
#define NTR   256
#define NTE   1200
#define LR    0.05

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static uint32_t urand(uint32_t n){ return r32()%n; }
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}
static int    TEPOCHS = 250;
static double AMP     = 3.0;
static int    RESTARTS = 3;
static double g_target = 0.85;
static int    g_nops = 1;

static double Xtr[NTR][G][G], Xte[NTE][G][G];
static int    ytr[NTR], yte[NTE];

static void put_orient(double g[G][G],int o)
{ int r=(int)urand(G-2), c=(int)urand(G-2);
  if(o==0){ g[r][c]+=AMP; g[r][c+1]+=AMP; g[r][c+2]+=AMP; }
  else if(o==1){ g[r][c]+=AMP; g[r+1][c]+=AMP; g[r+2][c]+=AMP; }
  else { g[r][c]+=AMP; g[r+1][c+1]+=AMP; g[r+2][c+2]+=AMP; } }
static void gen(double X[][G][G],int*y,int n)
{
    int s,i,j,o,drop;
    for(s=0;s<n;s++){
        for(i=0;i<G;i++) for(j=0;j<G;j++) X[s][i][j]=runif();
        y[s]=(int)(r32()&1u);
        drop = y[s] ? -1 : (int)urand((uint32_t)g_nops);
        for(o=0;o<g_nops;o++) if(o!=drop) put_orient(X[s],o);
    }
}
static void new_task(uint32_t seed){ rseed(seed); gen(Xtr,ytr,NTR); gen(Xte,yte,NTE); }

/* 1-LAYER extractor: conv 1->C, per-channel max-pool, linear head. */
static double train_eval_1(int C, uint32_t seed)
{
    static double w[CMAX][KK][KK], bc[CMAX], rd[CMAX], rb;
    static double a[CMAX][SP1][SP1];
    int e,s,i,j,co,r,c, cc=0;
    wseed(seed);
    for(co=0;co<C;co++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]=0.3*wunif(); bc[co]=0; rd[co]=0.3*wunif(); }
    rb=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        double p[CMAX]; int rm[CMAX],cm[CMAX]; double dp[CMAX], zo=rb, o, dout;
        for(co=0;co<C;co++){ double best=-1e9; int br=0,bc2=0;
            for(r=0;r<SP1;r++) for(c=0;c<SP1;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xtr[s][r+i][c+j];
                double v=tanh(pre); a[co][r][c]=v; if(v>best){best=v;br=r;bc2=c;} }
            p[co]=best; rm[co]=br; cm[co]=bc2; zo+=rd[co]*best; }
        o=1.0/(1.0+exp(-zo)); dout=(o-ytr[s])*o*(1.0-o);
        for(co=0;co<C;co++) dp[co]=dout*rd[co];
        for(co=0;co<C;co++) rd[co]-=LR*dout*p[co];
        rb-=LR*dout;
        for(co=0;co<C;co++){ double v=a[co][rm[co]][cm[co]]; double dpre=dp[co]*(1.0-v*v);
            bc[co]-=LR*dpre;
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]-=LR*dpre*Xtr[s][rm[co]+i][cm[co]+j]; }
    }
    for(s=0;s<NTE;s++){ double zo=rb;
        for(co=0;co<C;co++){ double best=-1e9;
            for(r=0;r<SP1;r++) for(c=0;c<SP1;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xte[s][r+i][c+j];
                double v=tanh(pre); if(v>best)best=v; }
            zo+=rd[co]*best; }
        { double o=1.0/(1.0+exp(-zo)); if((o>0.5)==(yte[s]==1)) cc++; } }
    return (double)cc/NTE;
}

/* 2-LAYER extractor: conv 1->C1FIX, conv C1FIX->C, per-channel max-pool over layer 2, linear head. */
static double train_eval_2(int C, uint32_t seed)
{
    static double w1[C1FIX][KK][KK], b1[C1FIX];
    static double w2[CMAX][C1FIX][KK][KK], b2[CMAX], rd[CMAX], rb;
    static double a1[C1FIX][SP1][SP1], a2[CMAX][SP2][SP2], da1[C1FIX][SP1][SP1];
    int e,s,i,j,c1,co,r,c, cc=0;
    wseed(seed);
    for(c1=0;c1<C1FIX;c1++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w1[c1][i][j]=0.3*wunif(); b1[c1]=0; }
    for(co=0;co<C;co++){ for(c1=0;c1<C1FIX;c1++) for(i=0;i<KK;i++) for(j=0;j<KK;j++) w2[co][c1][i][j]=0.3*wunif();
        b2[co]=0; rd[co]=0.3*wunif(); }
    rb=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        double p[CMAX]; int rm[CMAX],cm[CMAX]; double dp[CMAX], zo=rb, o, dout;
        for(c1=0;c1<C1FIX;c1++) for(r=0;r<SP1;r++) for(c=0;c<SP1;c++){ double pre=b1[c1];
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w1[c1][i][j]*Xtr[s][r+i][c+j];
            a1[c1][r][c]=tanh(pre); }
        for(co=0;co<C;co++){ double best=-1e9; int br=0,bc2=0;
            for(r=0;r<SP2;r++) for(c=0;c<SP2;c++){ double pre=b2[co];
                for(c1=0;c1<C1FIX;c1++) for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w2[co][c1][i][j]*a1[c1][r+i][c+j];
                double v=tanh(pre); a2[co][r][c]=v; if(v>best){best=v;br=r;bc2=c;} }
            p[co]=best; rm[co]=br; cm[co]=bc2; zo+=rd[co]*best; }
        o=1.0/(1.0+exp(-zo)); dout=(o-ytr[s])*o*(1.0-o);
        for(co=0;co<C;co++) dp[co]=dout*rd[co];
        for(co=0;co<C;co++) rd[co]-=LR*dout*p[co];
        rb-=LR*dout;
        for(c1=0;c1<C1FIX;c1++) for(r=0;r<SP1;r++) for(c=0;c<SP1;c++) da1[c1][r][c]=0;
        for(co=0;co<C;co++){ double v=a2[co][rm[co]][cm[co]]; double dpre=dp[co]*(1.0-v*v);
            b2[co]-=LR*dpre;
            for(c1=0;c1<C1FIX;c1++) for(i=0;i<KK;i++) for(j=0;j<KK;j++){
                da1[c1][rm[co]+i][cm[co]+j]+=dpre*w2[co][c1][i][j];
                w2[co][c1][i][j]-=LR*dpre*a1[c1][rm[co]+i][cm[co]+j]; } }
        for(c1=0;c1<C1FIX;c1++) for(r=0;r<SP1;r++) for(c=0;c<SP1;c++){ double d=da1[c1][r][c];
            if(d==0.0) continue;
            double dpre=d*(1.0-a1[c1][r][c]*a1[c1][r][c]); b1[c1]-=LR*dpre;
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) w1[c1][i][j]-=LR*dpre*Xtr[s][r+i][c+j]; }
    }
    for(s=0;s<NTE;s++){ double zo=rb;
        for(c1=0;c1<C1FIX;c1++) for(r=0;r<SP1;r++) for(c=0;c<SP1;c++){ double pre=b1[c1];
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w1[c1][i][j]*Xte[s][r+i][c+j];
            a1[c1][r][c]=tanh(pre); }
        for(co=0;co<C;co++){ double best=-1e9;
            for(r=0;r<SP2;r++) for(c=0;c<SP2;c++){ double pre=b2[co];
                for(c1=0;c1<C1FIX;c1++) for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w2[co][c1][i][j]*a1[c1][r+i][c+j];
                double v=tanh(pre); if(v>best)best=v; }
            zo+=rd[co]*best; }
        { double o=1.0/(1.0+exp(-zo)); if((o>0.5)==(yte[s]==1)) cc++; } }
    return (double)cc/NTE;
}
static double best1(int C,uint32_t seed){ int r; double b=0; for(r=0;r<RESTARTS;r++){ double a=train_eval_1(C,seed*131u+(uint32_t)r*97u+1u); if(a>b)b=a; } return b; }
static double best2(int C,uint32_t seed){ int r; double b=0; for(r=0;r<RESTARTS;r++){ double a=train_eval_2(C,seed*131u+(uint32_t)r*97u+1u); if(a>b)b=a; } return b; }

int main(void)
{
    int seeds=envint("SEEDS",8), sd, K, C;
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("2-D DEPTH-IN-EXTRACTOR: does a 2nd conv layer let C final channels cover K orientations?\n");
    printf("G=%dx%d k=%d, %d seeds x %d restarts, AMP=%.1f. 2-layer = conv 1->%d, conv %d->C; linear head.\n",
           G, G, KK, seeds, RESTARTS, AMP, C1FIX, C1FIX);
    printf("K oriented bars all present vs one dropped, balanced. does 2-layer solve K=3 at C=3?\n\n");
    printf("  extractor | K | accuracy at final C=1..%d\n", CMAX);
    for(K=2;K<=3;K++){ g_nops=K;
        printf("  1-layer   | %d |", K);
        for(C=1;C<=CMAX;C++){ double sum=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)K*53u+1u)); sum+=best1(C,(uint32_t)(sd*7u+1u)); nv++; }
            printf(" %.3f", sum/nv); }
        printf("\n  2-layer   | %d |", K);
        for(C=1;C<=CMAX;C++){ double sum=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)K*53u+1u)); sum+=best2(C,(uint32_t)(sd*7u+1u)); nv++; }
            printf(" %.3f", sum/nv); }
        printf("\n");
    }
    printf("\n2-layer extractor recovers K=3 if its K=3 row crosses target at C=3 where 1-layer did not\n");
    printf("(the limit was 'one conv layer cannot specialize'); if not, it is a harder capacity wall.\n");
    return 0;
}
