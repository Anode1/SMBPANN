/* emerge_2d_compete.c -- why K=3 worsens, and nature's fix: COMPETITION (lateral inhibition).
 *
 * Sections 14-16: a 3-operation 2-D task is not solved at 3 channels, and adding depth/channels barely
 * helps or hurts. Diagnosis: the C channels do not SPECIALIZE -- plain gradient descent from random init
 * has nothing forcing channel A -> horizontal, B -> vertical, C -> diagonal, so channels redundantly
 * grab the easy orientation and the hard one goes uncovered. The symmetry between channels never breaks.
 *
 * Nature breaks it with COMPETITION: cortical orientation columns self-organize by lateral inhibition /
 * competitive learning (von der Malsburg 1973; Kohonen maps), where detectors REPEL in feature space so
 * each claims a different feature -- the same "shift the winner, push the neighbours away" refinement the
 * user pointed at. This adds exactly that pressure: after each step the channel filters repel each other
 * (a decorrelation term = lateral inhibition, strength lambda). Question: does competition let C=3 solve
 * K=3, where plain SGD (lambda=0) could not?
 *
 * Same K-operation task and multi-channel conv engine as emerge_2d_grow. Fixed C=3 (the operation count),
 * sweeping the competition strength lambda, for K=2 (control, already solvable) and K=3 (the failure).
 *
 * FINDING (8 seeds) -- NEGATIVE: competition does NOT fix K=3. K=3 at C=3 stays ~0.76-0.78 across every
 * lambda (target 0.85) -- at most a +0.02 bump, within noise; K=2 is unaffected (already solved). So the
 * K=3 wall is NOT a channel-specialization / symmetry-breaking problem: forcing the filters apart does
 * not help. This is the FOURTH failed fix (after more channels, an MLP head, a 2nd conv layer). The wall
 * is therefore not a single missing operator but a genuine capacity limit of this shallow conv+max-pool
 * setup on a 3-way conjunction over cluttered input -- per-channel detection error compounds
 * multiplicatively and no one refinement moves it. Self-contained C99. Build: make emerge_2d_compete
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G     12
#define KK    3
#define SP    (G-KK+1)
#define CMAX  3
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
static double g_lambda = 0.0;      /* competition (lateral inhibition) strength */

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

/* conv 1->C + per-channel max-pool + linear head. If g_lambda>0, channel filters repel each other
 * (lateral inhibition / decorrelation) after each step -> they specialize to different orientations. */
static double train_eval(int C, uint32_t seed)
{
    static double w[CMAX][KK][KK], bc[CMAX], rd[CMAX], rb;
    static double out[CMAX][SP][SP];
    int e,s,i,j,co,cb,r,c, cc=0;
    wseed(seed);
    for(co=0;co<C;co++){ for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]=0.3*wunif(); bc[co]=0; rd[co]=0.3*wunif(); }
    rb=0;
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        double p[CMAX]; int rm[CMAX],cm[CMAX]; double dp[CMAX], zo=rb, o, dout;
        for(co=0;co<C;co++){ double best=-1e9; int br=0,bc2=0;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xtr[s][r+i][c+j];
                double v=tanh(pre); out[co][r][c]=v; if(v>best){best=v;br=r;bc2=c;} }
            p[co]=best; rm[co]=br; cm[co]=bc2; zo+=rd[co]*best; }
        o=1.0/(1.0+exp(-zo)); dout=(o-ytr[s])*o*(1.0-o);
        for(co=0;co<C;co++) dp[co]=dout*rd[co];
        for(co=0;co<C;co++) rd[co]-=LR*dout*p[co];
        rb-=LR*dout;
        for(co=0;co<C;co++){ double v=out[co][rm[co]][cm[co]]; double dpre=dp[co]*(1.0-v*v);
            bc[co]-=LR*dpre;
            for(i=0;i<KK;i++) for(j=0;j<KK;j++) w[co][i][j]-=LR*dpre*Xtr[s][rm[co]+i][cm[co]+j]; }
        /* COMPETITION: each pair of channel filters repels (reduce their alignment) -> specialization */
        if(g_lambda>0.0) for(co=0;co<C;co++) for(cb=co+1;cb<C;cb++){
            double dot=0; for(i=0;i<KK;i++) for(j=0;j<KK;j++) dot+=w[co][i][j]*w[cb][i][j];
            for(i=0;i<KK;i++) for(j=0;j<KK;j++){ double a=w[co][i][j], b=w[cb][i][j];
                w[co][i][j]-=LR*g_lambda*dot*b; w[cb][i][j]-=LR*g_lambda*dot*a; } }
    }
    for(s=0;s<NTE;s++){ double zo=rb;
        for(co=0;co<C;co++){ double best=-1e9;
            for(r=0;r<SP;r++) for(c=0;c<SP;c++){ double pre=bc[co];
                for(i=0;i<KK;i++) for(j=0;j<KK;j++) pre+=w[co][i][j]*Xte[s][r+i][c+j];
                double v=tanh(pre); if(v>best)best=v; }
            zo+=rd[co]*best; }
        { double o=1.0/(1.0+exp(-zo)); if((o>0.5)==(yte[s]==1)) cc++; } }
    return (double)cc/NTE;
}
static double best_eval(int C, uint32_t seed)
{ int r; double best=0; for(r=0;r<RESTARTS;r++){ double a=train_eval(C,seed*131u+(uint32_t)r*97u+1u); if(a>best) best=a; } return best; }

int main(void)
{
    int seeds=envint("SEEDS",8), sd, K, li;
    double lam[4]={0.0, 0.10, 0.30, 0.60};
    g_target=envdbl("TARGET",0.85);
    TEPOCHS=envint("TEPOCHS",TEPOCHS); AMP=envdbl("AMP",AMP); RESTARTS=envint("RESTARTS",RESTARTS);
    printf("COMPETITION fixes specialization? C=3 channels, K-orientation task, lateral-inhibition strength lambda\n");
    printf("G=%dx%d k=%d, %d seeds x %d restarts, AMP=%.1f. lambda=0 is plain SGD (K=3 fails at ~0.75).\n\n", G, G, KK, seeds, RESTARTS, AMP);
    printf("  K ops (C=3) | accuracy at lambda = 0.00, 0.10, 0.30, 0.60\n");
    for(K=2;K<=3;K++){ g_nops=K;
        printf("  K=%d         |", K);
        for(li=0;li<4;li++){ g_lambda=lam[li]; double sum=0; int nv=0;
            for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*911u+(unsigned)K*53u+1u)); sum+=best_eval(CMAX,(uint32_t)(sd*7u+1u)); nv++; }
            printf("  %.3f", sum/nv); }
        printf("\n");
    }
    printf("\ncompetition is the missing piece if K=3 at C=3 crosses target for lambda>0 where lambda=0 fails:\n");
    printf("lateral inhibition breaks channel symmetry so the three channels specialize to the three orientations.\n");
    return 0;
}
