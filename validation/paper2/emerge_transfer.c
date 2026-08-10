/* emerge_transfer.c -- paper 2: make the convolution ACTUALLY OPTIMAL, then ask whether search finds it.
 * Self-contained C99. Build: make emerge_transfer
 *
 * WHY THIS EXISTS. Three models, three energy definitions, one answer: the objective ranked the
 * hand-built convolution BELOW what the search found unaided (see ../../articles/smbpann2/FINDINGS.md).
 * Tuning operators against that is pointless. The target has to become the optimum first.
 *
 * WHY IT WAS NEVER THE OPTIMUM. Weight sharing earns its keep by working at positions never trained
 * on. No task in this line ever asked for that: every position appeared in training and the label
 * summed over all of them, so sharing cost nothing extra and bought nothing either, and a smaller
 * scattered support won on energy.
 *
 * WHY THE OBVIOUS FIX IS NOT ENOUGH -- the trap that refuted emerge_gen2. Holding out positions does
 * nothing if the genome is an offset mask, because then EVERY candidate is convolutional, transfer is
 * constant across candidates, and the objective is flat over the outcome. That is precisely how
 * emerge_gen2 died: fitness constant at 2 - 1/12 for every single-group genome.
 *
 * THE COMBINATION THIS PROBE ADDS. Both at once, which neither predecessor had:
 *   - SHARING IS A GENE the search may switch off (emerge_local had this; its task never charged for it)
 *   - TRANSFER IS WHAT PAYS FOR IT (emerge_gen2 had this; its genome could not choose)
 * plus honest parameter counting, which only bites once sharing is optional: a shared kernel stores
 * `taps` weights, an unshared one stores `taps * H`. That factor of H is the whole economic case for
 * convolution, and it has never before been in play.
 *
 * TASK. A K-tap motif is planted at ONE position per example. Positives carry the true motif shape,
 * negatives a scrambled shape of equal magnitude, so mere presence is uninformative and only SHAPE
 * discriminates. Training examples plant it only at TRAINPOS positions; the held-out set uses the
 * positions never trained on. Readout is MAX-POOL over positions, so position itself never reaches the
 * output and cannot be memorised.
 *
 * PREDICTIONS, fixed before running (this is protocol item 9, run before any search):
 *   unshared should fit the trained positions and collapse to chance on held-out ones;
 *   shared + compact aligned support should transfer;
 *   therefore the hand-built convolution should be the OPTIMUM, for the first time in this line.
 * If it is not, the premise that convolution is the efficient architecture does not hold at this
 * scale, which is a larger result than any operator finding. Either way the search question only
 * becomes meaningful AFTER this check passes.
 *
 * env: SEEDS EPOCHS LR LAMBDA TRAINPOS GENS POP PFLIP
 */
#include "common.h"

#define N      12               /* inputs                                    */
#define K      3                /* motif width -- the target                 */
#define H      (N - K + 1)      /* positions the motif may sit at (10)       */
#define NOFF   (N + H - 1)      /* 21 offsets; shared weights indexed by i-j+OFF0 */
#define OFF0   (H - 1)
#define NTR    96
#define NTE    600
#define POPMAX 64
#define AMP    2.0

static int    g_seeds=20, g_epochs=200, g_gens=50, g_pop=24, g_elite=4, g_trainpos=3;
static double g_lr=0.05, g_lambda=1.0, g_pflip=0.10;

/* a structure: which offsets are live, and whether weights are tied across positions */
typedef struct { char off[NOFF]; int shared; } Geno;

/* ============================== TASK ============================== */
static double motif[K];
static int    trainpos[H], ntp, testpos[H], nte_pos;
static double Xtr[NTR][N], Xte[NTE][N];
static int    ytr[NTR], yte[NTE];

/* plant the motif (positive) or a scrambled version (negative) at a position drawn from `pos` */
static void gen_at(double X[][N], int *y, int n, const int *pos, int npos)
{
    int s,i,k;
    for(s=0;s<n;s++){
        int p = pos[(int)(r32()%(uint32_t)npos)];
        for(i=0;i<N;i++) X[s][i]=0.3*runif();
        y[s] = (int)(r32()&1u);
        if(y[s]) for(k=0;k<K;k++) X[s][p+k] += AMP*motif[k];
        else {                                   /* same magnitudes, wrong shape */
            for(k=0;k<K;k++) X[s][p+k] += AMP*motif[(k+1)%K] * ((k&1)?-1.0:1.0);
        }
    }
}
static void new_task(uint32_t seed)
{
    int k,i,j,tmp,perm[H];
    rseed(seed);
    for(k=0;k<K;k++) motif[k]=runif();
    for(i=0;i<H;i++) perm[i]=i;
    for(i=H-1;i>0;i--){ j=(int)(r32()%(uint32_t)(i+1)); tmp=perm[i]; perm[i]=perm[j]; perm[j]=tmp; }
    ntp=g_trainpos; if(ntp<1) ntp=1; if(ntp>H-1) ntp=H-1;
    for(i=0;i<ntp;i++) trainpos[i]=perm[i];
    nte_pos=H-ntp;
    for(i=0;i<nte_pos;i++) testpos[i]=perm[ntp+i];
    gen_at(Xtr,ytr,NTR,trainpos,ntp);
    gen_at(Xte,yte,NTE,testpos,nte_pos);        /* positions NEVER trained on */
}

/* ================= THE ONE SCORER: train on trained positions, report HELD-OUT positions =========
 * Shared: one weight per live offset. Unshared: a separate weight per (position, live offset), which
 * is H times as many parameters and cannot transfer to a position it never saw. Max-pool readout, so
 * position never reaches the output. */
static int nparams(const Geno *g)
{ int i,d=0; for(i=0;i<NOFF;i++) d+=(g->off[i]!=0); return g->shared ? d : d*H; }
static double energy(const Geno *g){ return (double)nparams(g)/(double)(NOFF*H); }

static double score(const Geno *g, uint32_t seed, int on_heldout)
{
    static double W[H][NOFF];                    /* unshared weights; row 0 doubles as the shared one */
    double rs, rb=0, h[H];
    int i,j,e,s,c=0, ns = on_heldout?NTE:NTR;
    double (*Xe)[N] = on_heldout?Xte:Xtr; int *ye = on_heldout?yte:ytr;

    wseed(seed);
    for(j=0;j<H;j++) for(i=0;i<NOFF;i++) W[j][i] = g->off[i] ? 0.3*wunif() : 0.0;
    if(g->shared) for(j=1;j<H;j++) for(i=0;i<NOFF;i++) W[j][i]=W[0][i];
    rs=0.5;

    for(e=0;e<g_epochs;e++)
    for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double o,dout,mx; int am=0;
        for(j=0;j<H;j++){ double pre=0;
            for(i=0;i<N;i++){ int oo=i-j+OFF0; if(g->off[oo]) pre += W[j][oo]*x[i]; }
            h[j]=tanh(pre); }
        mx=h[0]; for(j=1;j<H;j++) if(h[j]>mx){ mx=h[j]; am=j; }      /* MAX-POOL */
        o=1.0/(1.0+exp(-(rs*mx+rb))); dout=(o-ytr[s])*o*(1.0-o);
        { double dpre = dout*rs*(1.0-h[am]*h[am]);
          rs -= g_lr*dout*mx; rb -= g_lr*dout;
          if(g->shared){                          /* one gradient, applied to the tied kernel */
              for(i=0;i<N;i++){ int oo=i-am+OFF0; if(g->off[oo]) W[0][oo] -= g_lr*dpre*x[i]; }
              for(j=1;j<H;j++) for(i=0;i<NOFF;i++) W[j][i]=W[0][i];
          } else {
              for(i=0;i<N;i++){ int oo=i-am+OFF0; if(g->off[oo]) W[am][oo] -= g_lr*dpre*x[i]; }
          } }
    }
    for(s=0;s<ns;s++){
        const double *x=Xe[s]; double mx,o; int jj;
        for(j=0;j<H;j++){ double pre=0;
            for(i=0;i<N;i++){ int oo=i-j+OFF0; if(g->off[oo]) pre += W[j][oo]*x[i]; }
            h[j]=tanh(pre); }
        mx=h[0]; for(jj=1;jj<H;jj++) if(h[jj]>mx) mx=h[jj];
        o=1.0/(1.0+exp(-(rs*mx+rb))); if((o>0.5)==(ye[s]==1)) c++;
    }
    return (double)c/ns;
}
static double objective(const Geno *g, double acc){ return acc - g_lambda*energy(g); }
static int taps_of(const Geno *g){ int i,d=0; for(i=0;i<NOFF;i++) d+=(g->off[i]!=0); return d; }
static int contig_of(const Geno *g)
{ int i,lo=NOFF,hi=-1,d=0;
  for(i=0;i<NOFF;i++) if(g->off[i]){ d++; if(i<lo) lo=i; if(i>hi) hi=i; }
  return d>0 && d==hi-lo+1; }

/* ============================ PRODUCERS ============================ */
static void make_ideal(Geno *g, int shared)
{ int i; for(i=0;i<NOFF;i++) g->off[i]=0;
  for(i=OFF0; i<OFF0+K && i<NOFF; i++) g->off[i]=1;
  g->shared=shared; }
static void make_full(Geno *g, int shared)
{ int i; for(i=0;i<NOFF;i++) g->off[i]=1; g->shared=shared; }
static void make_random(Geno *g, int taps, int shared)
{ int i,n=0; for(i=0;i<NOFF;i++) g->off[i]=0;
  while(n<taps && n<NOFF){ int r=(int)(r32()%(uint32_t)NOFF); if(!g->off[r]){ g->off[r]=1; n++; } }
  g->shared=shared; }

static void produce_ga(uint32_t seed, Geno *best_out)
{
    static Geno pop[POPMAX], nxt[POPMAX];
    double fit[POPMAX]; int idx[POPMAX], gn,p,q,i,best=0; double bf=-1e300;
    rseed(seed);
    for(p=0;p<g_pop;p++) make_full(&pop[p], 1);      /* dense and shared; the search may undo either */
    for(p=0;p<g_pop;p++) fit[p]=objective(&pop[p], score(&pop[p],(uint32_t)(seed+p*2654435761u+1u),0));
    for(gn=0; gn<g_gens; gn++){
        for(p=0;p<g_pop;p++) idx[p]=p;
        for(p=0;p<g_pop;p++) for(q=p+1;q<g_pop;q++)
            if(fit[idx[q]]>fit[idx[p]]){ int t=idx[p]; idx[p]=idx[q]; idx[q]=t; }
        for(p=0;p<g_elite;p++) nxt[p]=pop[idx[p]];
        for(p=g_elite;p<g_pop;p++){
            int a=idx[(int)(r32()%(uint32_t)g_elite)];
            nxt[p]=pop[a];
            for(i=0;i<NOFF;i++) if(rprob()<g_pflip) nxt[p].off[i]^=1;
            if(rprob()<0.05) nxt[p].shared^=1;       /* sharing is a GENE, not a setting */
        }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<g_pop;p++)
            fit[p]=objective(&pop[p], score(&pop[p],(uint32_t)(seed+(uint32_t)(gn*g_pop+p)+7u),0));
    }
    for(p=0;p<g_pop;p++) if(fit[p]>bf){ bf=fit[p]; best=p; }
    *best_out = pop[best];
}


/* STANDING CHECK (protocol item 9, done properly): the target is not "beats the five arms I chose",
 * it is "is the argmax of a family". Enumerate every contiguous shared support and print the top of
 * the ranking. A hand-built target that is rank 3 of 231 is not an optimum. ~30s at 20 seeds. */
static void scan_contiguous(void)
{
    int lo, len, sd, i, nb=0; double best[8]; int blo[8], blen[8];
    for(i=0;i<8;i++){ best[i]=-1e300; blo[i]=0; blen[i]=0; }
    printf("SCAN  every contiguous shared support (lo,len), objective at LAMBDA=%.2f, %d seeds\n",
           g_lambda, g_seeds);
    for(lo=0; lo<NOFF; lo++) for(len=1; lo+len<=NOFF; len++){
        Geno g; double ob=0;
        for(i=0;i<NOFF;i++) g.off[i]=0;
        for(i=lo;i<lo+len;i++) g.off[i]=1;
        g.shared=1;
        for(sd=1; sd<=g_seeds; sd++){
            new_task((uint32_t)(sd*911u+1u));
            ob += objective(&g, score(&g,(uint32_t)(sd*7u+1u),1));
        }
        ob/=g_seeds;
        for(i=0;i<8;i++) if(ob>best[i]){
            int q; for(q=7;q>i;q--){ best[q]=best[q-1]; blo[q]=blo[q-1]; blen[q]=blen[q-1]; }
            best[i]=ob; blo[i]=lo; blen[i]=len; break; }
        nb++;
    }
    printf("SCAN  %d supports enumerated. aligned run starts at lo=%d, planted width K=%d.\n", nb, OFF0, K);
    for(i=0;i<8;i++) printf("SCAN  #%d  lo=%2d len=%2d  objective %.4f%s\n", i+1, blo[i], blen[i], best[i],
                            (blo[i]==OFF0 && blen[i]==K) ? "   <-- the hand-built target" : "");
}

static void row(const char *nm, const Geno *g, double tr, double te, double obj)
{ printf("  %-26s %5d %7s %6.2f %9.3f %9.3f %10.4f\n", nm, taps_of(g),
         g->shared?"yes":"no", (double)contig_of(g), tr, te, obj); }

int main(void)
{
    int sd, i;
    double a_tr[6]={0}, a_te[6]={0}, a_ob[6]={0}, a_tap[6]={0}, a_con[6]={0}, a_sh[6]={0};
    static const char *nm[6] = {
        "ideal conv (shared)", "ideal support, UNSHARED", "full support (shared)",
        "tap-matched random", "full support, UNSHARED", "GA (searches both)" };

    g_seeds=envint("SEEDS",20); g_epochs=envint("EPOCHS",200); g_gens=envint("GENS",50);
    g_pop=envint("POP",24); g_lr=envdbl("LR",0.05); g_lambda=envdbl("LAMBDA",1.0);
    g_pflip=envdbl("PFLIP",0.10); g_trainpos=envint("TRAINPOS",3);
    if(g_pop>POPMAX) g_pop=POPMAX;

    printf("emerge_transfer -- sharing is a GENE, and TRANSFER is what pays for it\n");
    printf("motif K=%d planted at 1 of %d positions; train on %d positions, test on the other %d.\n",
           K, H, g_trainpos, H-g_trainpos);
    printf("max-pool readout (position never reaches the output). energy = params/(%d*%d):\n", NOFF, H);
    printf("  shared = taps, unshared = taps*%d. objective = heldout_acc - %.2f*energy.\n", H, g_lambda);
    printf("%d seeds x %d epochs. GA: pop %d, %d gens, flips offsets AND the sharing gene.\n\n",
           g_seeds, g_epochs, g_pop, g_gens);

    if(envint("SCAN",0)){ scan_contiguous(); return 0; }

    for(sd=1; sd<=g_seeds; sd++){
        Geno g[6]; double tr,te; int k;
        new_task((uint32_t)(sd*911u+1u));
        make_ideal(&g[0],1); make_ideal(&g[1],0);
        make_full(&g[2],1);
        make_random(&g[3], K, 1);
        make_full(&g[4],0);
        produce_ga((uint32_t)(sd*7u+1u), &g[5]);
        for(k=0;k<6;k++){
            tr = score(&g[k],(uint32_t)(sd*7u+1u),0);
            te = score(&g[k],(uint32_t)(sd*7u+1u),1);
            a_tr[k]+=tr; a_te[k]+=te; a_ob[k]+=objective(&g[k],te);
            a_tap[k]+=taps_of(&g[k]); a_con[k]+=contig_of(&g[k]); a_sh[k]+=g[k].shared;
        }
    }

    printf("  %-26s %5s %7s %6s %9s %9s %10s\n",
           "producer","taps","shared","contig","train acc","heldout","objective");
    for(i=0;i<6;i++){
        printf("  %-26s %5.1f %7.2f %6.2f %9.3f %9.3f %10.4f\n", nm[i],
               a_tap[i]/g_seeds, a_sh[i]/g_seeds, a_con[i]/g_seeds,
               a_tr[i]/g_seeds, a_te[i]/g_seeds, a_ob[i]/g_seeds);
    }
    printf("\nGATE (protocol item 9, before any operator work):\n");
    printf("  PASS if 'ideal conv (shared)' has the highest objective -- the convolution is finally the\n");
    printf("  optimum, and the search question becomes meaningful.\n");
    printf("  The UNSHARED rows are the mechanism check: they should fit train and collapse on heldout.\n");
    printf("  If they transfer too, the task does not demand sharing and this probe is not yet valid.\n");
    (void)row;
    return 0;
}
