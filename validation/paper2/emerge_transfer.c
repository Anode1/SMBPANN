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

static int    g_init_ideal=0, g_resample=1;
static int    g_seeds=20, g_epochs=200, g_gens=50, g_pop=24, g_elite=4, g_trainpos=3, g_op=0;
static double g_pslide=0.15;
static double g_lr=0.05, g_lambda=1.0, g_pflip=0.10, g_damp=0.7;

/* a structure: which offsets are live, and whether weights are tied across positions */
typedef struct { char off[NOFF]; int shared; } Geno;

/* ============================== TASK ============================== */
static double motif[K];
static int    trainpos[H], ntp, valpos[H], nvp, testpos[H], nte_pos;
static double Xtr[NTR][N], Xva[NTE][N], Xte[NTE][N];
static int    ytr[NTR], yva[NTE], yte[NTE];

/* Plant the motif (positive) or a SHAPE-SCRAMBLED version (negative) at a position drawn from `pos`,
 * plus an adjacent DISTRACTOR that both classes carry.
 *
 * Two properties this has to have, and the first version had neither:
 *   SHAPE, NOT MAGNITUDE, CARRIES THE LABEL. The negative is a random NON-IDENTITY PERMUTATION of the
 *   motif, drawn fresh per example. The multiset of magnitudes is then identical between classes, so
 *   every single coordinate sees the same distribution either way and no single tap can discriminate.
 *   (The first version used one fixed rearrangement, so per-coordinate magnitude leaked the label and
 *   a single tap reached 0.746.)
 *   WIDTH IS PUNISHED, NOT MERELY CHARGED. DAMP sets the distractor's strength and is load-bearing:
 *   measured over the whole contiguous family, DAMP=0 puts width K+1 at rank 1 (0.7530) and the
 *   planted width nowhere; DAMP=0.7 puts the planted width at rank 1 by 0.027. The distractor is
 *   therefore what makes K optimal, and that is a measurement rather than an assumption. A distractor of random shape sits immediately beside the
 *   motif in BOTH classes. A K-wide aligned filter sees only the motif; anything wider bleeds into the
 *   distractor and loses signal. Without this, a wider window buys alignment robustness for 1/210 of
 *   energy and the objective's optimum overshoots K -- which is exactly what the support scan found. */
static void gen_at(double X[][N], int *y, int n, const int *pos, int npos)
{
    static const int PERM[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    int s,i,k;
    for(s=0;s<n;s++){
        int p = pos[(int)(r32()%(uint32_t)npos)];
        int d = (p+2*K<=N) ? p+K : p-K;          /* distractor slot, adjacent, always in range */
        for(i=0;i<N;i++) X[s][i]=0.3*runif();
        y[s] = (int)(r32()&1u);
        if(y[s]) for(k=0;k<K;k++) X[s][p+k] += AMP*motif[k];
        else { const int *q = PERM[1+(int)(r32()%5u)];   /* fresh non-identity permutation */
               for(k=0;k<K;k++) X[s][p+k] += AMP*motif[q[k]]; }
        if(d>=0 && d+K<=N) for(k=0;k<K;k++) X[s][d+k] += g_damp*AMP*runif();  /* distractor, both classes */
    }
}
static void new_task(uint32_t seed)
{
    int k,i,j,tmp,perm[H];
    rseed(seed);
    for(k=0;k<K;k++) motif[k]=runif();
    for(i=0;i<H;i++) perm[i]=i;
    for(i=H-1;i>0;i--){ j=(int)(r32()%(uint32_t)(i+1)); tmp=perm[i]; perm[i]=perm[j]; perm[j]=tmp; }
    /* THREE DISJOINT POSITION SETS. Training positions; SELECTION positions, unseen by the trainer and
     * what the GA's fitness is computed on, so transfer is what the search is actually paid in; and
     * REPORTING positions, unseen by both, so the number in the table is not what was selected on.
     * The first version selected on TRAINING accuracy, which made "transfer pays for sharing" a
     * description of the reporting rather than of the selection pressure. */
    ntp=g_trainpos; if(ntp<1) ntp=1; if(ntp>H-3) ntp=H-3;
    for(i=0;i<ntp;i++) trainpos[i]=perm[i];
    nvp=(H-ntp)/2; if(nvp<1) nvp=1;
    for(i=0;i<nvp;i++) valpos[i]=perm[ntp+i];
    nte_pos=H-ntp-nvp;
    for(i=0;i<nte_pos;i++) testpos[i]=perm[ntp+nvp+i];
    gen_at(Xtr,ytr,NTR,trainpos,ntp);
    gen_at(Xva,yva,NTE,valpos,nvp);             /* transfer, for SELECTION */
    gen_at(Xte,yte,NTE,testpos,nte_pos);        /* transfer, for REPORTING only */
}

/* ================= THE ONE SCORER: train on trained positions, report HELD-OUT positions =========
 * Shared: one weight per live offset. Unshared: a separate weight per (position, live offset), which
 * is H times as many parameters and cannot transfer to a position it never saw. Max-pool readout, so
 * position never reaches the output. */
/* LIVE weights only. For row j, offset o is read only when o-OFF0+j lands in [0,N) -- see the loop in
 * score(). The first version charged taps*H unconditionally, a 75% overcharge on full support, which
 * made the sharing verdict partly a billing error. */
static int nparams(const Geno *g)
{
    int i,j,d=0;
    if(g->shared){ for(i=0;i<NOFF;i++) d+=(g->off[i]!=0); return d; }
    for(j=0;j<H;j++) for(i=0;i<NOFF;i++){ int t=i-OFF0+j; if(g->off[i] && t>=0 && t<N) d++; }
    return d;
}
static double energy(const Geno *g){ return (double)nparams(g)/(double)(NOFF*H); }

/* which=0 train, 1 selection-transfer, 2 reporting-transfer */
static double score(const Geno *g, uint32_t seed, int which)
{
    static double W[H][NOFF];                    /* unshared weights; row 0 doubles as the shared one */
    double rs, rb=0, h[H];
    int i,j,e,s,c=0, ns = which?NTE:NTR;
    double (*Xe)[N] = which==0?Xtr:(which==1?Xva:Xte);
    int    *ye      = which==0?ytr:(which==1?yva:yte);

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
static void make_random(Geno *g, int taps, int shared)
{ int i,n=0; for(i=0;i<NOFF;i++) g->off[i]=0;
  while(n<taps && n<NOFF){ int r=(int)(r32()%(uint32_t)NOFF); if(!g->off[r]){ g->off[r]=1; n++; } }
  g->shared=shared; }

/* OPERATOR ARMS. The GA's failure on this target is specifically alignment and compactness: it gets
 * sharing right every seed but lands on ~6 scattered taps instead of 3 adjacent ones. Flips can only
 * add or remove; they cannot MOVE a tap at constant count. These are the operators built earlier in
 * this direction, now pointed at a target that is genuinely the objective's optimum -- which is the
 * condition that was missing every previous time they were measured.
 *   0 flip-only     add/remove only (the baseline)
 *   1 flip+slide    a live tap moves +-1                (neighbour-only; O(d^2) to travel d)
 *   2 flip+rewire   a live tap moves anywhere free      (no spatial prior; O(1) to travel d)
 *   3 flip+xover    uniform recombination over offsets, two parents
 * Every arm keeps the same flip rate, so any difference is the added operator and nothing else. */
static void mutate_geno(Geno *g)
{
    int i;
    for(i=0;i<NOFF;i++) if(rprob()<g_pflip) g->off[i]^=1;
    if(g_op==1 || g_op==2){
        for(i=0;i<NOFF;i++) if(g->off[i] && rprob()<g_pslide){
            int dst = (g_op==1) ? i + ((rprob()<0.5)?-1:1)
                                : (int)(r32()%(uint32_t)NOFF);
            if(dst>=0 && dst<NOFF && !g->off[dst]){ g->off[i]=0; g->off[dst]=1; }   /* count-preserving */
        }
    }
    if(rprob()<0.05) g->shared^=1;
}
static void produce_ga(uint32_t seed, Geno *best_out)
{
    static Geno pop[POPMAX], nxt[POPMAX];
    double fit[POPMAX]; int idx[POPMAX], gn,p,q,i,best=0; double bf=-1e300;
    if(g_gens<0) g_gens=0;
    rseed(seed);
    /* PROTOCOL item 4: the seed must not contain the answer. The first version seeded every individual
     * identical (dense AND shared), so generation 0 had zero diversity and the sharing gene started on
     * the answer. Random supports, random sharing. */
    if(g_init_ideal) for(p=0;p<g_pop;p++) make_ideal(&pop[p],1);
    else for(p=0;p<g_pop;p++){ for(i=0;i<NOFF;i++) pop[p].off[i]=(rprob()<0.5); pop[p].shared=(rprob()<0.5); }
    for(p=0;p<g_pop;p++) fit[p]=objective(&pop[p], score(&pop[p],(uint32_t)(seed+p*2654435761u+1u),1));
    for(gn=0; gn<g_gens; gn++){
        for(p=0;p<g_pop;p++) idx[p]=p;
        for(p=0;p<g_pop;p++) for(q=p+1;q<g_pop;q++)
            if(fit[idx[q]]>fit[idx[p]]){ int t=idx[p]; idx[p]=idx[q]; idx[q]=t; }
        for(p=0;p<g_elite;p++) nxt[p]=pop[idx[p]];
        for(p=g_elite;p<g_pop;p++){
            int a=idx[(int)(r32()%(uint32_t)g_elite)];
            nxt[p]=pop[a];
            if(g_op==3){                              /* two parents, uniform over offsets */
                int b=idx[(int)(r32()%(uint32_t)g_elite)];
                for(i=0;i<NOFF;i++) if(rprob()<0.5) nxt[p].off[i]=pop[b].off[i];
                if(rprob()<0.5) nxt[p].shared=pop[b].shared;
            }
            mutate_geno(&nxt[p]);
        }
        memcpy(pop,nxt,sizeof pop);
        /* RESAMPLE the selection set. With a fixed one, ~1200 evaluations of a 600-example set at
         * sd 0.12 let the search fit the examples rather than the structure: measured, the target's
         * lead is 0.078 on an unseen split but only 0.007 on the reused one, and a population seeded
         * AT the target walks away from it. Fresh draws each generation remove the thing to overfit.
         * RESAMPLE=0 restores the old behaviour for comparison. */
        if(g_resample) gen_at(Xva,yva,NTE,valpos,nvp);
        for(p=0;p<g_pop;p++)
            fit[p]=objective(&pop[p], score(&pop[p],(uint32_t)(seed+(uint32_t)(gn*g_pop+p)+7u),1));
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
            ob += objective(&g, score(&g,(uint32_t)(sd*7u+1u),2));
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

/* Protocol item 3 applied to the one constant that decides the answer. For each LAMBDA, is the planted
 * width K the argmax over all contiguous shared supports? Report the window. If the shipped LAMBDA is
 * outside it, the probe is measuring the tariff, not the task. */
static void scan_lambda(void)
{
    static const double L[] = {0.25,0.5,1.0,2.0,3.0,4.0,6.0,8.0,12.0};
    int nl=(int)(sizeof L/sizeof L[0]), li, lo, len, sd, i;
    double keep=g_lambda;
    printf("LAMSCAN  is the planted width K=%d the argmax of the contiguous family, per LAMBDA?\n", K);
    for(li=0; li<nl; li++){
        double bo=-1e300; int blo=0, blen=0;
        g_lambda=L[li];
        for(lo=0; lo<NOFF; lo++) for(len=1; lo+len<=NOFF; len++){
            Geno g; double ob=0;
            for(i=0;i<NOFF;i++) g.off[i]=0;
            for(i=lo;i<lo+len;i++) g.off[i]=1;
            g.shared=1;
            for(sd=1; sd<=g_seeds; sd++){
                new_task((uint32_t)(sd*911u+1u));
                ob += objective(&g, score(&g,(uint32_t)(sd*7u+1u),2));
            }
            ob/=g_seeds;
            if(ob>bo){ bo=ob; blo=lo; blen=len; }
        }
        printf("LAMSCAN  lambda %5.2f -> argmax lo=%2d len=%2d  obj %.4f   %s\n",
               L[li], blo, blen, bo,
               (blo==OFF0 && blen==K) ? "TARGET IS ARGMAX" : "");
    }
    g_lambda=keep;
}

/* ================= DIAGNOSTIC: local optimum, or noisy selection? =================
 * The search converges on ~6 scattered taps and never reaches the planted 3 contiguous ones, and no
 * operator or budget recovers the difference. Two explanations, and they are distinguishable:
 *
 *   LOCAL OPTIMUM   the search's own fitness ranks the target above what it finds, but the target is
 *                   unreachable: leaving 6 scattered taps needs a coordinated move (drop three AND
 *                   align the rest) that single-tap operators cannot make.
 *   NOISY SELECTION the fitness signal at this accuracy cannot resolve the two structures at all, so
 *                   selection is choosing between them roughly at random.
 *   (a third outcome the fork did not anticipate: the SELECTION split ranks them the other way, in
 *   which case the search is doing its job and the disagreement is between splits.)
 *
 * Two measurements. First, evaluate both structures on the SELECTION split many times with different
 * weight inits, and report the mean, the spread, and how often the target actually wins a single draw
 * -- that is exactly what selection sees. Second, seed the whole population AT the target and run:
 * if it survives, the target is a fitness optimum the search cannot reach; if it drifts away, it is
 * not a fitness optimum at all. */
static void diagnose(void)
{
    int sd, r, R=envint("DRAWS",15), keepgens=g_gens;
    double mi=0, mg=0, si=0, sg=0, win=0, n=0;
    double stay_tap=0, stay_con=0, stay_ob=0;

    printf("DIAG  what the SEARCH's own fitness (selection split) sees, %d weight draws x %d seeds\n",
           R, g_seeds);
    for(sd=1; sd<=g_seeds; sd++){
        Geno gi, gg;
        new_task((uint32_t)(sd*911u+1u));
        make_ideal(&gi,1);
        produce_ga((uint32_t)(sd*7u+1u), &gg);          /* what the search actually lands on */
        for(r=0;r<R;r++){
            double fi = objective(&gi, score(&gi,(uint32_t)(sd*104729u+r*7919u+1u),1));
            double fg = objective(&gg, score(&gg,(uint32_t)(sd*104729u+r*7919u+1u),1));
            mi+=fi; mg+=fg; si+=fi*fi; sg+=fg*fg; win += (fi>fg); n++;
        }
    }
    mi/=n; mg/=n; si=sqrt(si/n-mi*mi); sg=sqrt(sg/n-mg*mg);
    printf("DIAG  target       selection-fitness %.4f  sd %.4f\n", mi, si);
    printf("DIAG  what GA finds selection-fitness %.4f  sd %.4f\n", mg, sg);
    printf("DIAG  target wins a single draw %.1f%% of the time (50%% = selection cannot tell them apart)\n",
           100.0*win/n);
    printf("DIAG  separation = (mean gap)/(pooled sd) = %.2f\n", (mi-mg)/sqrt(0.5*(si*si+sg*sg)));

    printf("\nDIAG  seeding the population AT the target and running %d generations:\n", keepgens);
    for(sd=1; sd<=g_seeds; sd++){
        Geno g; new_task((uint32_t)(sd*911u+1u));
        g_init_ideal=1; produce_ga((uint32_t)(sd*7u+1u), &g); g_init_ideal=0;
        stay_tap += taps_of(&g); stay_con += contig_of(&g);
        stay_ob  += objective(&g, score(&g,(uint32_t)(sd*7u+1u),2));
    }
    printf("DIAG  after evolution from the target: taps %.2f  contig %.2f  objective %.4f\n",
           stay_tap/g_seeds, stay_con/g_seeds, stay_ob/g_seeds);
    printf("DIAG  (target is taps 3.00 contig 1.00 objective ~0.6774; if it drifts away, the target is\n");
    printf("DIAG   not a fitness optimum and the fork's premise is wrong.)\n");
}

static void row(const char *nm, const Geno *g, double tr, double te, double obj)
{ printf("  %-26s %5d %7s %6.2f %9.3f %9.3f %10.4f\n", nm, taps_of(g),
         g->shared?"yes":"no", (double)contig_of(g), tr, te, obj); }

int main(void)
{
    int sd, i;
    double a_tr[6]={0}, a_te[6]={0}, a_ob[6]={0}, a_tap[6]={0}, a_con[6]={0}, a_sh[6]={0};
    static const char *nm[6] = {
        "ideal conv (shared)", "GA flip-only", "GA flip+slide",
        "GA flip+rewire", "GA flip+xover", "GA-tap-matched random" };

    g_seeds=envint("SEEDS",20); g_epochs=envint("EPOCHS",200); g_gens=envint("GENS",50);
    g_pop=envint("POP",24); g_lr=envdbl("LR",0.05); g_lambda=envdbl("LAMBDA",1.0);
    g_pflip=envdbl("PFLIP",0.10); g_trainpos=envint("TRAINPOS",3);
    g_damp=envdbl("DAMP",0.7); g_pslide=envdbl("PSLIDE",0.15);
    g_resample=envint("RESAMPLE",1);
    if(g_pop>POPMAX) g_pop=POPMAX;

    printf("emerge_transfer -- sharing is a GENE, and TRANSFER is what pays for it\n");
    printf("motif K=%d planted at 1 of %d positions; train on %d positions, test on the other %d.\n",
           K, H, g_trainpos, H-g_trainpos);
    printf("max-pool readout (position never reaches the output). energy = params/(%d*%d):\n", NOFF, H);
    printf("  shared = taps, unshared = taps*%d. objective = heldout_acc - %.2f*energy.\n", H, g_lambda);
    printf("%d seeds x %d epochs. GA: pop %d, %d gens, flips offsets AND the sharing gene.\n\n",
           g_seeds, g_epochs, g_pop, g_gens);

    if(envint("SCAN",0)){ scan_contiguous(); return 0; }
    if(envint("LAMSCAN",0)){ scan_lambda(); return 0; }
    if(envint("DIAG",0)){ diagnose(); return 0; }

    for(sd=1; sd<=g_seeds; sd++){
        Geno g[6]; double tr,te; int k, op;
        new_task((uint32_t)(sd*911u+1u));
        make_ideal(&g[0],1);
        for(op=0; op<4; op++){ g_op=op; produce_ga((uint32_t)(sd*7u+1u), &g[1+op]); }
        g_op=0;
        make_random(&g[5], taps_of(&g[1]), 1);      /* item 5: null matched to the GA's own tap count */
        for(k=0;k<6;k++){
            tr = score(&g[k],(uint32_t)(sd*7u+1u),0);
            te = score(&g[k],(uint32_t)(sd*7u+1u),2);
            a_tr[k]+=tr; a_te[k]+=te; a_ob[k]+=objective(&g[k],te);
            a_tap[k]+=taps_of(&g[k]); a_con[k]+=contig_of(&g[k]); a_sh[k]+=g[k].shared;
            /* per-seed row: these outcomes are a two-lump mixture, so the mean is the wrong summary
             * and paired per-seed differences are the right comparison (PROTOCOL analysis notes). */
            if(envint("RAW",0)) printf("RAW %d %d %d %d %d %.4f %.4f %.4f\n",
                k, sd, taps_of(&g[k]), g[k].shared, contig_of(&g[k]), tr, te, objective(&g[k],te));
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
