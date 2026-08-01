/* emerge_gen.c -- DOES AN ENERGY BUDGET INDUCE TRANSLATION GENERALIZATION?
 *
 * WHY THIS PROBE EXISTS. A previous probe (emerge_tile.c) scored a STRUCTURAL predicate: one shared
 * filter, placements on a regular stride, no gap wider than the kernel. That predicate is satisfied for
 * free by a saturated genome, because a fully occupied single-group placement has all gaps equal to 1.
 * Since the parameter energy groups/P is exactly indifferent to placement, occupancy was an unconstrained
 * free variable, and every operator that added units appeared to discover structure. Controls confirmed
 * it: an operator that copies nothing scored 26%, one that deliberately destroyed the target spacing
 * scored 59%, and the operator designed to build the structure scored 17%. The measurement, not the
 * search, produced the result.
 *
 * THE FIX IS TO MEASURE THE FUNCTION, NOT THE SHAPE. Regularity is not functionally meaningful once
 * coverage is complete: a dense tiling and a ragged full-coverage placement compute the same map. What a
 * convolution actually buys is GENERALIZATION ACROSS TRANSLATION. So we test that directly.
 *
 *   Train on motifs planted at a few positions only.  Test on the positions never trained on.
 *
 * A network with shared weights over adequate coverage transfers to unseen positions. A network of
 * per-position detectors cannot, however many units it has: the unseen positions carry untrained
 * weights. Saturation is therefore no longer a route to a good score, there is no threshold to game, and
 * the outcome is continuous with a known floor (chance) and a measurable ceiling (a hand-built
 * convolution).
 *
 * THE QUESTION. Does a parameter-counting budget (charge per distinct filter) induce networks that
 * generalize across translation, and does it need help from a copying operator to do so?
 *
 * DISCIPLINE, after the failure above:
 *   - every mutation rate is exposed to the environment and swept, including padd/pmerge/psplit;
 *   - every operator has a NULL TWIN, identical except for the property being claimed;
 *   - the equivariance measure carries a NEGATIVE class, so an always-fire detector cannot score 0;
 *   - the first thing checked is that the two reference networks separate on the measure at all.
 *
 * Build: make emerge_gen
 * Env: SEEDS GENS NIN NTRAINPOS TARGET PREM PADD PMERGE PSPLIT POPRATE OP
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NMAX   24
#define K      3
#define NTR    96          /* training examples, drawn only from the trained positions */
#define NVAL   300
#define NTE    600         /* held-out: motifs at positions never trained on */
#define TEPOCHS_MAX 400
#define LR     0.1
#define POP    24
#define ELITE  4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static double rprob(void){return (double)r32()/4294967296.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int    g_n=12, g_ntrainpos=2;
static int    g_trainpos[NMAX], g_ntp=0;      /* positions the motif is planted at during training */
static int    g_valpos[NMAX], g_nvp=0;        /* positions FITNESS is computed on: train positions plus
                                               * g_nvalpos extra. With g_nvalpos=0 selection cannot see
                                               * any position the weights were not trained on, which is
                                               * the condition under test. */
static int    g_nvalpos=0;
static double wstar[K];
static double Xtr[NTR][NMAX], Xval[NVAL][NMAX], Xte[NTE][NMAX];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double g_target=0.90;
static double g_prem=0.030, g_padd=0.006, g_pmerge=0.060, g_psplit=0.010;
static double g_oprate=0.30;                  /* firing rate of whichever structural operator is active */
static int    g_op=0;                         /* 0 none, 1 tandem duplication, 2 null-random-filter,
                                               * 3 null-copied-filter-random-place, 4 segment repeat */
static double g_alpha=1.0, g_beta=0.0;
static int    g_gens=80;
static int    g_epmax=400;      /* hard cap on development, so a run always terminates */
static int    g_patience=20;    /* epochs without a real improvement in TRAINING loss before stopping */
static double g_eps=1e-3;       /* what counts as a real improvement, relative */
static long   g_ep_sum=0, g_ep_n=0, g_ep_cap=0;   /* audit: what the rule actually did */
static int    g_restart=1;
static int    g_seeddump=0;   /* emit one line per task seed, so comparisons can be paired */      /* independent weight initialisations averaged into one fitness */
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

typedef struct { char on[NMAX]; unsigned char g[NMAX]; } Geno;

/* A sample is positive iff the motif is present. During TRAINING it is planted only at positions in
 * g_trainpos; the held-out set plants only at the complement. Negatives are noise at matched scale, so
 * a detector that fires on everything scores exactly chance rather than perfectly. */
static void gen_at(double X[][NMAX], int *y, int n, const int *pos, int npos)
{
    int s,i,k,p;
    for(s=0;s<n;s++){
        for(i=0;i<g_n;i++) X[s][i]=0.30*runif();
        y[s]=(rprob()<0.5);
        if(y[s]){ p=pos[(int)(r32()%(uint32_t)npos)];
                  for(k=0;k<K;k++) X[s][(p+k)%g_n]+=wstar[k]; }
    }
}
static void new_task(uint32_t seed)
{
    int k,i,j,tmp,perm[NMAX],held[NMAX],nh=0; double nrm=0;
    rseed(seed);
    for(k=0;k<K;k++) wstar[k]=runif();
    for(k=0;k<K;k++) nrm+=wstar[k]*wstar[k];
    nrm=sqrt(nrm); if(nrm<1e-9) nrm=1.0;
    for(k=0;k<K;k++) wstar[k]*=2.0/nrm;
    for(i=0;i<g_n;i++) perm[i]=i;                       /* choose the trained positions at random */
    for(i=g_n-1;i>0;i--){ j=(int)(r32()%(uint32_t)(i+1)); tmp=perm[i];perm[i]=perm[j];perm[j]=tmp; }
    g_ntp = g_ntrainpos;
    for(i=0;i<g_ntp;i++) g_trainpos[i]=perm[i];
    g_nvp = g_ntp + g_nvalpos; if(g_nvp > g_n-2) g_nvp = g_n-2;   /* leave at least 2 held out */
    for(i=0;i<g_nvp;i++) g_valpos[i]=perm[i];
    for(i=g_nvp;i<g_n;i++) held[nh++]=perm[i];
    gen_at(Xtr,ytr,NTR,g_trainpos,g_ntp);               /* weights see only the trained positions */
    gen_at(Xval,yval,NVAL,g_valpos,g_nvp);              /* selection sees these */
    gen_at(Xte,yte,NTE,held,nh);                        /* the outcome: seen by neither */
}

static void geno_stats(const Geno *z,int *places,int *groups)
{
    char seen[NMAX]; int p,np=0,ng=0; memset(seen,0,sizeof seen);
    for(p=0;p<g_n;p++) if(z->on[p]){ np++; if(!seen[z->g[p]]){ seen[z->g[p]]=1; ng++; } }
    *places=np; *groups=ng;
}
static double energy(const Geno *z)
{ int np,ng; double t=g_alpha+g_beta; geno_stats(z,&np,&ng);
  return t>0 ? (g_alpha*ng + g_beta*np)/(t*g_n) : 0.0; }

/* Train on the trained positions; report accuracy on the HELD-OUT positions when on_test. */
static double run_net(const Geno *z, uint32_t seed, int on_test)
{
    static double w[NMAX][K], bh[NMAX], v[NMAX], h[NMAX];
    double bo=0; int i,k,p,e,s,c=0, ns = on_test?NTE:NVAL;
    double (*Xe)[NMAX] = on_test?Xte:Xval; int *ye = on_test?yte:yval;
    wseed(seed);
    for(p=0;p<g_n;p++){ for(k=0;k<K;k++) w[p][k]=0.1*wunif(); bh[p]=0; v[p]=0.1*wunif(); h[p]=0; }
    /* DEVELOPMENT: run until the training loss stops improving, then stop. The budget is therefore a
     * measured property of this genome on this task, not a constant carried over from another setting.
     * Stopping is on TRAINING loss, never on the validation signal fitness reads: stopping on the
     * selection signal would hand each genome its best-over-epochs score, an optimistic bias of a size
     * that differs between genomes. Here development simply runs to completion and fitness is read once. */
    { double best=-1.0; int stale=0;
    for(e=0;e<g_epmax;e++){ double loss=0;
      for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout;
        double dw[NMAX][K], dv[NMAX], dbh[NMAX]; int amax[NMAX]; char used[NMAX];
        for(p=0;p<g_n;p++){ amax[p]=-1; used[p]=0; }
        for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]]; int gp=z->g[p];
            for(k=0;k<K;k++) pre += w[gp][k]*x[(p+k)%g_n];
            h[p]=tanh(pre);
            if(amax[gp]<0 || h[p]>h[amax[gp]]) amax[gp]=p;
            used[gp]=1; }
        for(p=0;p<g_n;p++) if(used[p] && amax[p]>=0) opre += v[p]*h[amax[p]];
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        memset(dw,0,sizeof dw); memset(dv,0,sizeof dv); memset(dbh,0,sizeof dbh);
        for(i=0;i<g_n;i++) if(used[i] && amax[i]>=0){ int pm=amax[i];
            double dpre=dout*v[i]*(1.0-h[pm]*h[pm]);
            dv[i]+=dout*h[pm]; dbh[i]+=dpre;
            for(k=0;k<K;k++) dw[i][k] += dpre*x[(pm+k)%g_n]; }
        bo-=LR*dout;
        for(i=0;i<g_n;i++){ v[i]-=LR*dv[i]; bh[i]-=LR*dbh[i];
            for(k=0;k<K;k++) w[i][k]-=LR*dw[i][k]; }
        if((o>0.5)==(ytr[s]==1)) loss += 1.0;      /* training ACCURACY, not loss */
      }
      loss/=NTR;
      /* Training loss has no plateau: on a small set the network drives cross-entropy down
       * indefinitely by memorising, so a loss criterion never terminates (measured: 357 epochs mean,
       * cap hit 84% of the time). Training accuracy is bounded above and must saturate, so it does
       * plateau, and the plateau is what "development has finished" means here. */
      if(loss > best + g_eps){ best=loss; stale=0; } else stale++;
      if(stale>=g_patience) break;
    }
    g_ep_sum += e; g_ep_n++; if(e>=g_epmax) g_ep_cap++; }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o,hm[NMAX]; char us[NMAX]; int gg;
        memset(us,0,sizeof us);
        for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]], hh; int gp=z->g[p];
            for(k=0;k<K;k++) pre += w[gp][k]*x[(p+k)%g_n];
            hh=tanh(pre); if(!us[gp] || hh>hm[gp]){ hm[gp]=hh; us[gp]=1; } }
        for(gg=0;gg<g_n;gg++) if(us[gg]) opre += v[gg]*hm[gg];
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    return (double)c/ns;
}
/* Develop the same genome from g_restart independent weight initialisations and average. With
 * g_restart=1 fitness is a single noisy draw, and hard elitism then selects partly on which genome got
 * a lucky initialisation. */
static double run_net_avg(const Geno *z, uint32_t seed, int on_test)
{
    int r; double a=0;
    for(r=0;r<g_restart;r++) a += run_net(z, seed + (uint32_t)r*2654435761u, on_test);
    return a/g_restart;
}

static double objective(const Geno *z,double acc)
{ return (acc>=g_target) ? (2.0 - energy(z)) : acc; }

static void seed_disordered(Geno *z)
{ int p; for(p=0;p<g_n;p++){ z->on[p]=(rprob()<0.5); z->g[p]=(unsigned char)(r32()%(uint32_t)g_n); } }

/* One structural operator, selected by g_op, all firing at g_oprate so the arms are rate-matched.
 * 1 is the claim; 2 and 3 are its null twins, differing only in what they copy. */
static void structural_op(Geno *z)
{
    int act[NMAX],na=0,emp[NMAX],ne=0,p,i,d=0,q,t,T;
    for(p=0;p<g_n;p++){ if(z->on[p]) act[na++]=p; else emp[ne++]=p; }
    if(g_op==4){ if(na==0) return;                       /* global: repeat the first T slots */
        T = 1 + (int)(r32()%(uint32_t)(g_n/2));
        for(p=T;p<g_n;p++){ z->on[p]=z->on[p%T]; z->g[p]=z->g[p%T]; }
        return; }
    if(na==0 || ne==0) return;
    if(g_op==2){ q=emp[(int)(r32()%(uint32_t)ne)]; z->on[q]=1;   /* null: copies nothing */
                 z->g[q]=(unsigned char)(r32()%(uint32_t)g_n); return; }
    if(g_op==3){ q=emp[(int)(r32()%(uint32_t)ne)]; z->on[q]=1;   /* null: filter, not spacing */
                 z->g[q]=z->g[act[(int)(r32()%(uint32_t)na)]]; return; }
    if(g_op==5){                    /* filter AND spacing, but addition-rate matched to op3 */
        int i2,d2=0,q2,t2;
        p = act[(int)(r32()%(uint32_t)na)];
        for(i2=1;i2<=g_n/2;i2++){ int a=(p+i2)%g_n,b=(p-i2+g_n)%g_n;
            if(z->on[a] && z->g[a]==z->g[p]){ d2=i2; break; }
            if(z->on[b] && z->g[b]==z->g[p]){ d2=i2; break; } }
        if(d2==0) d2 = 1 + (int)(r32()%(uint32_t)K);
        for(t2=0;t2<2;t2++){ q2=(t2==0)?(p+d2)%g_n:(p-d2+g_n)%g_n;
            if(!z->on[q2]){ z->on[q2]=1; z->g[q2]=z->g[p]; return; } }
        q2=emp[(int)(r32()%(uint32_t)ne)];        /* fall back rather than give up */
        z->on[q2]=1; z->g[q2]=z->g[p]; return; }
    p = act[(int)(r32()%(uint32_t)na)];                          /* 1: filter AND spacing */
    for(i=1;i<=g_n/2;i++){ int a=(p+i)%g_n,b=(p-i+g_n)%g_n;
        if(z->on[a] && z->g[a]==z->g[p]){ d=i; break; }
        if(z->on[b] && z->g[b]==z->g[p]){ d=i; break; } }
    if(d==0) d = 1 + (int)(r32()%(uint32_t)K);
    for(t=0;t<2;t++){ q=(t==0)?(p+d)%g_n:(p-d+g_n)%g_n;
        if(!z->on[q]){ z->on[q]=1; z->g[q]=z->g[p]; return; } }
}

static void mutate(Geno *z)
{
    int p,q,act[NMAX],na=0;
    for(p=0;p<g_n;p++) if(z->on[p]) act[na++]=p;
    for(p=0;p<g_n;p++){
        if(z->on[p]){ if(rprob()<g_prem) z->on[p]=0; }
        else        { if(rprob()<g_padd){ z->on[p]=1; z->g[p]=(unsigned char)(r32()%(uint32_t)g_n); } }
        if(na>0 && rprob()<g_pmerge){ q=act[(int)(r32()%(uint32_t)na)]; z->g[p]=z->g[q]; }
        if(rprob()<g_psplit) z->g[p]=(unsigned char)(r32()%(uint32_t)g_n);
    }
    if(g_op && rprob()<g_oprate) structural_op(z);
}

/* out += {heldout acc, places, groups} and their squares in [3..5] */
static void run_ga(uint32_t seed, double out[8])
{
    static Geno pop[POP], nxt[POP];
    double fit[POP]; int idx[POP],g,p,q;
    rseed(seed);
    for(p=0;p<POP;p++) seed_disordered(&pop[p]);
    for(p=0;p<POP;p++) fit[p]=objective(&pop[p],run_net_avg(&pop[p],(uint32_t)(seed+p*2654435761u+1u),0));
    for(g=0;g<g_gens;g++){
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        for(p=0;p<ELITE;p++) nxt[p]=pop[idx[p]];
        for(p=ELITE;p<POP;p++){ nxt[p]=pop[idx[(int)(r32()%ELITE)]]; mutate(&nxt[p]); }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++) fit[p]=objective(&pop[p],run_net_avg(&pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0));
    }
    { int bi=0,np,ng; double bf=-1,acc; for(p=0;p<POP;p++) if(fit[p]>bf){bf=fit[p];bi=p;}
      geno_stats(&pop[bi],&np,&ng);
      acc=run_net(&pop[bi],seed+999u,1);
      if(g_seeddump) printf("SEED %u acc %.6f places %d groups %d\n",(unsigned)seed,acc,np,ng);
      out[0]+=acc; out[3]+=acc*acc;
      out[1]+=np;  out[4]+=(double)np*np;
      out[2]+=ng;  out[5]+=(double)ng*ng; }
}

/* PROTOCOL CHECK 1 (see ~/articles/smbpann2/PROTOCOL.md).
 * Build genomes spanning the full range of the OUTCOME and confirm FITNESS actually varies over it.
 * If fitness is flat, nothing the search does can be credited for the outcome, and every operator that
 * moves the free variable will look like a discovery. Two complete probes died on exactly this. */
static int sensitivity_check(int seeds)
{
    int m,sd,k; double fmin=1e30,fmax=-1e30,amin=1e30,amax=-1e30;
    double gmin=1e30,gmax=-1e30,ga_min=1e30,ga_max=-1e30; int ngood=0;
    printf("  CHECK 1: does fitness vary over the outcome range?\n");
    printf("    %-8s %-16s %-16s\n","places","held-out acc","fitness");
    for(m=1;m<=g_n;m+=(g_n>6?3:1)){
        Geno z; double a=0,f=0;
        for(k=0;k<g_n;k++){ z.on[k]=(k<m); z.g[k]=0; }     /* one shared filter, m places */
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1));
            { double acc_v=run_net_avg(&z,(uint32_t)(sd*7+1),0);
              a+=run_net(&z,(uint32_t)(sd*7+1)+999u,1); f+=objective(&z,acc_v); } }
        a/=seeds; f/=seeds;
        printf("    %-8d %-16.3f %-16.5f%s\n",m,a,f, f>=1.0?"   (clears target)":"");
        if(f>=1.0){                      /* inside the feasible region, where selection operates */
            ngood++;
            if(f<gmin) gmin=f;
            if(f>gmax) gmax=f;
            if(a<ga_min) ga_min=a;
            if(a>ga_max) ga_max=a; }
        if(f<fmin) fmin=f;
        if(f>fmax) fmax=f;
        if(a<amin) amin=a;
        if(a>amax) amax=a;
    }
    printf("    overall:  outcome spans %.3f   fitness spans %.5f\n", amax-amin, fmax-fmin);
    /* The decisive test is INSIDE the feasible region. A threshold objective varies while genomes are
     * still failing the bar, then goes flat once they clear it. Selection spends its time in the flat
     * part, so variation below the bar does not rescue a flat objective above it. */
    if(ngood >= 2){
        printf("    feasible: outcome spans %.3f   fitness spans %.5f\n", ga_max-ga_min, gmax-gmin);
        if(gmax-gmin < 0.02*(ga_max-ga_min)){
            printf("    *** FAILS: among genomes that clear the target, fitness is FLAT while the\n");
            printf("    *** outcome varies by %.3f. The objective cannot see what is being measured,\n", ga_max-ga_min);
            printf("    *** so no arm difference can be attributed to the search. See PROTOCOL.md item 1.\n\n");
            return 0; }
    } else {
        printf("    (fewer than two feasible points; widen the sweep or lower the target)\n");
    }
    if(fmax-fmin < 0.02*(amax-amin)){
        printf("    *** FAILS: fitness is flat over the outcome.\n\n");
        return 0; }
    printf("    passes.\n\n");
    return 1;
}

int main(void)
{
    int seeds=envint("SEEDS",50), sd, k;
    g_n=envint("NIN",12); if(g_n>NMAX) g_n=NMAX;
    g_ntrainpos=envint("NTRAINPOS",2); g_nvalpos=envint("NVALPOS",0); g_epmax=envint("EPMAX",400); g_patience=envint("PATIENCE",20); g_eps=envdbl("EPS",0.005); g_restart=envint("RESTART",1); g_seeddump=envint("SEEDDUMP",0);
    g_gens=envint("GENS",80); g_target=envdbl("TARGET",0.90);
    g_prem=envdbl("PREM",0.030); g_padd=envdbl("PADD",0.006);
    g_pmerge=envdbl("PMERGE",0.060); g_psplit=envdbl("PSPLIT",0.010);
    g_oprate=envdbl("OPRATE",0.30); g_op=envint("OP",0);
    g_alpha=envdbl("ALPHA",1.0); g_beta=envdbl("BETA",0.0);

    printf("TRANSLATION GENERALIZATION: weights trained on %d of %d positions;\n", g_ntrainpos, g_n);
    printf("  FITNESS computed on %d positions (%d beyond training); held-out = the rest.\n",
           g_ntrainpos+g_nvalpos, g_nvalpos);
    printf("N=%d K=%d, %d seeds x %d gens, target %.2f, energy = (%.1f*groups + %.1f*places)/%d\n",
           g_n, K, seeds, g_gens, g_target, g_alpha, g_beta, g_n);
    printf("development: early stopping (patience %d, eps %.4f, cap %d), %d restart(s) averaged\n",
           g_patience, g_eps, g_epmax, g_restart);
    printf("rates: prem=%.4f padd=%.4f pmerge=%.4f psplit=%.4f  op=%d @ %.2f\n\n",
           g_prem, g_padd, g_pmerge, g_psplit, g_op, g_oprate);

    /* STEP ZERO: do the two reference networks separate on this measure at all?
     * If a hand-built convolution and a locally-connected net score the same on held-out positions,
     * the measure is dead and nothing after it means anything. */
    { Geno conv, lc, half; double ac=0,al=0,ah=0,ac2=0,al2=0; int p;
      for(p=0;p<g_n;p++){ conv.on[p]=1; conv.g[p]=0;
                          lc.on[p]=1;   lc.g[p]=(unsigned char)p;
                          half.on[p]=(p%2==0); half.g[p]=0; }
      for(sd=1;sd<=seeds;sd++){ double a,b; new_task((uint32_t)(sd*131+1));
          a=run_net(&conv,(uint32_t)(sd*7+1)+999u,1); ac+=a; ac2+=a*a;
          b=run_net(&lc,(uint32_t)(sd*7+1)+999u,1);   al+=b; al2+=b*b;
          if(g_seeddump) printf("REF %d conv %.6f lc %.6f\n", sd, a, b);
          ah+=run_net(&half,(uint32_t)(sd*7+1)+999u,1); }
      ac/=seeds; al/=seeds; ah/=seeds;
      printf("  REFERENCE (held-out accuracy)\n");
      printf("    convolution, one filter everywhere : %.3f +- %.3f   <- ceiling\n",
             ac, sqrt(ac2/seeds-ac*ac>0?ac2/seeds-ac*ac:0));
      printf("    locally connected, own filter each : %.3f +- %.3f   <- floor (chance = 0.500)\n",
             al, sqrt(al2/seeds-al*al>0?al2/seeds-al*al:0));
      printf("    one filter, every OTHER position   : %.3f          <- sharing without full coverage\n\n", ah);
      if(ac - al < 0.10){
          printf("  MEASURE FAILS: the references do not separate. Stop here.\n");
          return 1; }
      printf("  measure separates by %.3f; proceeding.\n\n", ac-al); }

    if(!sensitivity_check(seeds >= 30 ? 30 : seeds)) return 2;

    /* PROTOCOL CHECK 2: does evolution beat no evolution? If GENS=0 matches the evolved arms, the
     * headline is the initial population's number and selection is contributing nothing. */
    { int save=g_gens; double o[8]={0}; double m;
      g_gens=0;
      for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga((uint32_t)(sd*7+1),o); }
      for(k=0;k<8;k++) o[k]/=seeds;
      m=o[0];
      printf("  CHECK 2: GENS=0 baseline (best of the initial population, no selection)\n");
      printf("    held-out %.3f   places %.1f   <- the evolved arms must beat this\n\n", m, o[1]);
      g_gens=save; }

    g_ep_sum=0; g_ep_n=0; g_ep_cap=0;
    { double o[8]={0}; double m,s2;
      printf("  EVOLVED (op=%d)\n", g_op);
      for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_ga((uint32_t)(sd*7+1),o); }
      for(k=0;k<8;k++) o[k]/=seeds;
      m=o[0]; s2=o[3]-m*m;
      printf("    held-out accuracy : %.3f +- %.3f\n", m, s2>0?sqrt(s2):0.0);
      m=o[1]; s2=o[4]-m*m; printf("    places            : %.1f +- %.1f\n", m, s2>0?sqrt(s2):0.0);
      m=o[2]; s2=o[5]-m*m; printf("    groups            : %.1f +- %.1f\n", m, s2>0?sqrt(s2):0.0);
      printf("    epochs used       : %.1f mean   (cap hit in %.2f%% of developments)\n",
             g_ep_n? (double)g_ep_sum/g_ep_n : 0.0, g_ep_n? 100.0*g_ep_cap/g_ep_n : 0.0); }
    return 0;
}
