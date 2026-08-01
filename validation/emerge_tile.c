/* emerge_tile.c -- DOES THE TILING EMERGE, AND WHAT MUTATION REACHES IT?
 *
 * Paper 1 (emerge_offset.c) found the compact FILTER emerges only once mutation acts on the shared
 * offset instead of on one edge: the GRANULARITY OF MUTATION decides what can emerge. But it bought
 * that with an offset-mask genome, and said so in its own header: "making the genome an offset mask
 * also makes the connectivity translation-invariant by construction". So the TILING was imposed.
 * This probe removes the imposition and asks the same granularity question one level up.
 *
 * CIRCULAR DOMAIN. Positions wrap, so P = N and every position is a valid window start. This matters:
 * the equivariance theorem is stated for the CYCLIC group, and on a finite non-circular strip even a
 * hand-built convolution is not equivariant (edge positions sit under fewer overlapping windows than
 * interior ones). A first non-circular run measured exactly that: the hand-built convolution scored
 * WORSE equivariance error (0.278) than an evolved net (0.237). The measure was right; the domain was
 * wrong. On the circle the hand-built convolution should score ~0, which is the check that makes the
 * equivariance column mean anything at all.
 *
 * GENOME over P positions:  on[p] = is there a unit reading the K-window at p?
 *                           g[p]  = which weight group it draws its taps from (0..P-1: each CAN be its own)
 * Exhaustive seed = locally connected: every position occupied, every position its own filter.
 * Nothing in the genome encodes translation.
 *
 * TWO ENERGY TERMS (the point). Paper 1's mechanism was that sharing DECOUPLES connection count from
 * parameter count. Pushed forward, the two halves of a convolution are charged by different costs:
 *   E_param = groups/P  -- pays per distinct weight. Sharing dear, placement free.
 *   E_conn  = places/P  -- pays per connection.     Placement dear, sharing free.
 * Both are per-neuron fractions, so the budget scales with the network instead of punishing size.
 *
 * THE MACRO-MUTATION. Per-position edits may be unable to build a tiling at all, because the
 * intermediate states are not fitter: half a tiling costs energy and buys nothing. Biology does not
 * build repeated structure one limb at a time either; it duplicates a segment and repeats it (serial
 * homology, the mechanism behind a millipede). So we add a rare GLOBAL operator: pick a period T and
 * repeat the genome's first T slots across the whole range. It confers PERIODICITY, not a convolution
 * -- the search still has to find the period and the filter, and selection decides whether to keep it.
 * Fired two ways: at a fixed low rate, or on STAGNATION (best fitness flat for a while), which is the
 * more biologically apt trigger.
 *
 * Build: make emerge_tile   Env: SEEDS GENS TARGET NIN
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NMAX  24
#define K     3
#define NTR   24          /* scarce: paper 1 Sec 3.3, the sharing advantage grows as data thins */
#define NVAL  300
#define NTE   1000
#define NPOSS 100         /* per-position probes for the equivariance measure */
#define TEPOCHS 50
#define LR    0.1
#define POP   24
#define ELITE 4

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static double rprob(void){return (double)r32()/4294967296.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int    g_n = 12;                 /* input size; on the circle P == g_n */
static double wstar[K];
static double Xtr[NTR][NMAX], Xval[NVAL][NMAX], Xte[NTE][NMAX];
static int    ytr[NTR], yval[NVAL], yte[NTE];
static double Xpos[NMAX][NPOSS][NMAX];  /* motif planted at EXACTLY position p */
static double g_target=0.90, g_prem=0.030, g_padd=0.006, g_pmerge=0.060, g_psplit=0.010;
static double g_alpha=1.0, g_beta=0.0;
static int    g_nullop=0;   /* 1 = random position + random filter; 2 = random position + copied filter */
static double g_ptile=0.0;              /* per-offspring rate of the tiling macro-mutation */
static double g_pgrow=0.0;  /* rate of the local crystal-growth operator */
static int    g_cap=0;      /* max units; 0 = no cap beyond the genome length */
static int    g_pool=0;     /* 0 = sum-pool within a group; 1 = max-pool (what a ConvNet does) */
static int    g_obj=0;      /* 0 = threshold objective; 1 = continuous acc - lambda*energy */
static double g_lambda=0.10;
static int    g_tprior=0;   /* 0 = uniform period; 1 = P(T) ~ 1/T */
static int    g_stag=0;                 /* if >0, fire tiling after this many generations without gain */
static int    g_gens=80;
static int    envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}
static double envdbl(const char*k,double d){const char*e=getenv(k);return e&&*e?atof(e):d;}

typedef struct { char on[NMAX]; unsigned char g[NMAX]; } Geno;

/* DETECTION task on the circle: a K-tap motif planted at a uniformly random position, or absent.
 * A unit sees the motif only if its window covers that position, so coverage bounds accuracy. */
static void gen(double X[][NMAX],int*y,int n)
{
    int s,i,k,pos;
    for(s=0;s<n;s++){
        for(i=0;i<g_n;i++) X[s][i]=0.30*runif();
        y[s]=(rprob()<0.5);
        if(y[s]){ pos=(int)(r32()%(uint32_t)g_n); for(k=0;k<K;k++) X[s][(pos+k)%g_n]+=wstar[k]; }
    }
}
static void new_task(uint32_t seed)
{
    int k,p,s,i; double nrm=0;
    rseed(seed);
    for(k=0;k<K;k++) wstar[k]=runif();
    for(k=0;k<K;k++) nrm+=wstar[k]*wstar[k];
    nrm=sqrt(nrm); if(nrm<1e-9) nrm=1.0;
    for(k=0;k<K;k++) wstar[k]*=2.0/nrm;
    gen(Xtr,ytr,NTR); gen(Xval,yval,NVAL); gen(Xte,yte,NTE);
    for(p=0;p<g_n;p++) for(s=0;s<NPOSS;s++){
        for(i=0;i<g_n;i++) Xpos[p][s][i]=0.30*runif();
        for(i=0;i<K;i++)   Xpos[p][s][(p+i)%g_n]+=wstar[i]; }
}

/* A CONVOLUTION here: one shared filter, placements on a regular (circular) stride, and no uncovered
 * arc wider than the kernel. This admits a STRIDED convolution, which is still a convolution. */
#define IS_CONV(np,ng,r,mg) (((ng)==1 && (r)>=0.999 && (mg)<=K && (np)>=3) ? 1.0 : 0.0)

static void geno_stats(const Geno *z, int *places, int *groups, double *reg, int *maxgap)
{
    char seen[NMAX]; int p,q,np=0,ng=0,pos[NMAX],gaps[NMAX],ngap=0,bestc=0,mg=0;
    memset(seen,0,sizeof seen);
    for(p=0;p<g_n;p++) if(z->on[p]){ pos[np++]=p; if(!seen[z->g[p]]){ seen[z->g[p]]=1; ng++; } }
    *places=np; *groups=ng;
    if(np==0){ *maxgap=g_n; *reg=0.0; return; }
    for(p=1;p<np;p++){ gaps[ngap]=pos[p]-pos[p-1]; if(gaps[ngap]>mg) mg=gaps[ngap]; ngap++; }
    gaps[ngap]=pos[0]+g_n-pos[np-1]; if(gaps[ngap]>mg) mg=gaps[ngap]; ngap++;  /* the wrap gap */
    *maxgap=mg;
    if(np<3){ *reg=0.0; return; }
    for(p=0;p<ngap;p++){ int c=0,q2; for(q2=0;q2<ngap;q2++) if(gaps[q2]==gaps[p]) c++; if(c>bestc) bestc=c; }
    (void)q; *reg=(double)bestc/ngap;
}
/* Longest run of consecutive occupied positions sharing one filter: the size of the largest ordered
 * domain, and the natural order parameter for a nucleation measurement. Circular. */
static int longest_run(const Geno *z)
{
    int p,best=0,cur=0,start=-1,i,np=0;
    for(p=0;p<g_n;p++) if(z->on[p]) np++;
    if(np==0) return 0;
    if(np==g_n){                     /* fully occupied: the run is bounded by GROUP identity, not by
                                      * occupancy. Returning g_n here scored a locally-connected net
                                      * (12 distinct filters) as a perfect domain of 12. */
        int bestf=0,curf=1;
        for(i=1;i<=g_n;i++){ p=i%g_n;
            if(z->g[p]==z->g[(p-1+g_n)%g_n]) curf++; else { if(curf>bestf) bestf=curf; curf=1; } }
        if(curf>bestf) bestf=curf;
        return bestf>g_n ? g_n : bestf; }
    for(p=0;p<g_n;p++) if(!z->on[p]){ start=p; break; }
    if(start<0) return g_n;
    for(i=1;i<=g_n;i++){ p=(start+i)%g_n;
        if(z->on[p] && (cur==0 || z->g[p]==z->g[(p-1+g_n)%g_n])) cur++;
        else { if(cur>best) best=cur; cur = z->on[p] ? 1 : 0; } }
    if(cur>best) best=cur;
    return best;
}

static double energy(const Geno *z)
{
    int np,ng,mg; double r; geno_stats(z,&np,&ng,&r,&mg);
    return (g_alpha*ng + g_beta*np) / ((g_alpha+g_beta)*g_n);
}

/* EQUIVARIANCE, measured functionally rather than structurally. The theorem (Kondor & Trivedi 2018;
 * Cohen & Welling 2016) says an equivariant map IS a convolution, so we test the property instead of
 * inspecting the wiring: a translation-equivariant detector fires at the same rate wherever the motif
 * sits. Plant it at each position in turn, take the spread of the detection rate. 0 = equivariant.
 * This assumes nothing about the architecture, so it transfers to any network structure. */
static double equiv_err(const Geno *z, double w[NMAX][K], const double *bh, const double *v, double bo)
{
    double d[NMAX], mu=0, var=0; int p,q,s,k;
    for(p=0;p<g_n;p++){ int c=0;
        for(s=0;s<NPOSS;s++){ const double *x=Xpos[p][s]; double opre=bo,o;
            if(!g_pool){
                for(q=0;q<g_n;q++) if(z->on[q]){ double pre=bh[z->g[q]];
                    for(k=0;k<K;k++) pre += w[z->g[q]][k]*x[(q+k)%g_n];
                    opre += v[z->g[q]]*tanh(pre); }
            } else { double hm[NMAX]; char us[NMAX]; int gg;
                memset(us,0,sizeof us);
                for(q=0;q<g_n;q++) if(z->on[q]){ double pre=bh[z->g[q]], hh;
                    for(k=0;k<K;k++) pre += w[z->g[q]][k]*x[(q+k)%g_n];
                    hh=tanh(pre); gg=z->g[q];
                    if(!us[gg] || hh>hm[gg]){ hm[gg]=hh; us[gg]=1; } }
                for(gg=0;gg<g_n;gg++) if(us[gg]) opre += v[gg]*hm[gg]; }
            o=1.0/(1.0+exp(-opre)); if(o>0.5) c++; }
        d[p]=(double)c/NPOSS; mu+=d[p]; }
    mu/=g_n;
    for(p=0;p<g_n;p++) var += (d[p]-mu)*(d[p]-mu);
    return sqrt(var/g_n);
}

static double run_net_eq(const Geno *z, uint32_t seed, int on_test, double *eq)
{
    static double w[NMAX][K], bh[NMAX], v[NMAX], h[NMAX];
    double bo=0; int i,k,p,e,s,c=0, ns = on_test?NTE:NVAL;
    double (*Xe)[NMAX] = on_test?Xte:Xval; int *ye = on_test?yte:yval;
    wseed(seed);
    for(p=0;p<g_n;p++){ for(k=0;k<K;k++) w[p][k]=0.1*wunif(); bh[p]=0; v[p]=0.1*wunif(); h[p]=0; }
    for(e=0;e<TEPOCHS;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout;
        double dw[NMAX][K], dv[NMAX], dbh[NMAX];
        int amax[NMAX]; char used[NMAX];
        for(p=0;p<g_n;p++){ amax[p]=-1; used[p]=0; }
        for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]]; int gp=z->g[p];
            for(k=0;k<K;k++) pre += w[gp][k]*x[(p+k)%g_n];
            h[p]=tanh(pre);
            if(!g_pool) opre += v[gp]*h[p];
            else if(amax[gp]<0 || h[p]>h[amax[gp]]) amax[gp]=p;
            used[gp]=1; }
        if(g_pool) for(p=0;p<g_n;p++) if(used[p] && amax[p]>=0) opre += v[p]*h[amax[p]];
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        memset(dw,0,sizeof dw); memset(dv,0,sizeof dv); memset(dbh,0,sizeof dbh);
        /* filter, bias AND readout weight are all shared within a group: that is a channel with
         * sum-pooling, and it is what makes the composed map translation-equivariant on the circle.
         * With a per-position readout (the earlier version) a shared filter still sits under an
         * unshared locally-connected layer, and the hand-built "convolution" is not equivariant. */
        /* sum-pool: every unit in a group gets gradient. max-pool: only the argmax unit does,
         * which is what a ConvNet's pooling layer routes. */
        if(!g_pool){
            for(p=0;p<g_n;p++) if(z->on[p]){ int gp=z->g[p];
                double dpre=dout*v[gp]*(1.0-h[p]*h[p]);
                dv[gp]  += dout*h[p];
                dbh[gp] += dpre;
                for(k=0;k<K;k++) dw[gp][k] += dpre*x[(p+k)%g_n]; }
        } else {
            int gg;
            for(gg=0;gg<g_n;gg++) if(used[gg] && amax[gg]>=0){ int pm=amax[gg];
                double dpre=dout*v[gg]*(1.0-h[pm]*h[pm]);
                dv[gg]  += dout*h[pm];
                dbh[gg] += dpre;
                for(k=0;k<K;k++) dw[gg][k] += dpre*x[(pm+k)%g_n]; } }
        bo-=LR*dout;
        for(i=0;i<g_n;i++){ v[i]-=LR*dv[i]; bh[i]-=LR*dbh[i];
            for(k=0;k<K;k++) w[i][k]-=LR*dw[i][k]; }
    }
    for(s=0;s<ns;s++){ const double *x=Xe[s]; double opre=bo,o;
        if(!g_pool){
            for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]];
                for(k=0;k<K;k++) pre += w[z->g[p]][k]*x[(p+k)%g_n];
                opre += v[z->g[p]]*tanh(pre); }
        } else { double hm[NMAX]; char us[NMAX]; int gg;
            memset(us,0,sizeof us);
            for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]], hh; int gp=z->g[p];
                for(k=0;k<K;k++) pre += w[gp][k]*x[(p+k)%g_n];
                hh=tanh(pre);
                if(!us[gp] || hh>hm[gp]){ hm[gp]=hh; us[gp]=1; } }
            for(gg=0;gg<g_n;gg++) if(us[gg]) opre += v[gg]*hm[gg]; }
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(ye[s]==1)) c++; }
    if(eq) *eq = equiv_err(z,w,bh,v,bo);
    return (double)c/ns;
}
static double run_net(const Geno *z, uint32_t seed, int on_test)
{ return run_net_eq(z,seed,on_test,NULL); }

/* Two objectives. The THRESHOLD form is paper 1's and it has a failure mode this study walked into:
 * if the target outruns what any architecture in the space can reach, the energy branch fires only on
 * validation noise, so the energy term silently stops being a constraint. The CONTINUOUS form has no
 * threshold to go stale -- energy always has a gradient, at every accuracy and every problem size. */
static double objective(const Geno *z, double acc)
{
    if(g_obj) return acc - g_lambda*energy(z);
    return (acc>=g_target) ? (2.0 - energy(z)) : acc;
}

/* Printable genome: '.' = no unit at that position, otherwise a letter naming the weight group.
 * Same letter = same filter. A convolution reads as one letter on an even stride. */
static void geno_str(const Geno *z, char *buf)
{
    int p, n=0, map[NMAX]; char seen[NMAX];
    memset(seen,0,sizeof seen); memset(map,-1,sizeof map);
    for(p=0;p<g_n;p++) if(z->on[p] && map[z->g[p]]<0) map[z->g[p]]=n++;
    for(p=0;p<g_n;p++) buf[p] = z->on[p] ? (char)('a'+map[z->g[p]]%26) : '.';
    buf[g_n]='\0';
}
static int g_dump=0;

/* Two seeds. The LOCALLY-CONNECTED seed occupies every position, which is already a perfect stride-1
 * placement: positional order is present from the start and the search only has to merge filters. That
 * makes it a test of whether order SURVIVES, not whether it emerges. The DISORDERED seed occupies a
 * random subset with random filters, so no positional order exists at generation zero and any that
 * appears has to be built. Emergence claims need the second. */
static int g_seedmode=0;   /* 0 = locally connected, 1 = disordered, 2 = disordered + planted nucleus */
static int g_nuc=0;        /* planted nucleus size, for the nucleation measurement */
static void seed_exhaustive(Geno *z)
{
    int p;
    if(g_seedmode==0){ for(p=0;p<g_n;p++){ z->on[p]=1; z->g[p]=(unsigned char)p; } }
    else { for(p=0;p<g_n;p++){ z->on[p]=(rprob()<0.5); z->g[p]=(unsigned char)(r32()%(uint32_t)g_n); }
           if(g_seedmode==2){                       /* plant an aligned nucleus of g_nuc units */
               for(p=0;p<g_nuc && p<g_n;p++){ z->on[p]=1; z->g[p]=0; } } }
}

/* THE MACRO-MUTATION: serial homology. Pick a period T and repeat the first T slots across the whole
 * genome. One move, global reach; it confers periodicity, not a convolution. */
/* THE OPERATOR CARRIES ITS OWN PRIOR: the distribution it draws the period T from.
 * With a UNIFORM draw over 1..N/2, only T <= K can leave no gap wider than the kernel, so the share of
 * useful draws is ~2K/N and decays as the network grows. That is a candidate explanation for why %conv
 * falls with N while extra generations do nothing. TPRIOR=1 swaps in P(T) proportional to 1/T, which
 * favours short periods without forbidding long ones; the gap between the two arms measures what the
 * prior is worth. Note this does not remove the prior, it relocates it: the search no longer has to
 * find the tiling, but someone still had to choose the distribution over periods. */
static void tile_mutation(Geno *z)
{
    int Tmax = g_n/2, T, p;
    if(Tmax<1) Tmax=1;
    if(g_tprior==0){ T = 1 + (int)(r32()%(uint32_t)Tmax); }
    else { double H=0, u, acc=0; int i;
           for(i=1;i<=Tmax;i++) H += 1.0/i;
           u = rprob()*H; T = Tmax;
           for(i=1;i<=Tmax;i++){ acc += 1.0/i; if(u<=acc){ T=i; break; } } }
    for(p=T;p<g_n;p++){ z->on[p]=z->on[p%T]; z->g[p]=z->g[p%T]; }
}

/* CRYSTAL GROWTH: extend an existing shared structure by ONE unit, at the spacing it already has.
 * This is tandem duplication: local in the genome and incremental, unlike tile_mutation, which
 * rewrites the whole individual in one jump. The distinction matters, because a move can be local in
 * space and still propagate structure rather than randomise it; per-position mutation is local AND
 * undirected, and it is the second property, not the first, that may be what blocks a tiling.
 *
 * Pick an occupied position, measure the spacing to the nearest unit sharing its filter (its lattice
 * constant, chosen at random if it is a lone nucleus), and place one more unit of the same filter at
 * that spacing. Growth is one unit at a time and selection judges each addition, so a tiling can only
 * extend if each extension pays. A cap bounds runaway expansion. */
static void grow_mutation(Geno *z)
{
    int act[NMAX], na=0, p, i, d=0, q, t, cap = g_cap ? g_cap : g_n;
    for(p=0;p<g_n;p++) if(z->on[p]) act[na++]=p;
    if(na==0 || na>=cap) return;
    if(g_nullop){                      /* controls: add one unit at a RANDOM empty position */
        int empt[NMAX], ne=0;
        for(p=0;p<g_n;p++) if(!z->on[p]) empt[ne++]=p;
        if(ne==0) return;
        q = empt[(int)(r32()%(uint32_t)ne)];
        z->on[q]=1;
        z->g[q] = (g_nullop==2) ? z->g[act[(int)(r32()%(uint32_t)na)]]   /* copies filter, not spacing */
                                : (unsigned char)(r32()%(uint32_t)g_n);  /* copies nothing */
        return; }
    p = act[(int)(r32()%(uint32_t)na)];
    for(i=1;i<=g_n/2;i++){
        int a=(p+i)%g_n, b=(p-i+g_n)%g_n;
        if(z->on[a] && z->g[a]==z->g[p]){ d=i; break; }
        if(z->on[b] && z->g[b]==z->g[p]){ d=i; break; }
    }
    if(d==0) d = 1 + (int)(r32()%(uint32_t)K);      /* lone nucleus: choose a lattice constant */
    for(t=0;t<2;t++){
        q = (t==0) ? (p+d)%g_n : (p-d+g_n)%g_n;
        if(!z->on[q]){ z->on[q]=1; z->g[q]=z->g[p]; return; }
    }
}

static void mutate(Geno *z)
{
    int p,q,act[NMAX],na=0;
    for(p=0;p<g_n;p++) if(z->on[p]) act[na++]=p;
    for(p=0;p<g_n;p++){
        if(z->on[p]){ if(rprob()<g_prem) z->on[p]=0; }
        else        { if(rprob()<g_padd) z->on[p]=1; }
        if(na>0 && rprob()<g_pmerge){ q=act[(int)(r32()%(uint32_t)na)]; z->g[p]=z->g[q]; }
        if(rprob()<g_psplit) z->g[p]=(unsigned char)(r32()%(uint32_t)g_n);
    }
}

/* out += {coverage, groups, reg, test, %conv, equiv-err, tilings-fired} */
static void run_ga(uint32_t seed, double out[20], int nevals_out[1])
{
    static Geno pop[POP], nxt[POP];
    double fit[POP]; int idx[POP]; int g,p,q,evals=0,flat=0,fired=0; double prevbest=-1e9;
    rseed(seed);
    for(p=0;p<POP;p++) seed_exhaustive(&pop[p]);
    if(g_dump){ char b[NMAX+1]; geno_str(&pop[0],b); printf("SEED0 %s\n", b); }
    for(p=0;p<POP;p++){ fit[p]=objective(&pop[p],run_net(&pop[p],(uint32_t)(seed+p*2654435761u+1u),0)); evals++; }
    for(g=0;g<g_gens;g++){
        double best; int tilenow=0;
        for(p=0;p<POP;p++) idx[p]=p;
        for(p=0;p<POP;p++) for(q=p+1;q<POP;q++) if(fit[idx[q]]>fit[idx[p]]){int t=idx[p];idx[p]=idx[q];idx[q]=t;}
        best=fit[idx[0]];
        if(best<=prevbest+1e-9) flat++; else { flat=0; prevbest=best; }
        if(g_stag>0 && flat>=g_stag){ tilenow=1; flat=0; }      /* stagnation trigger */
        for(p=0;p<ELITE;p++) nxt[p]=pop[idx[p]];
        for(p=ELITE;p<POP;p++){ nxt[p]=pop[idx[(int)(r32()%ELITE)]]; mutate(&nxt[p]);
            if(tilenow || (g_ptile>0.0 && rprob()<g_ptile)){ tile_mutation(&nxt[p]); fired++; }
            if(g_pgrow>0.0 && rprob()<g_pgrow){ grow_mutation(&nxt[p]); fired++; } }
        memcpy(pop,nxt,sizeof pop);
        for(p=0;p<POP;p++){ fit[p]=objective(&pop[p],run_net(&pop[p],(uint32_t)(seed+(uint32_t)(g*POP+p)+7u),0)); evals++; }
    }
    { int bi=0,np,ng,mg; double bf=-1,r,eq=0; for(p=0;p<POP;p++) if(fit[p]>bf){bf=fit[p];bi=p;}
      geno_stats(&pop[bi],&np,&ng,&r,&mg);
      { double acc=run_net_eq(&pop[bi],seed+999u,1,&eq); double cv=(double)np/g_n;
        int lr=longest_run(&pop[bi]);
        out[0]+=cv;  out[10]+=cv*cv;
        out[1]+=ng;
        out[2]+=r;   out[12]+=r*r;
        out[3]+=acc; out[13]+=acc*acc;
        out[5]+=eq;  out[15]+=eq*eq;
        out[4]+=IS_CONV(np,ng,r,mg); out[6]+=fired;
        out[7]+=((ng==1&&r>=0.90&&mg<=K&&np>=3)?1.0:0.0);   /* relaxed regularity thresholds, to show */
        out[8]+=((ng==1&&r>=0.80&&mg<=K&&np>=3)?1.0:0.0);   /* the ordering does not hinge on 0.999   */
        out[9]+=lr; out[19]+=(double)lr*lr; }
      if(g_dump){ char b[NMAX+1]; geno_str(&pop[bi],b);
          printf("GENOME %s conv=%d groups=%d places=%d\n", b, (int)IS_CONV(np,ng,r,mg), ng, np); } }
    nevals_out[0]=evals;
}

static void run_rand(uint32_t seed, int evals, double out[20])
{
    Geno z, best; double bf=-1,r,eq=0; int e,p,np,ng,mg;
    rseed(seed); seed_exhaustive(&best);
    for(e=0;e<evals;e++){ double f;
        for(p=0;p<g_n;p++){ z.on[p]=(rprob()<0.5); z.g[p]=(unsigned char)(r32()%(uint32_t)g_n); }
        f=objective(&z,run_net(&z,(uint32_t)(seed+(uint32_t)e*2654435761u+3u),0));
        if(f>bf){ bf=f; best=z; } }
    geno_stats(&best,&np,&ng,&r,&mg);
    { double acc=run_net_eq(&best,seed+999u,1,&eq); double cv=(double)np/g_n;
      out[0]+=cv; out[10]+=cv*cv; out[1]+=ng; out[2]+=r; out[12]+=r*r;
      out[3]+=acc; out[13]+=acc*acc; out[5]+=eq; out[15]+=eq*eq;
      out[4]+=IS_CONV(np,ng,r,mg); out[9]+=longest_run(&best); }
}

static int g_fast=0;   /* FAST=1: only the E_param arm, which is the one the operator pays under */

static void sweep(const char *label, int seeds, int stag, double ptile, double pgrow, int ev[1])
{
    struct { const char *name; double al, be; } arm[3] = {
        {"E_param (per group)",     1.0, 0.0},
        {"E_conn  (per placement)", 0.0, 1.0},
        {"both",                    1.0, 1.0} };
    int a,sd2;
    printf("\n%s\n", label);
    printf("  %-26s  cover        reg          test         eq-err       %%conv (reg>=.999/.90/.80)  run\n",
           "energy term");
    for(a=0;a<(g_fast?1:3);a++){
        double o[20]={0}, sd[20];
        int k2;
        g_alpha=arm[a].al; g_beta=arm[a].be; g_stag=stag; g_ptile=ptile; g_pgrow=pgrow;
        if(g_dump) printf("ARM %s | %s\n", label, arm[a].name);
        for(sd2=1;sd2<=seeds;sd2++){ new_task((uint32_t)(sd2*131+1)); run_ga((uint32_t)(sd2*7+1),o,ev); }
        for(k2=0;k2<20;k2++) o[k2]/=seeds;
        /* SD from the accumulated squares: index i holds the mean, i+10 the mean square */
        for(k2=0;k2<10;k2++){ double v=o[k2+10]-o[k2]*o[k2]; sd[k2]= v>0 ? sqrt(v) : 0.0; }
        printf("  %-26s  %.2f+-%.2f  %.2f+-%.2f  %.3f+-%.3f  %.3f+-%.3f   %3.0f%% %3.0f%% %3.0f%%        %.1f+-%.1f\n",
               arm[a].name, o[0], sd[0], o[2], sd[2], o[3], sd[3], o[5], sd[5],
               100.0*o[4], 100.0*o[7], 100.0*o[8], o[9], sd[9]);
    }
}


int main(void)
{
    int seeds=envint("SEEDS",16), sd, ev[1]={0};
    g_n=envint("NIN",12); if(g_n>NMAX) g_n=NMAX;
    g_gens=envint("GENS",80); g_target=envdbl("TARGET",0.90); g_dump=envint("DUMP",0); g_fast=envint("FAST",0); g_tprior=envint("TPRIOR",0); g_obj=envint("OBJ",0); g_pool=envint("POOL",0); g_pgrow=envdbl("PGROW",0.0); g_prem=envdbl("PREM",0.030); g_padd=envdbl("PADD",0.006); g_nullop=envint("NULLOP",0); g_cap=envint("CAP",0); g_seedmode=envint("SEEDMODE",0); g_nuc=envint("NUCLEUS",0); g_lambda=envdbl("LAMBDA",0.10);

    printf("DOES THE TILING EMERGE?  CIRCULAR domain, exhaustive seed = locally connected.\n");
    printf("N=P=%d K=%d, target acc >= %.2f, %d seeds x %d gens, prior=%s, prem=%.4f.\n",
           g_n, K, g_target, seeds, g_gens, g_tprior?"1/T":"uniform", g_prem);
    printf("SEED: %s\n", g_seedmode?"DISORDERED (random subset, random filters)":"locally connected (every position)");
    if(g_pool) printf("POOLING: max within group (ConvNet-style).\n");
    if(g_obj) printf("OBJECTIVE: continuous, acc - %.3f*energy (no threshold).\n", g_lambda);
    printf("so the budget scales with the network.  A CONVOLUTION = one filter on a regular stride.\n\n");

    printf("  %-26s  cover  groups  reg   test   eq-err\n", "reference");
    { Geno conv, lc; double ac=0, al=0, ec=0, el=0; int p;
      for(p=0;p<g_n;p++){ conv.on[p]=1; conv.g[p]=0; lc.on[p]=1; lc.g[p]=(unsigned char)p; }
      for(sd=1;sd<=seeds;sd++){ double e1=0,e2=0; new_task((uint32_t)(sd*131+1));
          ac+=run_net_eq(&conv,(uint32_t)(sd*7+1)+999u,1,&e1); ec+=e1;
          al+=run_net_eq(&lc,(uint32_t)(sd*7+1)+999u,1,&e2);   el+=e2; }
      printf("  %-26s  %.2f   %.1f    %.2f  %.3f  %.3f   (hand-built ceiling)\n",
             "CONVOLUTION",1.0,1.0,1.0,ac/seeds,ec/seeds);
      printf("  %-26s  %.2f   %.1f    %.2f  %.3f  %.3f   (exhaustive seed)\n",
             "locally connected",1.0,(double)g_n,1.0,al/seeds,el/seeds);
      printf("  [CHECK: eq-err for the hand-built convolution must be ~0 on the circle.]\n");
      if(g_dump){ char b[NMAX+1];
          printf("ARM reference | CONVOLUTION\n");       geno_str(&conv,b); printf("GENOME %s conv=1 groups=1 places=%d\n", b, g_n);
          printf("ARM reference | locally connected\n"); geno_str(&lc,b);   printf("GENOME %s conv=0 groups=%d places=%d\n", b, g_n, g_n); } }

    { double ptile=envdbl("PTILE",0.05); int stag=envint("STAG",8); char lab[128];
      sweep("(1) CONTROL: per-position mutation only (paper 1's mutation model)", seeds, 0, 0.0, 0.0, ev);
      sprintf(lab,"(2) TILING macro-mutation at a fixed low rate (p=%.2f per offspring)",ptile);
      sweep(lab, seeds, 0, ptile, 0.0, ev);
      if(!g_fast){ sprintf(lab,"(3) TILING fired on STAGNATION (best fitness flat for %d generations)",stag);
                   sweep(lab, seeds, stag, 0.0, 0.0, ev); }
      sprintf(lab,"(4) CRYSTAL GROWTH: local tandem duplication, one unit at a time (p=%.2f)",
              envdbl("PGROW2",0.05));
      sweep(lab, seeds, 0, 0.0, envdbl("PGROW2",0.05), ev); }

    if(g_fast){ printf("\n[FAST mode: E_param arm only; arm (3) and the random control skipped]\n"); return 0; }
    printf("\n  matched-budget random control (%d evals/seed)\n", ev[0]);
    { double o[20]={0}; g_alpha=1.0; g_beta=1.0; g_stag=0; g_ptile=0.0;
      { int k2; for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); run_rand((uint32_t)(sd*7+1),ev[0],o); }
        for(k2=0;k2<20;k2++) o[k2]/=seeds; }
      printf("  %-26s  %.2f   %.1f    %.2f  %.3f  %.3f   %.0f%%\n",
             "random", o[0], o[1], o[2], o[3], o[5], 100.0*o[4]); }

    printf("\nREAD: if (2) or (3) lifts %%conv above (1), the tiling is unreachable by local mutation\n");
    printf("and reachable by a rare global one -- paper 1's granularity mechanism, one level up.\n");
    return 0;
}
