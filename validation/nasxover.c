/* nasxover.c -- does crossover help on a REAL NAS benchmark, and does additive
 * separability predict it?
 *
 * Reads a NAS-Bench-201 (NATS-Bench topology) accuracy table, one architecture per
 * line: six operation indices (0..4 over the cell's six edges) then the test
 * accuracy on CIFAR-10, CIFAR-100, and ImageNet16-120 (extracted once from the
 * public benchmark; see validation/extract_nasbench.py). All fitness here is a table
 * lookup, so no network is trained. For each dataset we measure two things: the
 * additive-model R^2 (how much of the accuracy variance is explained by a sum of
 * per-edge main effects, i.e. how separable the space is), and the actual benefit of
 * a crossover+mutation GA over a mutation-only GA. The question is whether crossover
 * helps on this real space and whether separability predicts it.
 *
 * Self-contained C99 (own xorshift PRNG).  Build: make nasxover
 * Run: ./nasxover /path/to/nasbench201.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define K   6            /* cell edges (genes) */
#define V   5            /* operations per edge */
#define NA  15625        /* = V^K architectures */
#define NDS 3            /* datasets: cifar10, cifar100, imagenet16 */

static uint32_t rs;
static uint32_t r32(void){ uint32_t x=rs; x^=x<<13; x^=x>>17; x^=x<<5; rs=x; return x; }
static void rseed(uint32_t s){ rs=s?s:1u; }
static uint32_t rbelow(uint32_t m){ return m?r32()%m:0u; }

static double fit[NDS][NA];          /* accuracy per dataset, indexed by base-V code */
static const char *DSN[NDS] = {"cifar10", "cifar100", "imagenet16"};

static int code_of(const int *op){ int c=0,i; for(i=K-1;i>=0;i--) c=c*V+op[i]; return c; }
static void ops_of(int code,int *op){ int i; for(i=0;i<K;i++){ op[i]=code%V; code/=V; } }

static int load(const char *path)
{
    FILE *f=fopen(path,"r"); char line[256]; int n=0;
    if(!f) return -1;
    while(fgets(line,sizeof line,f)){
        int op[K],c; double a[NDS];
        if(line[0]=='#'||line[0]=='\n') continue;
        if(sscanf(line,"%d %d %d %d %d %d %lf %lf %lf",
                  &op[0],&op[1],&op[2],&op[3],&op[4],&op[5],&a[0],&a[1],&a[2])!=9) continue;
        c=code_of(op);
        fit[0][c]=a[0]; fit[1][c]=a[1]; fit[2][c]=a[2]; n++;
    }
    fclose(f);
    return n;
}

/* additive main-effects R^2 over the full space of one dataset */
static double additivity(const double *f)
{
    double mu=0, eff[K][V], cnt[K][V], sstot=0, ssres=0;
    int c,i,v,op[K];
    for(c=0;c<NA;c++) mu+=f[c];
    mu/=NA;
    for(i=0;i<K;i++) for(v=0;v<V;v++){ eff[i][v]=0; cnt[i][v]=0; }
    for(c=0;c<NA;c++){ ops_of(c,op); for(i=0;i<K;i++){ eff[i][op[i]]+=f[c]; cnt[i][op[i]]+=1; } }
    for(i=0;i<K;i++) for(v=0;v<V;v++) eff[i][v] = eff[i][v]/cnt[i][v] - mu;   /* main effect */
    for(c=0;c<NA;c++){
        double pred=mu, e; ops_of(c,op);
        for(i=0;i<K;i++) pred+=eff[i][op[i]];
        e=f[c]-pred; ssres+=e*e;
        e=f[c]-mu;   sstot+=e*e;
    }
    return 1.0 - ssres/sstot;
}

/* one GA run over dataset D; returns the best accuracy found within the budget.
 * pc>0 makes a fraction of offspring by uniform crossover of two elites. */
static double ga(const double *f, int pop, int gens, int elite, double pc, uint32_t seed)
{
    int P[64][K], Q[64][K], i, g, k, best;
    double fp[64];
    rseed(seed);
    for(i=0;i<pop;i++) for(k=0;k<K;k++) P[i][k]=(int)rbelow(V);
    best=0;
    for(g=1;g<=gens;g++){
        int bi=0;
        for(i=0;i<pop;i++){ fp[i]=f[code_of(P[i])]; if(fp[i]>fp[bi]) bi=i; }
        (void)best;
        /* elitism: keep the top by simple selection */
        {   int idx[64]; for(i=0;i<pop;i++) idx[i]=i;
            for(i=0;i<pop;i++) for(k=i+1;k<pop;k++) if(fp[idx[k]]>fp[idx[i]]){int t=idx[i];idx[i]=idx[k];idx[k]=t;}
            for(i=0;i<elite;i++) memcpy(Q[i],P[idx[i]],sizeof P[0]);
            for(i=elite;i<pop;i++){
                if(pc>0.0 && (double)r32()/4294967296.0 < pc){
                    int a=idx[rbelow((uint32_t)elite)], b=idx[rbelow((uint32_t)elite)];
                    for(k=0;k<K;k++) Q[i][k] = (r32()&1u)? P[a][k] : P[b][k];   /* uniform */
                } else {
                    int a=idx[rbelow((uint32_t)elite)];
                    memcpy(Q[i],P[a],sizeof P[0]);
                }
                if(rbelow(100) < 70u) Q[i][rbelow(K)] = (int)rbelow(V);         /* mutate one gene */
            }
            memcpy(P,Q,sizeof P);
        }
    }
    {   double bf=-1; for(i=0;i<pop;i++){ double a=f[code_of(P[i])]; if(a>bf) bf=a; } return bf; }
}

/* --- restricted sub-space: exactly 3 of the 5 ops allowed at each edge (so every
 * sub-space has 3^6 = 729 architectures, same size, varying only WHICH ops), which
 * varies the additivity while controlling for space size. --- */
#define SUB 729                                  /* = 3^6 */

static double additivity_sub(const double *f, int allowed[K][3])
{
    double mu=0, eff[K][3], sstot=0, ssres=0; int j[K],p,q,cnt[K][3];
    int list[SUB]; int m=0;
    for(p=0;p<K;p++) j[p]=0;
    for(;;){ int op[K]; for(p=0;p<K;p++) op[p]=allowed[p][j[p]]; list[m++]=code_of(op);
        for(p=0;p<K;p++){ if(++j[p]<3) break; j[p]=0; } if(p==K) break; }
    for(q=0;q<m;q++) mu+=f[list[q]];
    mu/=m;
    for(p=0;p<K;p++) for(q=0;q<3;q++){ eff[p][q]=0; cnt[p][q]=0; }
    { int idx=0; for(idx=0;idx<m;idx++){ int dig=idx,pp; for(pp=0;pp<K;pp++){ int jj=dig%3; dig/=3;
        eff[pp][jj]+=f[list[idx]]; cnt[pp][jj]++; } } }
    for(p=0;p<K;p++) for(q=0;q<3;q++) eff[p][q]=eff[p][q]/cnt[p][q]-mu;
    { int idx; for(idx=0;idx<m;idx++){ double pred=mu,e; int dig=idx,pp; for(pp=0;pp<K;pp++){ pred+=eff[pp][dig%3]; dig/=3; }
        e=f[list[idx]]-pred; ssres+=e*e; e=f[list[idx]]-mu; sstot+=e*e; } }
    return sstot>0 ? 1.0-ssres/sstot : 0.0;
}

static double ga_sub(const double *f, int allowed[K][3], int pop, int gens, int elite,
                     double pc, uint32_t seed)
{
    int P[64][K], Q[64][K], i,g,k; double fp[64];
    rseed(seed);
    for(i=0;i<pop;i++) for(k=0;k<K;k++) P[i][k]=allowed[k][rbelow(3)];
    for(g=1;g<=gens;g++){
        int idx[64]; for(i=0;i<pop;i++){ fp[i]=f[code_of(P[i])]; idx[i]=i; }
        for(i=0;i<pop;i++) for(k=i+1;k<pop;k++) if(fp[idx[k]]>fp[idx[i]]){int t=idx[i];idx[i]=idx[k];idx[k]=t;}
        for(i=0;i<elite;i++) memcpy(Q[i],P[idx[i]],sizeof P[0]);
        for(i=elite;i<pop;i++){
            if(pc>0.0 && (double)r32()/4294967296.0<pc){ int a=idx[rbelow((uint32_t)elite)],b=idx[rbelow((uint32_t)elite)];
                for(k=0;k<K;k++) Q[i][k]=(r32()&1u)?P[a][k]:P[b][k]; }
            else { int a=idx[rbelow((uint32_t)elite)]; memcpy(Q[i],P[a],sizeof P[0]); }
            if(rbelow(100)<70u){ int pp=rbelow(K); Q[i][pp]=allowed[pp][rbelow(3)]; }
        }
        memcpy(P,Q,sizeof P);
    }
    { double bf=-1; for(i=0;i<pop;i++){ double a=f[code_of(P[i])]; if(a>bf) bf=a; } return bf; }
}

int main(int argc,char**argv)
{
    const char *path = argc>1?argv[1]:"/tmp/claude-1000/nasbench201.txt";
    int n=load(path), d, s, runs=200, pop=12, gens=8, elite=3;
    int M=120;                                   /* sub-spaces per dataset */
    if(n<=0){ fprintf(stderr,"nasxover: cannot read %s\n",path); return 1; }
    printf("loaded %d architectures from %s\n", n, path);
    printf("\nfull space (all 15625 archs), tight budget %d evals:\n", pop*gens);
    printf("%-11s  add.R^2   xover benefit\n","dataset");
    for(d=0;d<NDS;d++){
        double r2=additivity(fit[d]), sm=0, sx=0; int ss;
        for(ss=1;ss<=runs;ss++){ sm+=ga(fit[d],pop,gens,elite,0.0,(uint32_t)ss);
                                 sx+=ga(fit[d],pop,gens,elite,0.9,(uint32_t)ss); }
        printf("%-11s  %.3f     %+.4f\n", DSN[d], r2, (sx-sm)/runs);
    }
    printf("\nHYPOTHESIS TEST: does additivity R^2 predict crossover benefit?\n");
    printf("  %d sub-spaces per dataset (3-of-5 ops per edge, 729 archs each), Pearson r:\n", M);
    for(d=0;d<NDS;d++){
        double sx2=0,sy2=0,sxy=0,sxs=0,sys=0; int m2, mm;
        rseed((uint32_t)(1000+d));
        for(mm=0;mm<M;mm++){
            int allowed[K][3], p, benefit_runs=60; double r2, sm=0, sx=0;
            for(p=0;p<K;p++){ int chosen[V]={0,0,0,0,0}, got=0; while(got<3){ int o=rbelow(V); if(!chosen[o]){chosen[o]=1;allowed[p][got++]=o;} } }
            r2=additivity_sub(fit[d],allowed);
            for(s=1;s<=benefit_runs;s++){ sm+=ga_sub(fit[d],allowed,pop,gens,elite,0.0,(uint32_t)(s*131+mm));
                                          sx+=ga_sub(fit[d],allowed,pop,gens,elite,0.9,(uint32_t)(s*131+mm)); }
            { double x=r2, y=(sx-sm)/benefit_runs; sxs+=x; sys+=y; sx2+=x*x; sy2+=y*y; sxy+=x*y; }
        }
        m2=M;
        { double cov=sxy-sxs*sys/m2, vx=sx2-sxs*sxs/m2, vy=sy2-sys*sys/m2;
          double r=(vx>0&&vy>0)?cov/sqrt(vx*vy):0.0;
          double t=r*sqrt((m2-2)/(1-r*r+1e-12));
          printf("  %-11s  r = %+.3f  (t=%.2f, n=%d)  mean benefit %+.4f\n",
                 DSN[d], r, t, m2, sys/m2); }
    }
    return 0;
}
