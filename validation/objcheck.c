/* objcheck.c -- PROTOCOL CHECK 1, run on candidate objectives BEFORE any search exists.
 *
 * Two complete probes (emerge_tile.c, emerge_gen2.c) produced confident results that were artifacts of
 * one defect: the fitness function could not see the property being measured. With E = groups/N, every
 * single-group genome scores identically whether it has 1 unit or 12, while held-out translation
 * generalization sweeps 0.53 to 0.95 over that same range. Nothing the search did could be credited
 * for the outcome.
 *
 * This probe runs no search. It builds genomes by hand spanning the full range of the OUTCOME, and asks
 * of each candidate OBJECTIVE: does it vary over that range, and in which direction?
 *
 * OUTCOME (functional, cannot be satisfied by saturation): train on a few positions, test on positions
 * never trained on. Coverage plus sharing transfers; per-position detectors do not.
 *
 * CANDIDATE OBJECTIVES, all computed from the genome alone, never from held-out data:
 *   A  groups/N                     what both failed probes used
 *   B  places/N                     connection count
 *   C  placement description length: bits to name the placement, cheap when it is a regular stride
 *   D  total description length:     filter bits + placement bits
 *
 * A cost that cannot separate a good architecture from a bad one is not a cost worth searching under,
 * and no amount of operator design will rescue it.
 *
 * Build: make objcheck    Env: SEEDS NIN NTRAINPOS
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define NMAX 24
#define K    3
#define NTR  96
#define NTE  600
#define EPOCHS 60
#define LR   0.1

static uint32_t rs=1u; static uint32_t r32(void){uint32_t x=rs;x^=x<<13;x^=x>>17;x^=x<<5;rs=x;return x;}
static void rseed(uint32_t s){rs=s?s:1u;} static double runif(void){return (double)r32()/4294967296.0*2.0-1.0;}
static double rprob(void){return (double)r32()/4294967296.0;}
static uint32_t wr=1u; static uint32_t wr32(void){uint32_t x=wr;x^=x<<13;x^=x>>17;x^=x<<5;wr=x;return x;}
static void wseed(uint32_t s){wr=s?s:1u;} static double wunif(void){return (double)wr32()/4294967296.0*2.0-1.0;}

static int g_n=12, g_ntp=2;
static double wstar[K];
static double Xtr[NTR][NMAX], Xte[NTE][NMAX];
static int ytr[NTR], yte[NTE];
static int envint(const char*k,int d){const char*e=getenv(k);return e&&*e?atoi(e):d;}

typedef struct { char on[NMAX]; unsigned char g[NMAX]; } Geno;

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
    int k,i,j,t,perm[NMAX],tr[NMAX],he[NMAX],nh=0; double nrm=0;
    rseed(seed);
    for(k=0;k<K;k++) wstar[k]=runif();
    for(k=0;k<K;k++) nrm+=wstar[k]*wstar[k];
    nrm=sqrt(nrm); if(nrm<1e-9) nrm=1.0;
    for(k=0;k<K;k++) wstar[k]*=2.0/nrm;
    for(i=0;i<g_n;i++) perm[i]=i;
    for(i=g_n-1;i>0;i--){ j=(int)(r32()%(uint32_t)(i+1)); t=perm[i];perm[i]=perm[j];perm[j]=t; }
    for(i=0;i<g_ntp;i++) tr[i]=perm[i];
    for(i=g_ntp;i<g_n;i++) he[nh++]=perm[i];
    gen_at(Xtr,ytr,NTR,tr,g_ntp);
    gen_at(Xte,yte,NTE,he,nh);
}

/* held-out accuracy: the outcome. Shared weights within a group, max-pooled, trained on the
 * trained positions only and evaluated on positions never seen. */
static double outcome(const Geno *z, uint32_t seed)
{
    static double w[NMAX][K], bh[NMAX], v[NMAX], h[NMAX];
    double bo=0; int i,k,p,e,s,c=0;
    wseed(seed);
    for(p=0;p<g_n;p++){ for(k=0;k<K;k++) w[p][k]=0.1*wunif(); bh[p]=0; v[p]=0.1*wunif(); h[p]=0; }
    for(e=0;e<EPOCHS;e++) for(s=0;s<NTR;s++){
        const double *x=Xtr[s]; double opre=bo,o,dout;
        double dw[NMAX][K],dv[NMAX],dbh[NMAX]; int amax[NMAX]; char used[NMAX];
        for(p=0;p<g_n;p++){ amax[p]=-1; used[p]=0; }
        for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]]; int gp=z->g[p];
            for(k=0;k<K;k++) pre += w[gp][k]*x[(p+k)%g_n];
            h[p]=tanh(pre); if(amax[gp]<0||h[p]>h[amax[gp]]) amax[gp]=p; used[gp]=1; }
        for(p=0;p<g_n;p++) if(used[p]&&amax[p]>=0) opre += v[p]*h[amax[p]];
        o=1.0/(1.0+exp(-opre)); dout=(o-ytr[s])*o*(1.0-o);
        memset(dw,0,sizeof dw); memset(dv,0,sizeof dv); memset(dbh,0,sizeof dbh);
        for(i=0;i<g_n;i++) if(used[i]&&amax[i]>=0){ int pm=amax[i];
            double dpre=dout*v[i]*(1.0-h[pm]*h[pm]);
            dv[i]+=dout*h[pm]; dbh[i]+=dpre;
            for(k=0;k<K;k++) dw[i][k]+=dpre*x[(pm+k)%g_n]; }
        bo-=LR*dout;
        for(i=0;i<g_n;i++){ v[i]-=LR*dv[i]; bh[i]-=LR*dbh[i];
            for(k=0;k<K;k++) w[i][k]-=LR*dw[i][k]; }
    }
    for(s=0;s<NTE;s++){ const double *x=Xte[s]; double opre=bo,o,hm[NMAX]; char us[NMAX]; int gg;
        memset(us,0,sizeof us);
        for(p=0;p<g_n;p++) if(z->on[p]){ double pre=bh[z->g[p]],hh; int gp=z->g[p];
            for(k=0;k<K;k++) pre += w[gp][k]*x[(p+k)%g_n];
            hh=tanh(pre); if(!us[gp]||hh>hm[gp]){ hm[gp]=hh; us[gp]=1; } }
        for(gg=0;gg<g_n;gg++) if(us[gg]) opre += v[gg]*hm[gg];
        o=1.0/(1.0+exp(-opre)); if((o>0.5)==(yte[s]==1)) c++; }
    return (double)c/NTE;
}

static void stats(const Geno *z,int *places,int *groups)
{
    char seen[NMAX]; int p,np=0,ng=0; memset(seen,0,sizeof seen);
    for(p=0;p<g_n;p++) if(z->on[p]){ np++; if(!seen[z->g[p]]){seen[z->g[p]]=1;ng++;} }
    *places=np; *groups=ng;
}

/* Placement description length. Naming an arbitrary subset of m positions costs m*log2(N) bits.
 * Naming an arithmetic progression costs 2*log2(N) (offset and stride) plus log2(N) per position that
 * deviates from it. A regular tiling is therefore cheap to describe and a ragged one is not, which is
 * the actual MDL argument for a convolution, and is exactly what groups/N cannot see. */
static double bits_placement(const Geno *z)
{
    int o,s,p,np,ng,best=1<<30; double ln=log2((double)g_n);
    stats(z,&np,&ng);
    if(np==0) return 0.0;
    for(s=1;s<=g_n;s++) for(o=0;o<g_n;o++){
        int miss=0;
        for(p=0;p<g_n;p++){
            int on_prog = ((p-o)%s+s)%s==0;
            if(z->on[p] != (char)on_prog) miss++; }
        if(miss<best) best=miss; }
    return 2.0*ln + best*ln;
}
static double bits_filters(const Geno *z)
{ int np,ng; stats(z,&np,&ng); return ng*K*8.0; }      /* 8 bits per shared weight */

int main(void)
{
    int seeds=envint("SEEDS",40), sd, m, mode;
    double ln;
    g_n=envint("NIN",12); if(g_n>NMAX) g_n=NMAX;
    g_ntp=envint("NTRAINPOS",2);
    ln=log2((double)g_n);

    printf("PROTOCOL CHECK 1 on candidate objectives. No search is run.\n");
    printf("N=%d K=%d, weights trained on %d positions, outcome measured on the other %d, %d seeds.\n\n",
           g_n, K, g_ntp, g_n-g_ntp, seeds);
    printf("  %-30s %-8s %-9s %-9s %-9s %-9s\n",
           "genome", "outcome", "A grp/N", "B pls/N", "C place", "D total");

    { double ax[64],cA[64],cB[64],cC[64],cD[64]; int nrow=0;
      for(mode=0;mode<3;mode++)
      for(m=1;m<=g_n;m+=(g_n>12?4:2)){
        Geno z; double a=0; int p,np,ng; char lab[64];
        for(p=0;p<g_n;p++){ z.on[p]=0; z.g[p]=0; }
        if(mode==0){ int step=g_n/m; if(step<1) step=1;                 /* regular stride, one filter */
                     for(p=0;p<g_n && (p/step)<m;p+=step){ z.on[p]=1; z.g[p]=0; }
                     sprintf(lab,"%2d places, regular, 1 filter",m); }
        else if(mode==1){ int c2=0;                                     /* ragged, one filter */
                     rseed((uint32_t)(m*7919+13));
                     while(c2<m){ p=(int)(r32()%(uint32_t)g_n); if(!z.on[p]){z.on[p]=1;z.g[p]=0;c2++;} }
                     sprintf(lab,"%2d places, ragged,  1 filter",m); }
        else { int step=g_n/m; if(step<1) step=1;                       /* regular, every unit its own */
                     for(p=0;p<g_n && (p/step)<m;p+=step){ z.on[p]=1; z.g[p]=(unsigned char)p; }
                     sprintf(lab,"%2d places, regular, unshared",m); }
        stats(&z,&np,&ng);
        if(np==0) continue;
        for(sd=1;sd<=seeds;sd++){ new_task((uint32_t)(sd*131+1)); a+=outcome(&z,(uint32_t)(sd*7+1)); }
        a/=seeds;
        ax[nrow]=a;
        cA[nrow]=(double)ng/g_n;
        cB[nrow]=(double)np/g_n;
        cC[nrow]=bits_placement(&z)/(g_n*ln);
        cD[nrow]=(bits_filters(&z)+bits_placement(&z))/(g_n*ln+g_n*K*8.0);
        printf("  %-30s %-8.3f %-9.3f %-9.3f %-9.3f %-9.3f\n", lab, a, cA[nrow], cB[nrow], cC[nrow], cD[nrow]);
        nrow++;
      }

      printf("\n  VERDICT  (a cost must VARY over the outcome, and vary the RIGHT WAY:\n");
      printf("            better outcome should mean lower cost)\n\n");
      { const char *nm[4]={"A  groups/N","B  places/N","C  placement MDL","D  total MDL"};
        double *cc[4]={cA,cB,cC,cD}; int j;
        for(j=0;j<4;j++){
            double mn=1e30,mx=-1e30,sx=0,sy=0,sxy=0,sxx=0,syy=0; int t;
            for(t=0;t<nrow;t++){ double c=cc[j][t];
                if(c<mn) mn=c;
                if(c>mx) mx=c;
                sx+=ax[t]; sy+=c; sxy+=ax[t]*c; sxx+=ax[t]*ax[t]; syy+=c*c; }
            { double num=nrow*sxy-sx*sy;
              double den=sqrt((nrow*sxx-sx*sx)*(nrow*syy-sy*sy));
              double r = den>1e-12 ? num/den : 0.0;
              const char *v;
              if(mx-mn < 0.02) v="FLAT  - cannot see the outcome at all";
              else if(r > 0.3)  v="WRONG WAY - better outcome costs MORE";
              else if(r < -0.3) v="usable  - better outcome costs less";
              else              v="uninformative - varies but uncorrelated";
              printf("    %-18s range %.3f   corr with outcome %+.2f   %s\n", nm[j], mx-mn, r, v); } }
      }
    }
    printf("\n  Outcome range across these genomes is the span the objective must be able to see.\n");
    return 0;
}
