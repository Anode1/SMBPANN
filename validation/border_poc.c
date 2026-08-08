/* border_poc.c -- proof of concept: represent an image by its BORDERS and reconstruct the smooth
 * interiors by diffusion. Tests the "edges are a near-complete, compact code" idea (Marr; Elder-Zucker;
 * diffusion curves).
 *
 * Borders are found as LAPLACIAN ZERO-CROSSINGS (Marr-Hildreth), NOT gradient magnitude: the Laplacian of
 * a smooth ramp is zero, so a gradient region produces no false edges -- only true intensity
 * discontinuities do. We keep only the thin border band (dilated by 1 so the diffusion sees the value on
 * each side) plus the image frame, discard the rest, and recover the interior by solving Laplace's
 * equation with the kept pixels fixed. Reports compactness (fraction kept) and reconstruction PSNR, and
 * writes PGMs so the result is visible.
 *
 * Pure C99, no deps. Build: cc -std=c99 -pedantic -Wall -Wextra -O2 border_poc.c -lm -o border_poc
 * Run:   ./border_poc [input.pgm] [slope_thr]   (no arg -> synthetic: flat shapes on a smooth gradient)
 * Out:   border_orig.pgm, border_edges.pgm, border_recon.pgm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int W, H;
static double *img;
static unsigned char *edge;

static double clampd(double v){ return v<0?0:(v>255?255:v); }

static void gen_synth(void)
{
    W=128; H=128; img=malloc(sizeof(double)*W*H);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        double v = 40.0 + 120.0*x/(double)W;                       /* smooth linear ramp (harmonic) */
        double dx=x-40, dy=y-42; if(dx*dx+dy*dy < 22*22) v=205;     /* flat disc */
        if(x>=72&&x<112&&y>=74&&y<114) v=95;                       /* flat rectangle */
        img[y*W+x]=v;
    }
}

static int read_pgm(const char*path)
{
    FILE*f=fopen(path,"rb"); if(!f) return 0;
    char m[3]={0}; int maxv;
    if(fscanf(f,"%2s",m)!=1||strcmp(m,"P5")){ fclose(f); return 0; }
    if(fscanf(f,"%d %d %d",&W,&H,&maxv)!=3){ fclose(f); return 0; }
    fgetc(f);
    img=malloc(sizeof(double)*W*H);
    for(int i=0;i<W*H;i++){ int c=fgetc(f); if(c==EOF){ fclose(f); return 0; } img[i]=(double)c*255.0/maxv; }
    fclose(f); return 1;
}

static void write_pgm(const char*path,const double*a)
{
    FILE*f=fopen(path,"wb"); if(!f) return;
    fprintf(f,"P5\n%d %d\n255\n",W,H);
    for(int i=0;i<W*H;i++) fputc((int)(clampd(a[i])+0.5),f);
    fclose(f);
}

/* Laplacian zero-crossings (after one light binomial smoothing pass), slope-thresholded, dilated by 1. */
static void find_edges(double slope_thr)
{
    int N=W*H, i,x,y,dx,dy;
    double *sm=malloc(sizeof(double)*N), *L=malloc(sizeof(double)*N), *t=calloc((size_t)N,sizeof(double));
    for(i=0;i<N;i++) sm[i]=img[i];
    for(y=0;y<H;y++) for(x=0;x<W;x++){                 /* 3x3 binomial smooth (LoG-ish) */
        double s=0,wsum=0;
        for(dy=-1;dy<=1;dy++) for(dx=-1;dx<=1;dx++){
            int yy=y+dy,xx=x+dx; if(yy<0||yy>=H||xx<0||xx>=W) continue;
            double wgt=(dy==0?2:1)*(dx==0?2:1); s+=wgt*sm[yy*W+xx]; wsum+=wgt;
        }
        t[y*W+x]=s/wsum;
    }
    memcpy(sm,t,sizeof(double)*N);
    for(i=0;i<N;i++) L[i]=0;
    for(y=1;y<H-1;y++) for(x=1;x<W-1;x++){ i=y*W+x; L[i]=sm[i-1]+sm[i+1]+sm[i-W]+sm[i+W]-4*sm[i]; }

    edge=malloc(N); memset(edge,0,N);
    for(y=1;y<H-1;y++) for(x=1;x<W-1;x++){             /* zero-crossing with a minimum slope */
        i=y*W+x;
        if((L[i]>0)!=(L[i+1]>0) && fabs(L[i]-L[i+1])>slope_thr){ edge[i]=1; edge[i+1]=1; }
        if((L[i]>0)!=(L[i+W]>0) && fabs(L[i]-L[i+W])>slope_thr){ edge[i]=1; edge[i+W]=1; }
    }
    { unsigned char *d=malloc(N); memcpy(d,edge,N);   /* dilate by 1: keep both sides of each border */
      for(y=1;y<H-1;y++) for(x=1;x<W-1;x++){ i=y*W+x; if(edge[i]){ d[i-1]=d[i+1]=d[i-W]=d[i+W]=1; } }
      memcpy(edge,d,N); free(d); }
    for(x=0;x<W;x++){ edge[x]=1; edge[(H-1)*W+x]=1; }  /* anchor the frame */
    for(y=0;y<H;y++){ edge[y*W]=1; edge[y*W+W-1]=1; }
    free(sm); free(L); free(t);
}

static double* reconstruct(int iters)
{
    int N=W*H,i,x,y,it; double *R=malloc(sizeof(double)*N), mean=0, w=1.9;
    for(i=0;i<N;i++) mean+=img[i];
    mean/=N;
    for(i=0;i<N;i++) R[i]= edge[i]? img[i] : mean;
    for(it=0;it<iters;it++) for(y=1;y<H-1;y++) for(x=1;x<W-1;x++){
        i=y*W+x; if(edge[i]) continue;
        R[i]=(1-w)*R[i] + w*0.25*(R[i-1]+R[i+1]+R[i-W]+R[i+W]);
    }
    return R;
}

int main(int argc,char**argv)
{
    double slope_thr = argc>2 ? atof(argv[2]) : 8.0;
    if(argc>1 && strcmp(argv[1],"-")!=0){ if(!read_pgm(argv[1])){ fprintf(stderr,"cannot read PGM %s\n",argv[1]); return 1; }
        printf("loaded %s: %dx%d\n",argv[1],W,H); }
    else { gen_synth(); printf("synthetic: %dx%d (smooth ramp + flat disc + flat rectangle)\n",W,H); }

    find_edges(slope_thr);
    int kept=0; for(int i=0;i<W*H;i++) kept+=edge[i];
    double *R=reconstruct(6000);
    double mse=0; for(int i=0;i<W*H;i++){ double d=img[i]-clampd(R[i]); mse+=d*d; } mse/=W*H;
    double psnr = mse>1e-9 ? 10.0*log10(255.0*255.0/mse) : 99.0;

    { double *e=malloc(sizeof(double)*W*H); for(int i=0;i<W*H;i++) e[i]=edge[i]?0:255;
      write_pgm("border_orig.pgm",img); write_pgm("border_edges.pgm",e); write_pgm("border_recon.pgm",R); free(e); }

    printf("kept border+frame pixels: %d / %d  (%.1f%%  ->  %.1fx fewer positions)\n",
           kept, W*H, 100.0*kept/(W*H), (double)(W*H)/kept);
    printf("reconstruction PSNR: %.1f dB  (MSE %.2f)\n", psnr, mse);
    printf("wrote border_orig.pgm, border_edges.pgm, border_recon.pgm\n");
    return 0;
}
