#!/usr/bin/env python3
# fig6_directed.svg -- honest two-panel: directed search is TIDIER (contiguity) at EQUAL task accuracy.
# Paper style: 820-wide, white bg, Segoe UI, slate palette. Numbers from scratch_prove_sweep.out (fixed, 50 seeds).
import math
N   = [12,16,20,24,28]
sq  = math.sqrt(50)
# contiguity (fixed budget)
gaC=[0.73,0.68,0.67,0.56,0.52]; gaCsd=[0.31,0.29,0.27,0.22,0.15]
rnC=[0.71,0.62,0.47,0.43,0.40]; rnCsd=[0.32,0.28,0.24,0.26,0.26]
# test accuracy (fixed budget)
gaA=[0.885,0.882,0.874,0.866,0.862]; gaAsd=[0.028,0.030,0.045,0.049,0.056]
rnA=[0.880,0.877,0.866,0.862,0.853]; rnAsd=[0.032,0.037,0.041,0.050,0.053]
sem=lambda v:[x/sq for x in v]
gaCs,rnCs,gaAs,rnAs = sem(gaCsd),sem(rnCsd),sem(gaAsd),sem(rnAsd)

INK="#1e293b"; MUT="#64748b"; GRID="#e2e8f0"; GA="#1f7a8c"; RN="#c56b3e"
GABAND="rgba(31,122,140,0.13)"; RNBAND="rgba(197,107,62,0.12)"
out=[]; t=out.append
t('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 300" font-family="Segoe UI, Helvetica, Arial, sans-serif">')
t('<title>Directed search is tidier than random at equal task accuracy</title>')
t('<rect x="0" y="0" width="820" height="300" fill="#ffffff"/>')
t(f'<text x="20" y="27" font-size="16" font-weight="600" fill="{INK}">Directed search finds a tidier filter than random, at the same task accuracy</text>')
t(f'<text x="20" y="46" font-size="12" fill="{MUT}">matched evaluation budget, 50 paired seeds; bands are ±1 SEM. GA wins on structure (what the objective rewards), ties on the task.</text>')

def panel(ox,oy,pw,ph,y0,y1,yticks,yfmt,title,ylab):
    t(f'<text x="{ox}" y="{oy-8}" font-size="12.5" font-weight="600" fill="{INK}">{title}</text>')
    X=lambda i: ox+pw*(N[i]-12)/16.0
    Y=lambda v: oy+ph*(y1-v)/(y1-y0)
    for v in yticks:
        y=Y(v); t(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        t(f'<text x="{ox-6}" y="{y+4:.1f}" font-size="10.5" text-anchor="end" fill="{MUT}">{yfmt(v)}</text>')
    for i,n in enumerate(N):
        t(f'<text x="{X(i):.1f}" y="{oy+ph+16}" font-size="10.5" text-anchor="middle" fill="{MUT}">{n}</text>')
    t(f'<text x="{ox+pw/2:.1f}" y="{oy+ph+30}" font-size="11" text-anchor="middle" fill="{MUT}">input size N</text>')
    t(f'<text transform="rotate(-90 {ox-34} {oy+ph/2:.1f})" x="{ox-34}" y="{oy+ph/2:.1f}" font-size="11" text-anchor="middle" fill="{MUT}">{ylab}</text>')
    return X,Y

def band(X,Y,m,s,fill):
    up=" ".join(f"{X(i):.1f},{Y(m[i]+s[i]):.1f}" for i in range(len(m)))
    dn=" ".join(f"{X(i):.1f},{Y(m[i]-s[i]):.1f}" for i in range(len(m)-1,-1,-1))
    t(f'<polygon points="{up} {dn}" fill="{fill}" stroke="none"/>')
def line(X,Y,m,color,dash=False):
    pts=" ".join(f"{X(i):.1f},{Y(m[i]):.1f}" for i in range(len(m)))
    da=' stroke-dasharray="6 5"' if dash else ''
    t(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.3"{da} stroke-linejoin="round"/>')
    for i in range(len(m)): t(f'<circle cx="{X(i):.1f}" cy="{Y(m[i]):.1f}" r="2.8" fill="{color}"/>')

# left: contiguity (structure)
X,Y=panel(70,80,300,150,0.3,0.8,[0.3,0.4,0.5,0.6,0.7,0.8],lambda v:f"{v:.1f}",
          "Structure: contiguity of the filter","contiguity (taps / span)")
band(X,Y,rnC,rnCs,RNBAND); band(X,Y,gaC,gaCs,GABAND)
line(X,Y,rnC,RN,True); line(X,Y,gaC,GA,False)
t(f'<text x="370" y="{Y(gaC[4])-7:.1f}" font-size="10.5" font-weight="700" text-anchor="end" fill="{GA}">GA</text>')
t(f'<text x="370" y="{Y(rnC[4])+14:.1f}" font-size="10.5" text-anchor="end" fill="{RN}">random</text>')

# right: accuracy (task) -- same axis honesty, wide range so a real tie reads as a tie
X2,Y2=panel(500,80,260,150,0.75,0.95,[0.75,0.80,0.85,0.90,0.95],lambda v:f"{v:.2f}",
            "Task: held-out accuracy","test accuracy")
band(X2,Y2,rnA,rnAs,RNBAND); band(X2,Y2,gaA,gaAs,GABAND)
line(X2,Y2,rnA,RN,True); line(X2,Y2,gaA,GA,False)

t('</svg>')
open("paper/images/fig6_directed.svg","w").write("\n".join(out))
print("wrote paper/images/fig6_directed.svg")
