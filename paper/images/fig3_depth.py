#!/usr/bin/env python3
# fig3_depth.svg -- emergent depth grows with the required receptive field but OVERSHOOTS it, and under a
# fair mean-over-restarts never reaches target. Numbers from emerge_compose.c (30 seeds x 4 restarts, mean).
# Paper style (matches fig6_directed.py).
L = [1,2,3,4,5]
TARGET = 0.85
series = [  # (label, need-depth, values, color) -- sequential teal, light->dark for larger s
    ("s=2 (need L=1)", 1, [0.608,0.745,0.814,0.760,0.695], "#9ecfd6"),
    ("s=4 (need L=2)", 2, [0.524,0.628,0.730,0.708,0.638], "#5ba3b0"),
    ("s=6 (need L=3)", 3, [0.522,0.551,0.624,0.662,0.666], "#2b8a9e"),
    ("s=8 (need L=4)", 4, [0.519,0.544,0.589,0.626,0.648], "#14506b"),
]
INK="#1e293b"; MUT="#64748b"; GRID="#e2e8f0"; RED="#c56b3e"
W,H=560,320
ox,oy,pw,ph=60,74,360,188
y0,y1=0.5,0.9
X=lambda i: ox+pw*(L[i]-1)/4.0
Y=lambda v: oy+ph*(y1-v)/(y1-y0)
out=[]; t=out.append
t(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">')
t('<title>Emergent depth grows with the receptive field but overshoots it</title>')
t(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
t(f'<text x="20" y="26" font-size="15.5" font-weight="600" fill="{INK}">Emergent depth grows with the required receptive field, but overshoots it</text>')
t(f'<text x="20" y="44" font-size="11.5" fill="{MUT}">test accuracy vs stack depth L, by required spike distance s; fair mean over 4 restarts, 30 seeds.</text>')
for v in [0.5,0.6,0.7,0.8,0.9]:
    y=Y(v); t(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
    t(f'<text x="{ox-7}" y="{y+4:.1f}" font-size="10.5" text-anchor="end" fill="{MUT}">{v:.1f}</text>')
yt=Y(TARGET)
t(f'<line x1="{ox}" y1="{yt:.1f}" x2="{ox+pw}" y2="{yt:.1f}" stroke="{RED}" stroke-width="1.3" stroke-dasharray="5 4"/>')
t(f'<text x="{ox+pw-2}" y="{yt-5:.1f}" font-size="10" text-anchor="end" fill="{RED}">target 0.85 (never reached)</text>')
for i in L:
    t(f'<text x="{X(L.index(i)):.1f}" y="{oy+ph+16}" font-size="10.5" text-anchor="middle" fill="{MUT}">{i}</text>')
t(f'<text x="{ox+pw/2:.1f}" y="{oy+ph+32}" font-size="11" text-anchor="middle" fill="{MUT}">stack depth L</text>')
t(f'<text transform="rotate(-90 {ox-38} {oy+ph/2:.1f})" x="{ox-38}" y="{oy+ph/2:.1f}" font-size="11" text-anchor="middle" fill="{MUT}">test accuracy</text>')
for lab,need,vals,col in series:
    pts=" ".join(f"{X(i):.1f},{Y(vals[i]):.1f}" for i in range(len(L)))
    t(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round"/>')
    for i in range(len(L)): t(f'<circle cx="{X(i):.1f}" cy="{Y(vals[i]):.1f}" r="2.8" fill="{col}"/>')
    # ring the peak (best depth) to show the overshoot vs need
    pk=vals.index(max(vals))
    t(f'<circle cx="{X(pk):.1f}" cy="{Y(vals[pk]):.1f}" r="5.5" fill="none" stroke="{col}" stroke-width="1.5"/>')
# legend (right, inside)
lx,ly=ox+12,oy+4
for k,(lab,need,vals,col) in enumerate(series):
    yy=ly+k*15
    t(f'<line x1="{lx}" y1="{yy}" x2="{lx+16}" y2="{yy}" stroke="{col}" stroke-width="2.6"/>')
    t(f'<text x="{lx+22}" y="{yy+3.5}" font-size="9.5" fill="{INK}">{lab}</text>')
t(f'<text x="20" y="{H-12}" font-size="10.5" fill="{MUT}">Best depth (ringed) peaks at L≈3,3,4,5 for s=2,4,6,8 — deeper than the required 1,2,3,4: a trend that overshoots.</text>')
t('</svg>')
open("fig3_depth.svg","w").write("\n".join(out))
print("wrote fig3_depth.svg")
