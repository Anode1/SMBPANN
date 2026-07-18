#!/usr/bin/env python3
# fig_boundary.svg -- the emergence phase boundary: GA on-relevance vs input size N at three search
# budgets. The boundary (where the GA stops landing on the generative offsets) recedes as budget grows.
# Numbers from emerge_prove.c BOUNDARY mode (8 seeds/cell). Paper style (matches fig6_directed.py).
Nx = [12,16,20,24,28,32]
# on-relevance per budget (gens -> evals)
series = [
    ("~1k evals (40 gens)",   [0.90,0.43,0.43,0.22,0.15,0.10], "#9ecfd6"),
    ("~3k evals (120 gens)",  [0.90,0.77,0.84,0.34,0.39,0.23], "#2b8a9e"),
    ("~9k evals (360 gens)",  [0.90,0.77,0.82,0.67,0.52,0.43], "#14506b"),
]
THR = 0.40
INK="#1e293b"; MUT="#64748b"; GRID="#e2e8f0"; RED="#c56b3e"
W,H = 560,330
ox,oy,pw,ph = 66,72,430,196
y0,y1 = 0.0,1.0
X=lambda n: ox+pw*(n-12)/(32-12)
Y=lambda v: oy+ph*(y1-v)/(y1-y0)
out=[]; t=out.append
t(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">')
t('<title>The emergence boundary recedes as search budget grows</title>')
t(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
t(f'<text x="20" y="27" font-size="15.5" font-weight="600" fill="{INK}">The emergence boundary recedes as the search budget grows</text>')
t(f'<text x="20" y="45" font-size="11.5" fill="{MUT}">Share of GA taps landing on the generative offsets, vs input size N, at three search budgets (8 seeds/cell).</text>')
# y gridlines
for v in [0.0,0.2,0.4,0.6,0.8,1.0]:
    y=Y(v); t(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
    t(f'<text x="{ox-7}" y="{y+4:.1f}" font-size="10.5" text-anchor="end" fill="{MUT}">{v:.1f}</text>')
# threshold line
yt=Y(THR)
t(f'<line x1="{ox}" y1="{yt:.1f}" x2="{ox+pw}" y2="{yt:.1f}" stroke="{RED}" stroke-width="1.4" stroke-dasharray="5 4"/>')
t(f'<text x="{ox+pw-2}" y="{yt-6:.1f}" font-size="10" text-anchor="end" fill="{RED}">N* threshold (0.40)</text>')
# x ticks
for n in Nx:
    t(f'<text x="{X(n):.1f}" y="{oy+ph+16}" font-size="10.5" text-anchor="middle" fill="{MUT}">{n}</text>')
t(f'<text x="{ox+pw/2:.1f}" y="{oy+ph+32}" font-size="11" text-anchor="middle" fill="{MUT}">input size N (offsets to prune)</text>')
t(f'<text transform="rotate(-90 {ox-42} {oy+ph/2:.1f})" x="{ox-42}" y="{oy+ph/2:.1f}" font-size="11" text-anchor="middle" fill="{MUT}">GA on-relevance</text>')
# series lines + markers
for lab,vals,col in series:
    pts=" ".join(f"{X(Nx[i]):.1f},{Y(vals[i]):.1f}" for i in range(len(Nx)))
    t(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round"/>')
    for i in range(len(Nx)):
        t(f'<circle cx="{X(Nx[i]):.1f}" cy="{Y(vals[i]):.1f}" r="3" fill="{col}"/>')
# direct labels at right end
for lab,vals,col in series:
    t(f'<text x="{X(32)+8:.1f}" y="{Y(vals[-1])+4:.1f}" font-size="10" fill="{col}" font-weight="600">{vals[-1]:.2f}</text>')
# legend (top-right inside plot)
lx,ly=ox+pw-186,oy+8
for k,(lab,vals,col) in enumerate(series):
    yy=ly+k*16
    t(f'<line x1="{lx}" y1="{yy}" x2="{lx+18}" y2="{yy}" stroke="{col}" stroke-width="2.6"/>')
    t(f'<text x="{lx+24}" y="{yy+3.5}" font-size="10" fill="{INK}">{lab}</text>')
# caption line
t(f'<text x="20" y="{H-12}" font-size="10.5" fill="{MUT}">N* (last N above 0.40): 20 → 20 → ≥32 as budget rises ~1k → 3k → 9k evals — pushed outward, not fixed.</text>')
t('</svg>')
open("fig_boundary.svg","w").write("\n".join(out))
print("wrote fig_boundary.svg")
