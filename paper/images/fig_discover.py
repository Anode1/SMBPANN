#!/usr/bin/env python3
# fig_discover.svg -- composition does NOT cleanly emerge when the search must find the decomposition.
# Left: energy-selected channel count overshoots the operation count (1-op solves at C=1; 2-op needs C=3).
# Right: the unsupervised discovered solution (0.81) falls short of the supervised curriculum (0.87)
# because the two channels specialize only 55% of the time. Numbers from emerge_discover.c (40 seeds).
# Paper style (matches fig6_directed.py).
C = [1,2,3,4]
oneop = [0.972,0.991,0.996,0.998]
twoop = [0.711,0.812,0.875,0.899]
TARGET = 0.85
# right panel: 2-op method levels (label, value, color, emphasis)
INK="#1e293b"; MUT="#64748b"; GRID="#e2e8f0"; TEAL="#1f7a8c"; ORNG="#c56b3e"; GRAY="#94a3b8"
levels = [
    ("aux-label curriculum (supervised)", 0.874, ORNG),
    ("discovered (composite label only)", 0.812, TEAL),
    ("single channel (cap, by constr.)",  0.750, GRAY),
    ("unshared from-scratch (floor)",      0.650, GRAY),
]
out=[]; t=out.append
t('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 300" font-family="Segoe UI, Helvetica, Arial, sans-serif">')
t('<title>Composition does not cleanly emerge without supervision</title>')
t('<rect x="0" y="0" width="820" height="300" fill="#ffffff"/>')
t(f'<text x="20" y="26" font-size="16" font-weight="600" fill="{INK}">Composition does not cleanly emerge when the search must find the decomposition</text>')
t(f'<text x="20" y="45" font-size="12" fill="{MUT}">Two-operation task (A AND B), no per-operation labels; 40 paired seeds. The search over-provisions channels and under-specializes.</text>')

# ---------- LEFT: accuracy vs channel count ----------
ox,oy,pw,ph = 66,80,300,160
y0,y1 = 0.5,1.0
X=lambda i: ox+pw*(C[i]-1)/3.0
Y=lambda v: oy+ph*(y1-v)/(y1-y0)
t(f'<text x="{ox}" y="{oy-9}" font-size="12.5" font-weight="600" fill="{INK}">Energy-selected channel count overshoots</text>')
for v in [0.5,0.6,0.7,0.8,0.9,1.0]:
    y=Y(v); t(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
    t(f'<text x="{ox-6}" y="{y+4:.1f}" font-size="10" text-anchor="end" fill="{MUT}">{v:.1f}</text>')
# target line
yt=Y(TARGET); t(f'<line x1="{ox}" y1="{yt:.1f}" x2="{ox+pw}" y2="{yt:.1f}" stroke="{GRAY}" stroke-width="1.3" stroke-dasharray="5 4"/>')
t(f'<text x="{ox+pw-2}" y="{yt-5:.1f}" font-size="9.5" text-anchor="end" fill="{MUT}">target 0.85</text>')
for i,c in enumerate(C):
    t(f'<text x="{X(i):.1f}" y="{oy+ph+16}" font-size="10.5" text-anchor="middle" fill="{MUT}">{c}</text>')
t(f'<text x="{ox+pw/2:.1f}" y="{oy+ph+31}" font-size="11" text-anchor="middle" fill="{MUT}">channel count C</text>')
t(f'<text transform="rotate(-90 {ox-38} {oy+ph/2:.1f})" x="{ox-38}" y="{oy+ph/2:.1f}" font-size="11" text-anchor="middle" fill="{MUT}">test accuracy</text>')
def line(vals,col):
    pts=" ".join(f"{X(i):.1f},{Y(vals[i]):.1f}" for i in range(len(vals)))
    t(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round"/>')
    for i in range(len(vals)): t(f'<circle cx="{X(i):.1f}" cy="{Y(vals[i]):.1f}" r="3" fill="{col}"/>')
line(oneop, TEAL)
line(twoop, ORNG)
# ring the energy-selected C* for each arm (first C reaching target): 1-op at C=1, 2-op at C=3
t(f'<circle cx="{X(0):.1f}" cy="{Y(oneop[0]):.1f}" r="6" fill="none" stroke="{TEAL}" stroke-width="1.6"/>')
t(f'<circle cx="{X(2):.1f}" cy="{Y(twoop[2]):.1f}" r="6" fill="none" stroke="{ORNG}" stroke-width="1.6"/>')
# legend in the empty lower-left of the panel
lgx,lgy=ox+10,oy+ph-40
for k,(lab,col) in enumerate([("1 operation  →  C*=1",TEAL),("2 operations  →  C*=3 (overshoot)",ORNG)]):
    yy=lgy+k*16
    t(f'<line x1="{lgx}" y1="{yy}" x2="{lgx+18}" y2="{yy}" stroke="{col}" stroke-width="2.6"/>')
    t(f'<circle cx="{lgx+9}" cy="{yy}" r="3" fill="{col}"/>')
    t(f'<text x="{lgx+24}" y="{yy+3.5}" font-size="10" fill="{INK}">{lab}</text>')

# ---------- RIGHT: 2-op method levels ----------
bx,by,bw,bh = 470,80,250,160
vx0,vx1 = 0.5,0.95
BX=lambda v: bx+bw*(v-vx0)/(vx1-vx0)
t(f'<text x="{bx}" y="{by-9}" font-size="12.5" font-weight="600" fill="{INK}">Where the discovered solution lands</text>')
rowh=bh/len(levels)
for k,(lab,val,col) in enumerate(levels):
    yc=by+rowh*(k+0.5)
    t(f'<rect x="{bx}" y="{yc-9:.1f}" width="{BX(val)-bx:.1f}" height="18" rx="3" fill="{col}" opacity="{0.95 if col!=GRAY else 0.55}"/>')
    t(f'<text x="{BX(val)+6:.1f}" y="{yc+4:.1f}" font-size="10.5" fill="{INK}" font-weight="600">{val:.2f}</text>')
    t(f'<text x="{bx}" y="{yc-13:.1f}" font-size="9.7" fill="{MUT}">{lab}</text>')
# target line vertical
xt=BX(TARGET); t(f'<line x1="{xt:.1f}" y1="{by-4}" x2="{xt:.1f}" y2="{by+bh+2}" stroke="{GRAY}" stroke-width="1.3" stroke-dasharray="5 4"/>')
t(f'<text x="{xt:.1f}" y="{by+bh+15}" font-size="9.5" text-anchor="middle" fill="{MUT}">target 0.85</text>')
# specialization annotation
t(f'<text x="{bx}" y="{by+bh+34}" font-size="10.5" fill="{INK}">The two channels split the two motifs in only <tspan font-weight="700" fill="{ORNG}">55%</tspan> of runs;</text>')
t(f'<text x="{bx}" y="{by+bh+48}" font-size="10.5" fill="{MUT}">the rest collapse onto one, so the honest split needs supervision.</text>')
t('</svg>')
open("fig_discover.svg","w").write("\n".join(out))
print("wrote fig_discover.svg")
