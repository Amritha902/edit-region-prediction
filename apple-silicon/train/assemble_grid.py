"""Assemble the complete threshold x dilation grid from every banked cell."""
import json, glob
from pathlib import Path
HERE=Path(__file__).resolve().parent
cells={}
for f in glob.glob(str(HERE/"stage2_joint*.json"))+glob.glob(str(HERE/"stage2_cell_*.json")):
    d=json.load(open(f))
    for k,v in d.get("sweep",{}).items():
        cells[(float(v["thr"]),int(v["dil"]))]=v
THRS=sorted({k[0] for k in cells}); DILS=sorted({k[1] for k in cells})
print(f"grid: {len(cells)} cells, thresholds {THRS}, dilations {DILS}\n")
print("NET")
print("      " + "".join(f"{d:>9}px" for d in DILS))
for t in THRS:
    row=f"  {t:<4}"
    for d in DILS:
        v=cells.get((t,d)); row += f"{v['net']:>+10.1%}" if v else f"{'—':>11}"
    print(row)
print("\nCOLLATERAL")
print("      " + "".join(f"{d:>9}px" for d in DILS))
for t in THRS:
    row=f"  {t:<4}"
    for d in DILS:
        v=cells.get((t,d)); row += f"{v['collateral']:>10.1%}" if v else f"{'—':>11}"
    print(row)
best=max(cells.items(), key=lambda kv: kv[1]["net"])
(bt,bd),bv=best
print(f"\n  BEST net: threshold {bt}, dilation {bd} px")
print(f"    net {bv['net']:+.1%}  recall {bv['recall']:.1%}  collateral {bv['collateral']:.1%}  PSNR {bv['psnr']:.2f}")
# interiority check
ti=THRS.index(bt); di=DILS.index(bd)
edge=[]
if ti in (0,len(THRS)-1): edge.append("threshold")
if di in (0,len(DILS)-1): edge.append("dilation")
print(f"    interior on both axes: {'YES' if not edge else 'NO — at the edge in '+', '.join(edge)}")
print("\n  per-row peak (optimal dilation rises with threshold):")
for t in THRS:
    r={d:cells[(t,d)] for d in DILS if (t,d) in cells}
    if not r: continue
    bd_=max(r,key=lambda d:r[d]["net"])
    at_edge = bd_==max(r) and len(r)>1 and r[bd_]["net"]>r[sorted(r)[-2]]["net"]
    print(f"    thr {t}: peak at {bd_:>2} px, net {r[bd_]['net']:+.1%}"
          f"{'   <-- still climbing at the grid edge' if at_edge else ''}")
ref=json.load(open(HERE/"stage2_sd.json"))["arms"]
print(f"\n  references: whole-frame {ref['A whole-frame']['net']:+.1%} | "
      f"MagicBrush {ref['D MagicBrush mask']['net']:+.1%} | oracle {ref['C GT mask']['net']:+.1%}")
json.dump({f"{t}_{d}":v for (t,d),v in cells.items()} , open(HERE/"grid_final.json","w"), indent=2)
print(f"\n  wrote train/grid_final.json")
