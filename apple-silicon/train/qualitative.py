"""Show what the model actually predicts: image | instruction | truth | prediction."""
import glob, io, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"datagen"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cv2, pandas as pd
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.device import pick_device
from model import EditRegionModel, TextCoeffHead
import mask_extraction as ME

dev=pick_device("auto")
model=EditRegionModel("best.pt", device=dev)
ck=torch.load(HERE/"edit_region_head.pt", map_location=dev)
cfg=ck.get("cfg",{})
model.head=TextCoeffHead(hidden=cfg.get("hidden",512), dropout=cfg.get("dropout",0.0)).to(dev)
model.head.load_state_dict(ck["head"]); model.head.eval()
print(f"loaded head: epoch {ck['epoch']} IoU {ck['iou']:.4f}")

df=pd.read_parquet(sorted(glob.glob(str(ROOT/"data/magicbrush/data/dev-*.parquet")))[0])
picks=[]
for want in ("insert","remove","modify"):
    for i in range(len(df)):
        ins=df.instruction.iloc[i].strip().lower()
        k = "insert" if ins.startswith(("put ","add ")) else "remove" if ins.startswith(("remove","delete")) else "modify"
        if k==want and i not in [p[0] for p in picks]:
            picks.append((i,k)); 
            if sum(1 for p in picks if p[1]==want)>=2: break

rows=[]
for i,k in picks:
    src=Image.open(io.BytesIO(df.source_img.iloc[i]["bytes"])).convert("RGB")
    tgt=Image.open(io.BytesIO(df.target_img.iloc[i]["bytes"])).convert("RGB").resize(src.size)
    ins=str(df.instruction.iloc[i])
    o=cv2.cvtColor(np.asarray(src),cv2.COLOR_RGB2BGR); e=cv2.cvtColor(np.asarray(tgt),cv2.COLOR_RGB2BGR)
    gt=ME.clean_mask(((ME.delta_e(o,e)>ME.DELTA_E_THRESHOLD)*255).astype(np.uint8))>0
    img=src.resize((640,640),Image.BILINEAR)
    x=torch.from_numpy(np.asarray(img)).permute(2,0,1).float().div(255)[None].to(dev)
    with torch.no_grad():
        lg=model(x,[ins])
        pr=(torch.sigmoid(lg)[0,0].cpu().numpy()>0.5)
    pr=cv2.resize(pr.astype(np.uint8),src.size,interpolation=cv2.INTER_NEAREST)>0
    inter=(pr&gt).sum(); iou=inter/max((pr|gt).sum(),1)
    rows.append((np.asarray(src),np.asarray(tgt),gt,pr,ins,k,iou))

n=len(rows)
fig,ax=plt.subplots(n,4,figsize=(15,3.3*n))
for r,(s_,t_,gt,pr,ins,k,iou) in enumerate(rows):
    def ov(base,m,c):
        v=base.copy(); v[m]=(v[m]*.4+np.array(c)*.6).astype(np.uint8); return v
    for c,(im,ti) in enumerate([(s_,"source"),(t_,"target (human edit)"),
                                (ov(s_,gt,[255,0,180]),f"ΔE ground truth  {gt.mean():.1%}"),
                                (ov(s_,pr,[0,190,255]),f"model prediction  IoU {iou:.3f}")]):
        ax[r,c].imshow(im); ax[r,c].axis("off")
        ax[r,c].set_title(ti,fontsize=9.5,color="#111")
    ax[r,0].text(0,-0.13,f'[{k}]  "{ins[:58]}"',transform=ax[r,0].transAxes,
                 fontsize=10.5,fontweight="bold",color="#8C3A1E")
plt.tight_layout(); plt.savefig(HERE/"qualitative.png",dpi=130,bbox_inches="tight")
print("saved", HERE/"qualitative.png")
print(f"mean IoU on these {n}: {np.mean([r[6] for r in rows]):.4f}")
