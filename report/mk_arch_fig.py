import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZ="#E8E4DC"; TRAIN="#C9DCE8"; EDGE="#3A3833"; TXT="#26241F"
fig,ax=plt.subplots(figsize=(11,6.2)); ax.set_xlim(0,11); ax.set_ylim(0,6.2); ax.axis("off")

def box(x,y,w,h,label,sub="",fc=FROZ,fs=9.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",
        fc=fc,ec=EDGE,lw=1.1))
    ax.text(x+w/2,y+h/2+(0.11 if sub else 0),label,ha="center",va="center",
            fontsize=fs,color=TXT,weight="bold")
    if sub: ax.text(x+w/2,y+h/2-0.16,sub,ha="center",va="center",fontsize=8,color="#5A5852")

def arr(x1,y1,x2,y2,label="",rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,
        lw=1.0,color=EDGE,connectionstyle=f"arc3,rad={rad}"))
    if label: ax.text((x1+x2)/2,(y1+y2)/2+0.14,label,ha="center",fontsize=7.6,color="#5A5852")

# inputs
box(0.15,4.55,1.5,0.75,"Image","640 x 640 RGB",fc="#FFFFFF")
box(0.15,1.25,1.5,0.75,"Instruction",'"put a hat on the dog"',fc="#FFFFFF",fs=9)

# frozen branches
box(2.15,4.3,2.5,1.25,"YOLOv8x-seg  (frozen)","71.75 M params")
box(2.15,1.05,2.5,1.15,"CLIP ViT-B/32 text  (frozen)","63.43 M params",fs=9)

arr(1.65,4.93,2.15,4.93); arr(1.65,1.63,2.15,1.63)

# taps
box(5.15,5.0,2.0,0.62,"prototypes","(B, 32, 160, 160)",fs=9)
box(5.15,4.15,2.0,0.62,"SPPF pooled","(B, 640)",fs=9)
box(5.15,1.32,2.0,0.62,"text embedding","(B, 512)",fs=9)

arr(4.65,5.25,5.15,5.31,"model[-1].proto")
arr(4.65,4.62,5.15,4.46,"model[9]")
arr(4.65,1.63,5.15,1.63)

# head
box(7.7,2.35,2.9,2.35,"","",fc=TRAIN)
ax.text(9.15,4.46,"Trainable head",ha="center",fontsize=10.5,color=TXT,weight="bold")
ax.text(9.15,4.22,"v2  SpatialCoeffHead",ha="center",fontsize=8.6,color="#5A5852",style="italic")
for i,t in enumerate([
        "FiLM:  v = γ(t)·v + β(t)",
        "global branch → 33 coeffs",
        "field → (16, 20, 20)",
        "conv → (32, 20, 20)",
        "bilinear ↑ to 160 × 160",
        "c(x,y) = c_global + c_field(x,y)"]):
    ax.text(9.15,3.92-i*0.235,t,ha="center",fontsize=8,color="#3A3833")
ax.text(9.15,2.52,"4,693,633 params  ·  the only part that learns",
        ha="center",fontsize=7.6,color="#5A5852",style="italic")

arr(7.15,5.31,7.9,4.7,rad=-0.15)
arr(7.15,4.46,7.7,4.0)
arr(7.15,1.63,7.9,2.35,rad=0.15)

# output
box(7.7,0.72,2.9,0.82,"mask = σ( Σᵢ cᵢ(x,y)·protoᵢ + b )","(B, 1, 160, 160)",fc="#FFFFFF",fs=9.5)
arr(9.15,2.35,9.15,1.54)

# legend
ax.add_patch(FancyBboxPatch((0.15,0.22),1.5,0.34,boxstyle="round,pad=0.05",fc=FROZ,ec=EDGE,lw=0.9))
ax.text(0.9,0.39,"frozen",ha="center",va="center",fontsize=8,color=TXT)
ax.add_patch(FancyBboxPatch((1.8,0.22),1.5,0.34,boxstyle="round,pad=0.05",fc=TRAIN,ec=EDGE,lw=0.9))
ax.text(2.55,0.39,"trained",ha="center",va="center",fontsize=8,color=TXT)

plt.tight_layout(pad=0.3)
plt.savefig("DA2_figures/architecture.png",dpi=190,facecolor="white")
print("wrote DA2_figures/architecture.png")
