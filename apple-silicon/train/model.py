"""Text-conditioned edit-region head over frozen FastSAM prototypes.

The whole idea in one line:

    FastSAM already emits 32 prototype masks for the frame; normally the
    coefficients that combine them come from a DETECTION, which is exactly why
    its output can only ever be an existing object. We predict the coefficients
    from the INSTRUCTION instead. Prototypes are spatial basis functions, not
    objects, so a text-derived combination can describe empty space.

    mask = sigmoid( sum_i  c_i(text, image) * proto_i  +  b )

Frozen: the YOLOv8x-seg backbone and Proto module (71.75M), and the CLIP text
encoder. Trainable: only the fusion MLP below — about 1M parameters, which is
why this fits on a laptop.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class TextCoeffHead(nn.Module):
    """(CLIP text 512-d, pooled image context 640-d) -> 32 coefficients + bias."""

    def __init__(self, text_dim=512, img_dim=640, nm=32, hidden=512, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.img_proj  = nn.Sequential(nn.LayerNorm(img_dim),  nn.Linear(img_dim,  hidden), nn.GELU())
        # FiLM-style modulation: the instruction gates the image context rather
        # than being concatenated, so the text cannot be ignored by the MLP.
        self.gamma = nn.Linear(hidden, hidden)
        self.beta  = nn.Linear(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, nm + 1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, text_emb, img_ctx):
        t = self.text_proj(text_emb)
        v = self.img_proj(img_ctx)
        v = self.gamma(t) * v + self.beta(t)      # instruction modulates image
        out = self.mlp(v)
        return out[:, :-1], out[:, -1:]           # coefficients, bias


class EditRegionModel(nn.Module):
    """Frozen FastSAM prototypes + frozen CLIP text + the trainable head."""

    def __init__(self, weights="best.pt", device="cpu", nm=32):
        super().__init__()
        from ultralytics import YOLO
        import clip
        self.device = device
        y = YOLO(weights)
        self.det = y.model.float().eval().to(device)
        for p in self.det.parameters(): p.requires_grad_(False)
        self.clip, _ = clip.load("ViT-B/32", device=device)
        self.clip = self.clip.eval()
        for p in self.clip.parameters(): p.requires_grad_(False)
        self.head = TextCoeffHead(nm=nm).to(device)

        self._feats = {}
        self.det.model[9].register_forward_hook(self._grab("sppf"))
        self.det.model[-1].proto.register_forward_hook(self._grab("proto"))

    def _grab(self, name):
        def f(_m, _i, o): self._feats[name] = o
        return f

    @torch.no_grad()
    def encode_text(self, texts):
        import clip
        tok = clip.tokenize(texts, truncate=True).to(self.device)
        e = self.clip.encode_text(tok).float()
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def backbone(self, images):
        """Run the frozen detector once; return (prototypes, pooled context)."""
        self._feats.clear()
        self.det(images)
        proto = self._feats["proto"]                       # (B, 32, H/4, W/4)
        ctx = F.adaptive_avg_pool2d(self._feats["sppf"], 1).flatten(1)   # (B, 640)
        return proto, ctx

    def forward(self, images, texts):
        proto, ctx = self.backbone(images)
        temb = self.encode_text(texts)
        coef, bias = self.head(temb, ctx)                  # (B,32), (B,1)
        logits = torch.einsum("bc,bchw->bhw", coef, proto) + bias[:, :, None]
        return logits.unsqueeze(1)                         # (B,1,H/4,W/4)

    def trainable_parameters(self):
        return [p for p in self.head.parameters() if p.requires_grad]


def dice_bce_loss(logits, target, eps=1.0, w_dice=1.0, w_bce=1.0, pos_weight=None):
    """BCE keeps per-pixel calibration; Dice handles the heavy class imbalance.

    Edit regions average ~10% of the frame, so BCE alone collapses to predicting
    background everywhere. Dice is scale-invariant and stops that.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    num = 2*(p*target).sum(dim=(1,2,3)) + eps
    den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    dice = 1 - (num/den).mean()
    return w_bce*bce + w_dice*dice, bce.detach(), dice.detach()


@torch.no_grad()
def mask_metrics(logits, target, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p*target).sum(dim=(1,2,3))
    union = ((p+target) > 0).float().sum(dim=(1,2,3)).clamp(min=1)
    dice = (2*inter / (p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))).clamp(min=1))
    return dict(iou=(inter/union).mean().item(), dice=dice.mean().item(),
                pred_frac=p.mean().item(), true_frac=target.mean().item())
