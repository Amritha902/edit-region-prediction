"""Architecture v2 — fixes derived from exp08/10/11, not from guessing.

Two measured problems with v1:

  (a) THE COEFFICIENTS ARE GLOBAL. v1 predicts one 32-vector for the whole
      image, so the mask is a single fixed linear combination of prototypes
      everywhere. Nothing lets the top-left of the frame use a different
      combination from the bottom-right. exp11 showed v1 saturates at 1.4M —
      more width does not help, because width is not the constraint.

  (b) THE BASIS IS TOO SMALL. exp10 measured the least-squares ceiling: adding
      12 smooth geometric functions lifts insert 0.197 -> 0.324 and remove
      0.464 -> 0.581. v1 reaches only 29-42% of even the unaugmented ceiling.

v2 addresses both:

  SpatialCoeffHead   predicts a coefficient FIELD at 20x20 instead of a single
                     vector, upsampled to prototype resolution. The mask becomes
                     sum_i c_i(x,y) * proto_i(x,y) — a spatially varying
                     combination. The instruction still drives it through FiLM,
                     and a global branch is retained so the text can set an
                     overall prior.

  geometric basis    the 12 functions exp10 found best per-function, appended to
                     the 32 prototypes. Cheap, fixed, no parameters.

exp10's discipline applies: any gain must beat a random-basis control of equal
size, so `basis="rand"` reproduces that control here.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

P = 160

def _geometric(p=P):
    yy, xx = np.mgrid[0:p, 0:p]
    yy = (yy/(p-1))*2-1; xx = (xx/(p-1))*2-1
    b = [np.ones_like(xx), xx, yy, xx*xx, yy*yy, xx*yy, xx**3, yy**3,
         np.sin(np.pi*xx), np.sin(np.pi*yy), np.cos(np.pi*xx), np.cos(np.pi*yy)]
    return torch.from_numpy(np.stack(b).astype(np.float32))

def _random(p=P, k=12, seed=1368):
    import cv2
    rng = np.random.default_rng(seed)
    r = np.stack([cv2.GaussianBlur(rng.standard_normal((p,p)).astype(np.float32),(0,0),12)
                  for _ in range(k)])
    r /= (r.std(axis=(1,2), keepdims=True)+1e-6)
    return torch.from_numpy(r)


class SpatialCoeffHead(nn.Module):
    """Instruction -> a coefficient FIELD over the prototype basis."""

    def __init__(self, text_dim=512, img_dim=640, nm=32, hidden=512,
                 grid=20, dropout=0.0, basis="geom", n_extra=12):
        super().__init__()
        self.grid = grid
        extra = {"geom": _geometric(), "rand": _random(k=n_extra), "none": None}[basis]
        if extra is not None:
            self.register_buffer("extra", extra)
            self.nb = nm + extra.shape[0]
        else:
            self.extra = None; self.nb = nm
        self.nm = nm

        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.img_proj  = nn.Sequential(nn.LayerNorm(img_dim),  nn.Linear(img_dim,  hidden), nn.GELU())
        self.gamma = nn.Linear(hidden, hidden); self.beta = nn.Linear(hidden, hidden)

        # global branch: the text sets an overall prior, as in v1
        self.glob = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, self.nb + 1))
        # spatial branch: a coefficient field, conditioned on the same fused vector
        self.field = nn.Sequential(
            nn.Linear(hidden, grid*grid*16), nn.GELU())
        self.field_conv = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, self.nb, 1))
        nn.init.zeros_(self.glob[-1].weight); nn.init.zeros_(self.glob[-1].bias)
        nn.init.zeros_(self.field_conv[-1].weight); nn.init.zeros_(self.field_conv[-1].bias)

    def basis(self, proto):
        if self.extra is None: return proto
        e = self.extra[None].expand(proto.shape[0], -1, -1, -1).to(proto.dtype)
        return torch.cat([proto, e], 1)

    def forward(self, text, ctx, proto):
        B = proto.shape[0]
        t = self.text_proj(text); v = self.img_proj(ctx)
        v = self.gamma(t)*v + self.beta(t)

        g = self.glob(v)                              # B, nb+1
        cg, bias = g[:, :-1], g[:, -1:]

        f = self.field(v).view(B, 16, self.grid, self.grid)
        cf = self.field_conv(f)                        # B, nb, grid, grid
        cf = F.interpolate(cf, size=(P, P), mode="bilinear", align_corners=False)

        c = cg[:, :, None, None] + cf                  # global prior + local field
        Bs = self.basis(proto)
        return (c*Bs).sum(1, keepdim=True) + bias[:, :, None, None]
