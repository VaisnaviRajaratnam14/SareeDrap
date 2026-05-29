"""
Saree Try-On Model Architecture
================================
VITON-HD style virtual try-on network adapted for saree draping.

Components:
  FeatureEncoder        – Shared backbone (ResNet-style)
  GarmentEncoder        – Encodes saree / blouse texture
  PoseEncoder           – Encodes 18-ch Gaussian pose heatmap
  SegmentationEncoder   – Encodes body segmentation mask
  SareeWarpingNetwork   – GMM: geometric matching + TPS warp
  TryOnGenerator        – U-Net: composites warped garment onto body
  Discriminator         – PatchGAN for adversarial training
  SareeTryOnModel       – Full end-to-end wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ── Utility Blocks ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with instance normalisation."""
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownBlock(nn.Module):
    """Stride-2 conv + norm + activation."""
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        layers: list = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Transpose conv + norm + ReLU for U-Net decoder."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([x, skip], dim=1))


# ── Encoders ───────────────────────────────────────────────────────────────────

class FeatureEncoder(nn.Module):
    """
    General-purpose feature backbone.
    Produces multi-scale feature maps for correlation matching.

    Input:  (B, in_ch, H, W)
    Output: list of feature maps at 1/2, 1/4, 1/8, 1/16 resolution
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.enc2 = DownBlock(base_ch,    base_ch * 2)
        self.enc3 = DownBlock(base_ch*2,  base_ch * 4)
        self.enc4 = DownBlock(base_ch*4,  base_ch * 8)
        self.enc5 = DownBlock(base_ch*8,  base_ch * 8)
        self.res  = nn.Sequential(*[ResBlock(base_ch * 8) for _ in range(4)])

    def forward(self, x: torch.Tensor):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        f5 = self.res(f5)
        return [f1, f2, f3, f4, f5]


class GarmentEncoder(nn.Module):
    """
    Encodes the garment (saree / blouse) image into a feature map.
    Also accepts an optional fabric-type embedding.

    Input:  garment BGR (B, 3, H, W) + optional fabric_emb (B, 8)
    Output: feature map (B, 256, H/8, W/8)
    """
    def __init__(self, base_ch: int = 64, n_fabric_types: int = 5,
                 emb_dim: int = 8):
        super().__init__()
        self.fabric_emb = nn.Embedding(n_fabric_types, emb_dim)
        # +emb_dim injected via a 1x1 conv after first layer
        self.enc = FeatureEncoder(in_ch=3, base_ch=base_ch)
        self.fabric_proj = nn.Sequential(
            nn.Linear(emb_dim, base_ch * 8),
            nn.ReLU(),
        )

    def forward(self, garment: torch.Tensor,
                fabric_idx: Optional[torch.Tensor] = None):
        feats = self.enc(garment)
        if fabric_idx is not None:
            fab = self.fabric_emb(fabric_idx)            # B x emb_dim
            fab = self.fabric_proj(fab)                  # B x 512
            fab = fab.unsqueeze(-1).unsqueeze(-1)        # B x 512 x 1 x 1
            feats[-1] = feats[-1] + fab.expand_as(feats[-1])
        return feats


class PoseEncoder(nn.Module):
    """
    Encodes 18-channel Gaussian pose heatmap into spatial features.

    Input:  pose heatmap (B, 18, H, W)
    Output: feature map (B, 128, H/4, W/4)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(18, 32, 3, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), ResBlock(128),
        )

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        return self.net(heatmap)


class SegmentationEncoder(nn.Module):
    """
    Encodes binary body segmentation mask into spatial features.

    Input:  mask (B, 1, H, W)
    Output: feature map (B, 64, H/4, W/4)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(mask)


# ── Saree Warping Network (GMM) ────────────────────────────────────────────────

class CorrelationLayer(nn.Module):
    """Cross-correlation between person and garment feature maps."""
    def forward(self, fp: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fp.shape
        fp = F.normalize(fp.view(b, c, -1), dim=1)   # B x C x HW
        fg = F.normalize(fg.view(b, c, -1), dim=1)
        corr = torch.bmm(fp.permute(0, 2, 1), fg)    # B x HW x HW
        return F.relu(corr).view(b, h * w, h, w)


class ThetaRegressor(nn.Module):
    """Predicts TPS (Thin Plate Spline) control point offsets."""
    _POOL_SIZE = 6   # fixed spatial size after adaptive pool

    def __init__(self, in_ch: int, grid_size: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.pool = nn.AdaptiveAvgPool2d(self._POOL_SIZE)
        flat_dim  = in_ch * self._POOL_SIZE * self._POOL_SIZE
        n_pts     = grid_size * grid_size * 2
        self.net  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_pts), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.pool(x)         # pool spatial dims to fixed size
        theta = self.net(x)
        return theta.view(-1, self.grid_size ** 2, 2)


class SareeWarpingNetwork(nn.Module):
    """
    Geometric Matching Module (GMM).
    Warps the garment to align with body pose using a learned TPS grid.

    Inputs:
      person_with_pose  (B, 3+18,   H, W)  — person RGB + pose heatmap
      garment           (B, 3,      H, W)  — saree / blouse image
      seg_mask          (B, 1,      H, W)  — body mask (optional)

    Outputs:
      warped_garment    (B, 3,      H, W)
      warp_grid         (B, H, W,   2)     — for regularisation loss
    """
    def __init__(self, grid_size: int = 5, img_h: int = 256, img_w: int = 192,
                 base_ch: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.img_h     = img_h
        self.img_w     = img_w

        self.person_enc  = FeatureEncoder(in_ch=3 + 18, base_ch=base_ch)
        self.garment_enc = FeatureEncoder(in_ch=3,      base_ch=base_ch)
        self.corr        = CorrelationLayer()
        # ThetaRegressor takes corr output channels and pools to fixed size
        # FeatureEncoder final feature map is at 1/16 resolution (4x stride-2 downs)
        # CorrelationLayer output: (B, feat_h*feat_w, feat_h, feat_w)
        feat_h   = img_h // 16
        feat_w   = img_w // 16
        corr_ch  = feat_h * feat_w          # channel count from CorrelationLayer
        self.theta = ThetaRegressor(corr_ch, grid_size)

    def _build_grid(self, theta: torch.Tensor,
                    b: int, device: torch.device) -> torch.Tensor:
        g  = self.grid_size
        xs = torch.linspace(-1, 1, g, device=device)
        ys = torch.linspace(-1, 1, g, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        base = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        base = base.unsqueeze(0).expand(b, -1, -1)
        ctrl = (base + theta.clamp(-0.5, 0.5)).view(b, g, g, 2).permute(0, 3, 1, 2)
        grid = F.interpolate(ctrl, size=(self.img_h, self.img_w),
                             mode="bilinear", align_corners=True)
        return grid.permute(0, 2, 3, 1)  # B x H x W x 2

    def forward(self, person_with_pose: torch.Tensor,
                garment: torch.Tensor,
                seg_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        fp = self.person_enc(person_with_pose)[-1]
        fg = self.garment_enc(garment)[-1]

        # Optionally weight person features by segmentation
        if seg_mask is not None:
            seg_down = F.adaptive_avg_pool2d(seg_mask.float(), fp.shape[-2:])
            fp = fp * (0.5 + 0.5 * seg_down)

        corr  = self.corr(fp, fg)    # (B, HW, H, W)
        theta = self.theta(corr)    # pools then regresses
        grid  = self._build_grid(theta, garment.size(0), garment.device)
        warped = F.grid_sample(garment, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return warped, grid


# ── Try-On Generator (U-Net) ───────────────────────────────────────────────────

class TryOnGenerator(nn.Module):
    """
    U-Net based composite generator (6 down-blocks, safe for 256x192 input).

    Inputs:  (B, in_ch, H, W)
      in_ch = person(3) + warped_saree(3) + warped_blouse(3)
              + pose_heatmap(18) + seg_mask(1) + style_emb projected(1)
            = 29 channels

    Outputs:
      rendered  (B, 3, H, W) — final composite RGB
      alpha     (B, 1, H, W) — blend mask [0,1]
    """
    def __init__(self, in_ch: int = 29, base_ch: int = 64):
        super().__init__()
        c = base_ch
        # Encoder — 6 down-blocks  (H,W) → (H/2,W/2) each
        # d1: in_ch → c      (128×96)
        # d2: c     → c*2    (64×48)
        # d3: c*2   → c*4    (32×24)
        # d4: c*4   → c*8    (16×12)
        # d5: c*8   → c*8    (8×6)
        # d6: c*8   → c*8    (4×3)
        self.d1 = DownBlock(in_ch, c,    normalize=False)
        self.d2 = DownBlock(c,     c*2)
        self.d3 = DownBlock(c*2,   c*4)
        self.d4 = DownBlock(c*4,   c*8,  dropout=0.4)
        self.d5 = DownBlock(c*8,   c*8,  dropout=0.4)
        self.d6 = DownBlock(c*8,   c*8,  dropout=0.4)
        # Bottleneck 3×3 convs — safe at small spatial size (no reflection pad)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c*8, c*8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c*8), nn.ReLU(inplace=True),
            nn.Conv2d(c*8, c*8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c*8), nn.ReLU(inplace=True),
        )
        # Decoder — UpBlock(in_ch, out_ch) where in_ch = x_channels + skip_channels
        # (UpBlock cats x and skip BEFORE ConvTranspose2d)
        # u1: x=bot(c*8) + skip=d6(c*8) → in=c*16, out=c*8
        # u2: x=u1(c*8)  + skip=d5(c*8) → in=c*16, out=c*8
        # u3: x=u2(c*8)  + skip=d4(c*8) → in=c*16, out=c*4
        # u4: x=u3(c*4)  + skip=d3(c*4) → in=c*8,  out=c*2
        # u5: x=u4(c*2)  + skip=d2(c*2) → in=c*4,  out=c
        # u6: x=u5(c)    + skip=d1(c)   → in=c*2,  out=c
        self.u1 = UpBlock(c*16, c*8,  dropout=0.4)
        self.u2 = UpBlock(c*16, c*8,  dropout=0.4)
        self.u3 = UpBlock(c*16, c*4)
        self.u4 = UpBlock(c*8,  c*2)
        self.u5 = UpBlock(c*4,  c)
        self.u6 = UpBlock(c*2,  c)
        # Final output: 4-ch (RGB+alpha) from last u6 output
        self.out_head = nn.Sequential(
            nn.Conv2d(c, 4, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d1  = self.d1(x)           # (B, c,   128, 96)
        d2  = self.d2(d1)          # (B, c*2,  64, 48)
        d3  = self.d3(d2)          # (B, c*4,  32, 24)
        d4  = self.d4(d3)          # (B, c*8,  16, 12)
        d5  = self.d5(d4)          # (B, c*8,   8,  6)
        d6  = self.d6(d5)          # (B, c*8,   4,  3)
        bot = self.bottleneck(d6)  # (B, c*8,   4,  3)
        u1  = self.u1(bot, d6)     # (B, c*8,   8,  6)
        u2  = self.u2(u1,  d5)     # (B, c*8,  16, 12)
        u3  = self.u3(u2,  d4)     # (B, c*4,  32, 24)
        u4  = self.u4(u3,  d3)     # (B, c*2,  64, 48)
        u5  = self.u5(u4,  d2)     # (B, c,   128, 96)
        u6  = self.u6(u5,  d1)     # (B, c,   256,192)
        out      = self.out_head(u6)                          # (B, 4, 256,192)
        rendered = out[:, :3]
        alpha    = (out[:, 3:4] + 1.0) / 2.0
        person   = x[:, :3]
        composed = alpha * rendered + (1.0 - alpha) * person
        return composed, alpha


# ── PatchGAN Discriminator ─────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN discriminator — classifies NxN patches as real/fake.
    Input: (person | composed) concatenated → (B, 6, H, W)
    """
    def __init__(self, in_ch: int = 6, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list = [
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(ch, min(ch * 2, 512), 4, 2, 1, bias=False),
                nn.InstanceNorm2d(min(ch * 2, 512)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, ch, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 4, 1, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, person: torch.Tensor,
                composed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([person, composed], dim=1))


# ── Full Model ─────────────────────────────────────────────────────────────────

class SareeTryOnModel(nn.Module):
    """
    End-to-end Saree Virtual Try-On Model.

    Forward pass:
      1. Warp saree onto body (SareeWarpingNetwork for GMM)
      2. Warp blouse onto body (SareeWarpingNetwork for blouse)
      3. Composite both onto person (TryOnGenerator)

    Inputs:
      person       (B, 3,  H, W)   person RGB (normalised -1..1)
      saree        (B, 3,  H, W)   saree texture
      blouse       (B, 3,  H, W)   blouse material
      pose_heatmap (B, 18, H, W)   Gaussian keypoint heatmap
      seg_mask     (B, 1,  H, W)   binary body mask
      fabric_idx   (B,)             int index: 0=silk,1=cotton,2=georgette,3=chiffon,4=other
      style_idx    (B,)             int index: 0=nivi,1=bridal,2=hanging,3=gujarati

    Outputs (dict):
      warped_saree   (B, 3, H, W)
      warped_blouse  (B, 3, H, W)
      rendered       (B, 3, H, W)  final composite
      alpha_mask     (B, 1, H, W)  blending mask
      saree_grid     (B, H, W, 2)  for reg loss
      blouse_grid    (B, H, W, 2)  for reg loss
    """

    FABRIC_TYPES  = ["silk", "cotton", "georgette", "chiffon", "other"]
    DRAPING_STYLES = ["nivi", "bridal", "hanging", "gujarati"]

    def __init__(self, img_h: int = 256, img_w: int = 192,
                 grid_size: int = 5, base_ch: int = 64):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        # Style embedding injected into generator via extra channel
        n_styles = len(self.DRAPING_STYLES)
        self.style_emb = nn.Embedding(n_styles, 64)
        self.style_proj = nn.Sequential(
            nn.Linear(64, img_h * img_w),
            nn.Tanh(),
        )

        # Warping networks
        self.saree_warp  = SareeWarpingNetwork(grid_size, img_h, img_w, base_ch)
        self.blouse_warp = SareeWarpingNetwork(grid_size, img_h, img_w, base_ch // 2)

        # Generator:
        # person(3) + warped_saree(3) + warped_blouse(3)
        # + pose(18) + seg(1) + style(1) = 29
        self.generator = TryOnGenerator(in_ch=29, base_ch=base_ch)

    def forward(
        self,
        person:       torch.Tensor,
        saree:        torch.Tensor,
        blouse:       torch.Tensor,
        pose_heatmap: torch.Tensor,
        seg_mask:     torch.Tensor,
        fabric_idx:   Optional[torch.Tensor] = None,
        style_idx:    Optional[torch.Tensor] = None,
    ) -> dict:
        b, _, h, w = person.shape

        # Style embedding → spatial map
        if style_idx is not None:
            s_emb = self.style_emb(style_idx)       # B x 64
            s_map = self.style_proj(s_emb)           # B x H*W
            s_map = s_map.view(b, 1, h, w)           # B x 1 x H x W
        else:
            s_map = torch.zeros(b, 1, h, w, device=person.device)

        # Warp saree
        person_pose         = torch.cat([person, pose_heatmap], dim=1)  # B x 21
        warped_saree, sg    = self.saree_warp(person_pose, saree, seg_mask)

        # Warp blouse
        warped_blouse, bg   = self.blouse_warp(person_pose, blouse, seg_mask)

        # Generator input
        gen_input = torch.cat([
            person, warped_saree, warped_blouse,
            pose_heatmap, seg_mask, s_map
        ], dim=1)   # B x 29 x H x W

        rendered, alpha = self.generator(gen_input)

        return {
            "warped_saree":  warped_saree,
            "warped_blouse": warped_blouse,
            "rendered":      rendered,
            "alpha_mask":    alpha,
            "saree_grid":    sg,
            "blouse_grid":   bg,
        }

    @classmethod
    def fabric_index(cls, name: str) -> int:
        name = name.lower().strip()
        return cls.FABRIC_TYPES.index(name) if name in cls.FABRIC_TYPES else 4

    @classmethod
    def style_index(cls, name: str) -> int:
        name = name.lower().strip()
        return cls.DRAPING_STYLES.index(name) if name in cls.DRAPING_STYLES else 0
