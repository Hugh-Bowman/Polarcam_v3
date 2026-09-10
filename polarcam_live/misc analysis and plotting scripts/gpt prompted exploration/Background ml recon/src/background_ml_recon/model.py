from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _split_channels(channels: int, ratio_global: float) -> tuple[int, int]:
    ratio = float(max(0.0, min(1.0, ratio_global)))
    c_global = int(round(channels * ratio))
    c_global = max(0, min(channels, c_global))
    c_local = channels - c_global
    return c_local, c_global


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        chs = [base_channels * (2**i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for ch in chs:
            self.down_blocks.append(ConvBlock(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch

        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        up_in = chs[-1] * 2
        for ch in reversed(chs):
            self.up_transpose.append(nn.ConvTranspose2d(up_in, ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(ch * 2, ch))
            up_in = ch

        self.head = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for block, pool in zip(self.down_blocks, self.pools):
            h = block(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)

        for up_t, up_b, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            h = up_t(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = up_b(h)

        return self.head(h)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class PretrainedMaskGuidedUNet(nn.Module):
    """
    Image branch: pretrained ResNet18 encoder on grayscale image (repeated to 3 channels).
    Mask branch: separate mask pyramid injected into decoder skips.
    This keeps mask semantically separate from image color channels.
    """

    def __init__(self, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        enc = resnet18(weights=weights)

        self.enc_conv1 = enc.conv1
        self.enc_bn1 = enc.bn1
        self.enc_relu = enc.relu
        self.enc_maxpool = enc.maxpool
        self.enc_layer1 = enc.layer1
        self.enc_layer2 = enc.layer2
        self.enc_layer3 = enc.layer3
        self.enc_layer4 = enc.layer4

        self.mask_proj1 = nn.Conv2d(1, 16, kernel_size=1)
        self.mask_proj2 = nn.Conv2d(1, 16, kernel_size=1)
        self.mask_proj3 = nn.Conv2d(1, 32, kernel_size=1)
        self.mask_proj4 = nn.Conv2d(1, 64, kernel_size=1)
        self.mask_proj5 = nn.Conv2d(1, 128, kernel_size=1)

        self.bottleneck = ConvBlock(512 + 128, 512)
        self.up4 = UpBlock(512 + (256 + 64), 256)
        self.up3 = UpBlock(256 + (128 + 32), 128)
        self.up2 = UpBlock(128 + (64 + 16), 64)
        self.up1 = UpBlock(64 + (64 + 16), 32)
        self.up0 = ConvBlock(32 + 1, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    @staticmethod
    def _mask_to_size(mask: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        if mask.shape[-2:] == size_hw:
            return mask
        return F.interpolate(mask, size=size_hw, mode="area")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x[:, 0:1]
        m = x[:, 1:2]

        img3 = img.repeat(1, 3, 1, 1)
        e1 = self.enc_relu(self.enc_bn1(self.enc_conv1(img3)))
        e2 = self.enc_layer1(self.enc_maxpool(e1))
        e3 = self.enc_layer2(e2)
        e4 = self.enc_layer3(e3)
        e5 = self.enc_layer4(e4)

        m1 = self.mask_proj1(self._mask_to_size(m, e1.shape[-2:]))
        m2 = self.mask_proj2(self._mask_to_size(m, e2.shape[-2:]))
        m3 = self.mask_proj3(self._mask_to_size(m, e3.shape[-2:]))
        m4 = self.mask_proj4(self._mask_to_size(m, e4.shape[-2:]))
        m5 = self.mask_proj5(self._mask_to_size(m, e5.shape[-2:]))

        b = self.bottleneck(torch.cat([e5, m5], dim=1))
        d4 = self.up4(b, torch.cat([e4, m4], dim=1))
        d3 = self.up3(d4, torch.cat([e3, m3], dim=1))
        d2 = self.up2(d3, torch.cat([e2, m2], dim=1))
        d1 = self.up1(d2, torch.cat([e1, m1], dim=1))

        d0 = F.interpolate(d1, size=m.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.up0(torch.cat([d0, m], dim=1))
        return self.head(d0)


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        spec = torch.fft.rfft2(x, norm="ortho")
        spec_ri = torch.view_as_real(spec)
        spec_ri = spec_ri.permute(0, 1, 4, 2, 3).contiguous()
        spec_ri = spec_ri.view(b, c * 2, h, spec_ri.shape[-1])

        y = self.act(self.bn(self.conv(spec_ri)))

        c2 = y.shape[1] // 2
        y = y.view(b, c2, 2, h, y.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
        y_complex = torch.view_as_complex(y)
        return torch.fft.irfft2(y_complex, s=(h, w), norm="ortho")


class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels, out_channels)
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.pre(x)
        x1 = self.fu(x0)
        return self.post(x0 + x1)


class FFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        self.in_cl, self.in_cg = _split_channels(in_channels, ratio_gin)
        self.out_cl, self.out_cg = _split_channels(out_channels, ratio_gout)

        self.l2l = None
        self.l2g = None
        self.g2l = None
        self.g2g_conv = None
        self.g2g_spec = None

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if self.in_cl > 0 and self.out_cl > 0:
            self.l2l = nn.Conv2d(self.in_cl, self.out_cl, **kwargs)
        if self.in_cl > 0 and self.out_cg > 0:
            self.l2g = nn.Conv2d(self.in_cl, self.out_cg, **kwargs)
        if self.in_cg > 0 and self.out_cl > 0:
            self.g2l = nn.Conv2d(self.in_cg, self.out_cl, **kwargs)
        if self.in_cg > 0 and self.out_cg > 0:
            if stride == 1:
                self.g2g_spec = SpectralTransform(self.in_cg, self.out_cg)
            else:
                self.g2g_conv = nn.Conv2d(self.in_cg, self.out_cg, **kwargs)

    @staticmethod
    def _add_or_set(dst: torch.Tensor | None, src: torch.Tensor | None) -> torch.Tensor | None:
        if src is None:
            return dst
        if dst is None:
            return src
        return dst + src

    def forward(self, x_l: torch.Tensor | None, x_g: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        out_l: torch.Tensor | None = None
        out_g: torch.Tensor | None = None

        if x_l is not None:
            if self.l2l is not None:
                out_l = self._add_or_set(out_l, self.l2l(x_l))
            if self.l2g is not None:
                out_g = self._add_or_set(out_g, self.l2g(x_l))

        if x_g is not None:
            if self.g2l is not None:
                out_l = self._add_or_set(out_l, self.g2l(x_g))
            if self.g2g_spec is not None:
                out_g = self._add_or_set(out_g, self.g2g_spec(x_g))
            if self.g2g_conv is not None:
                out_g = self._add_or_set(out_g, self.g2g_conv(x_g))

        return out_l, out_g


class FFCResBlock(nn.Module):
    def __init__(self, channels: int, ratio_g: float = 0.5) -> None:
        super().__init__()
        self.c_local, self.c_global = _split_channels(channels, ratio_g)

        self.ffc1 = FFC(channels, channels, ratio_gin=ratio_g, ratio_gout=ratio_g, stride=1)
        self.ffc2 = FFC(channels, channels, ratio_gin=ratio_g, ratio_gout=ratio_g, stride=1)

        self.bn1_l = nn.BatchNorm2d(self.c_local) if self.c_local > 0 else None
        self.bn1_g = nn.BatchNorm2d(self.c_global) if self.c_global > 0 else None
        self.bn2_l = nn.BatchNorm2d(self.c_local) if self.c_local > 0 else None
        self.bn2_g = nn.BatchNorm2d(self.c_global) if self.c_global > 0 else None

        self.act = nn.ReLU(inplace=True)

    def _norm_act(
        self,
        x_l: torch.Tensor | None,
        x_g: torch.Tensor | None,
        bn_l: nn.BatchNorm2d | None,
        bn_g: nn.BatchNorm2d | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if x_l is not None and bn_l is not None:
            x_l = self.act(bn_l(x_l))
        if x_g is not None and bn_g is not None:
            x_g = self.act(bn_g(x_g))
        return x_l, x_g

    @staticmethod
    def _add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    def forward(self, x_l: torch.Tensor | None, x_g: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        id_l, id_g = x_l, x_g

        y_l, y_g = self.ffc1(x_l, x_g)
        y_l, y_g = self._norm_act(y_l, y_g, self.bn1_l, self.bn1_g)

        y_l, y_g = self.ffc2(y_l, y_g)
        y_l, y_g = self._norm_act(y_l, y_g, self.bn2_l, self.bn2_g)

        out_l = self._add(id_l, y_l)
        out_g = self._add(id_g, y_g)
        return out_l, out_g


class FFCStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio_g: float, num_res_blocks: int) -> None:
        super().__init__()
        self.project = FFC(
            in_channels=in_channels,
            out_channels=out_channels,
            ratio_gin=ratio_g,
            ratio_gout=ratio_g,
            stride=1,
            kernel_size=3,
            padding=1,
        )
        self.blocks = nn.ModuleList([FFCResBlock(out_channels, ratio_g=ratio_g) for _ in range(num_res_blocks)])

    def forward(self, x_l: torch.Tensor | None, x_g: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x_l, x_g = self.project(x_l, x_g)
        for blk in self.blocks:
            x_l, x_g = blk(x_l, x_g)
        return x_l, x_g


class LaMaMaskGuidedUNet(nn.Module):
    """
    Lightweight LaMa-style model:
    - Local/global channel split with FFC residual processing.
    - Separate mask branch projected into each scale (not merged as image channel semantics).
    - U-Net-like encoder/decoder around FFC stages.
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 24,
        ratio_global: float = 0.5,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.ratio_global = float(max(0.0, min(0.95, ratio_global)))
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        c1_l, _ = _split_channels(c1, self.ratio_global)
        c2_l, _ = _split_channels(c2, self.ratio_global)
        c3_l, _ = _split_channels(c3, self.ratio_global)
        c4_l, _ = _split_channels(c4, self.ratio_global)

        self.in_conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.mask1 = nn.Conv2d(1, c1_l, kernel_size=1) if c1_l > 0 else None
        self.mask2 = nn.Conv2d(1, c2_l, kernel_size=1) if c2_l > 0 else None
        self.mask3 = nn.Conv2d(1, c3_l, kernel_size=1) if c3_l > 0 else None
        self.mask4 = nn.Conv2d(1, c4_l, kernel_size=1) if c4_l > 0 else None

        self.enc1 = FFCStage(c1, c1, ratio_g=self.ratio_global, num_res_blocks=num_res_blocks)
        self.down1 = FFC(c1, c2, ratio_gin=self.ratio_global, ratio_gout=self.ratio_global, stride=2)

        self.enc2 = FFCStage(c2, c2, ratio_g=self.ratio_global, num_res_blocks=num_res_blocks)
        self.down2 = FFC(c2, c3, ratio_gin=self.ratio_global, ratio_gout=self.ratio_global, stride=2)

        self.enc3 = FFCStage(c3, c3, ratio_g=self.ratio_global, num_res_blocks=num_res_blocks)
        self.down3 = FFC(c3, c4, ratio_gin=self.ratio_global, ratio_gout=self.ratio_global, stride=2)

        self.bottleneck = FFCStage(c4, c4, ratio_g=self.ratio_global, num_res_blocks=max(2, num_res_blocks + 1))

        self.dec3 = FFCStage(c4 + c3, c3, ratio_g=self.ratio_global, num_res_blocks=num_res_blocks)
        self.dec2 = FFCStage(c3 + c2, c2, ratio_g=self.ratio_global, num_res_blocks=num_res_blocks)
        self.dec1 = FFCStage(c2 + c1, c1, ratio_g=self.ratio_global, num_res_blocks=num_res_blocks)

        self.out_refine = nn.Sequential(
            nn.Conv2d(c1 + 1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

    @staticmethod
    def _to_size(mask: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        if mask.shape[-2:] == size_hw:
            return mask
        return F.interpolate(mask, size=size_hw, mode="area")

    @staticmethod
    def _merge_local_global(x_l: torch.Tensor | None, x_g: torch.Tensor | None) -> torch.Tensor:
        if x_l is None and x_g is None:
            raise RuntimeError("Both local and global branches are None.")
        if x_l is None:
            return x_g
        if x_g is None:
            return x_l
        return torch.cat([x_l, x_g], dim=1)

    @staticmethod
    def _split_for_stage(x: torch.Tensor, ratio_g: float) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        c_local, c_global = _split_channels(x.shape[1], ratio_g)
        x_l = x[:, :c_local] if c_local > 0 else None
        x_g = x[:, c_local : c_local + c_global] if c_global > 0 else None
        return x_l, x_g

    def _add_mask(self, x_l: torch.Tensor | None, m_feat: torch.Tensor) -> torch.Tensor | None:
        if x_l is None:
            return m_feat
        return x_l + m_feat

    def _project_mask(
        self,
        mask: torch.Tensor,
        size_hw: tuple[int, int],
        proj: nn.Conv2d | None,
    ) -> torch.Tensor | None:
        if proj is None:
            return None
        return proj(self._to_size(mask, size_hw))

    def _upsample_lg(
        self,
        x_l: torch.Tensor | None,
        x_g: torch.Tensor | None,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if x_l is not None:
            x_l = F.interpolate(x_l, size=target_hw, mode="bilinear", align_corners=False)
        if x_g is not None:
            x_g = F.interpolate(x_g, size=target_hw, mode="bilinear", align_corners=False)
        return x_l, x_g

    def _cat_lg(
        self,
        a_l: torch.Tensor | None,
        a_g: torch.Tensor | None,
        b_l: torch.Tensor | None,
        b_g: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        out_l = None
        out_g = None
        if a_l is not None and b_l is not None:
            out_l = torch.cat([a_l, b_l], dim=1)
        elif a_l is not None:
            out_l = a_l
        elif b_l is not None:
            out_l = b_l

        if a_g is not None and b_g is not None:
            out_g = torch.cat([a_g, b_g], dim=1)
        elif a_g is not None:
            out_g = a_g
        elif b_g is not None:
            out_g = b_g
        return out_l, out_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x[:, 0:1]
        mask = x[:, 1:2]

        x0 = self.in_conv(img)
        x_l, x_g = self._split_for_stage(x0, self.ratio_global)

        m1 = self._project_mask(mask, x0.shape[-2:], self.mask1)
        if m1 is not None:
            x_l = self._add_mask(x_l, m1)
        e1_l, e1_g = self.enc1(x_l, x_g)

        d1_l, d1_g = self.down1(e1_l, e1_g)
        e2_hw = d1_l.shape[-2:] if d1_l is not None else d1_g.shape[-2:]
        m2 = self._project_mask(mask, e2_hw, self.mask2)
        if m2 is not None:
            d1_l = self._add_mask(d1_l, m2)
        e2_l, e2_g = self.enc2(d1_l, d1_g)

        d2_l, d2_g = self.down2(e2_l, e2_g)
        e3_hw = d2_l.shape[-2:] if d2_l is not None else d2_g.shape[-2:]
        m3 = self._project_mask(mask, e3_hw, self.mask3)
        if m3 is not None:
            d2_l = self._add_mask(d2_l, m3)
        e3_l, e3_g = self.enc3(d2_l, d2_g)

        d3_l, d3_g = self.down3(e3_l, e3_g)
        b_hw = d3_l.shape[-2:] if d3_l is not None else d3_g.shape[-2:]
        m4 = self._project_mask(mask, b_hw, self.mask4)
        if m4 is not None:
            d3_l = self._add_mask(d3_l, m4)
        b_l, b_g = self.bottleneck(d3_l, d3_g)

        b_l, b_g = self._upsample_lg(b_l, b_g, e3_hw)
        u3_l, u3_g = self._cat_lg(b_l, b_g, e3_l, e3_g)
        u3_l, u3_g = self.dec3(u3_l, u3_g)

        u3_l, u3_g = self._upsample_lg(u3_l, u3_g, e2_hw)
        u2_l, u2_g = self._cat_lg(u3_l, u3_g, e2_l, e2_g)
        u2_l, u2_g = self.dec2(u2_l, u2_g)

        e1_hw = e1_l.shape[-2:] if e1_l is not None else e1_g.shape[-2:]
        u2_l, u2_g = self._upsample_lg(u2_l, u2_g, e1_hw)
        u1_l, u1_g = self._cat_lg(u2_l, u2_g, e1_l, e1_g)
        u1_l, u1_g = self.dec1(u1_l, u1_g)

        full = self._merge_local_global(u1_l, u1_g)
        full = F.interpolate(full, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_refine(torch.cat([full, mask], dim=1))


class LaMaBigAdapter(nn.Module):
    """
    Uses official Big-LaMa generator as a frozen (or mostly frozen) backbone and
    learns a small grayscale refinement adapter for domain-specific microscopy data.

    Input is expected to be [masked_input, mask] (2 channels), where mask=1 is hole.
    Output is delta wrt masked_input, matching the existing training pipeline:
      pred_full = masked_input + mask * pred_delta
    """

    def __init__(
        self,
        lama_model_dir: str = "third_party/models/big-lama",
        lama_repo_dir: str = "third_party/lama",
        adapter_channels: int = 16,
        freeze_lama: bool = True,
        unfreeze_last_n_params: int = 0,
    ) -> None:
        super().__init__()
        self.lama_generator = self._load_lama_generator(
            lama_model_dir=Path(lama_model_dir),
            lama_repo_dir=Path(lama_repo_dir),
        )

        # Freeze by default for low compute fine-tuning.
        if freeze_lama:
            for p in self.lama_generator.parameters():
                p.requires_grad = False

        # Optional tiny partial unfreeze for extra capacity.
        if int(unfreeze_last_n_params) > 0:
            params = list(self.lama_generator.parameters())
            n = min(len(params), int(unfreeze_last_n_params))
            for p in params[-n:]:
                p.requires_grad = True

        ch = int(max(4, adapter_channels))
        # Adapter sees masked input, mask, LaMa gray prediction, and known-region prior.
        self.adapter = nn.Sequential(
            nn.Conv2d(4, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, kernel_size=1),
        )

    @staticmethod
    def _load_lama_generator(lama_model_dir: Path, lama_repo_dir: Path) -> nn.Module:
        if not lama_model_dir.exists():
            raise FileNotFoundError(f"LaMa model dir not found: {lama_model_dir}")
        if not lama_repo_dir.exists():
            raise FileNotFoundError(f"LaMa repo dir not found: {lama_repo_dir}")

        if str(lama_repo_dir) not in sys.path:
            sys.path.insert(0, str(lama_repo_dir))

        try:
            from omegaconf import OmegaConf  # type: ignore
            from saicinpainting.training.trainers import load_checkpoint  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed importing LaMa modules. Ensure third_party/lama dependencies are installed."
            ) from exc

        train_config_path = lama_model_dir / "config.yaml"
        checkpoint_path = lama_model_dir / "models" / "best.ckpt"
        if not train_config_path.exists():
            raise FileNotFoundError(f"LaMa config missing: {train_config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LaMa checkpoint missing: {checkpoint_path}")

        train_config = OmegaConf.load(train_config_path)
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location="cpu")
        gen = model.generator
        gen.eval()
        return gen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[:,0:1] is masked input by dataset design.
        masked_input = x[:, 0:1]
        mask = x[:, 1:2]

        lama_gray = self.predict_lama_full(masked_input, mask)
        # Known-region prior stabilizes refinement on visible pixels.
        known_prior = masked_input * (1.0 - mask)
        adapter_in = torch.cat([masked_input, mask, lama_gray, known_prior], dim=1)
        refine = self.adapter(adapter_in)

        pred_full = lama_gray + refine
        pred_delta = pred_full - masked_input
        return pred_delta

    def predict_lama_full(self, masked_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lama_img3 = masked_input.repeat(1, 3, 1, 1)
        lama_in = torch.cat([lama_img3, mask], dim=1)

        # Run LaMa backbone; gradients usually disabled via requires_grad=False.
        lama_rgb = self.lama_generator(lama_in)
        if lama_rgb.shape[-2:] != masked_input.shape[-2:]:
            lama_rgb = F.interpolate(lama_rgb, size=masked_input.shape[-2:], mode="bilinear", align_corners=False)
        lama_gray = lama_rgb.mean(dim=1, keepdim=True)
        return lama_gray


def build_model(model_cfg: dict) -> nn.Module:
    model_name = str(model_cfg.get("name", "pretrained_mask_guided_unet")).lower()
    if model_name == "unetsmall":
        return UNetSmall(
            in_channels=int(model_cfg.get("in_channels", 2)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 16)),
            depth=int(model_cfg.get("depth", 3)),
        )
    if model_name == "pretrained_mask_guided_unet":
        return PretrainedMaskGuidedUNet(
            out_channels=int(model_cfg.get("out_channels", 1)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    if model_name == "lama_mask_guided_unet":
        return LaMaMaskGuidedUNet(
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 24)),
            ratio_global=float(model_cfg.get("ratio_global", 0.5)),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        )
    if model_name == "lama_big_adapter":
        return LaMaBigAdapter(
            lama_model_dir=str(model_cfg.get("lama_model_dir", "third_party/models/big-lama")),
            lama_repo_dir=str(model_cfg.get("lama_repo_dir", "third_party/lama")),
            adapter_channels=int(model_cfg.get("adapter_channels", 16)),
            freeze_lama=bool(model_cfg.get("freeze_lama", True)),
            unfreeze_last_n_params=int(model_cfg.get("unfreeze_last_n_params", 0)),
        )
    raise ValueError(f"Unsupported model.name: {model_name}")
