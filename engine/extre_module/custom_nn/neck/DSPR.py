import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, (k - 1) // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FixedGradientProbe(nn.Module):
    """Parameter-free gradient response on feature maps."""

    def __init__(self, eps=1e-6):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x, persistent=False)
        self.register_buffer("kernel_y", kernel_y, persistent=False)
        self.eps = eps

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        response = torch.sqrt(gx * gx + gy * gy + self.eps)
        norm = response.mean(dim=(2, 3), keepdim=True).detach().clamp_min(self.eps)
        return torch.sigmoid(response / norm - 1.0)


class DSPR(nn.Module):
    """Detail-Semantic Proxy Router.

    DSPR builds a compact P2 proxy from shallow detail and P3 semantics. The
    proxy is used for routing/fusion, not as an extra detection level, so it
    improves small-object detail without the heavy P2 decoder cost.
    """

    def __init__(self, c1, c2, semantic_weight=0.2, gate_ratio=0.5, export_gate=True):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            p2_channels, p3_channels = c1[0], c1[1]
        else:
            p2_channels, p3_channels = c1, c1

        gate_channels = max(int(c2 * gate_ratio), 16)
        self.semantic_weight = float(semantic_weight)
        self.export_gate = bool(export_gate)

        self.gradient_probe = FixedGradientProbe()
        self.p2_proj = ConvBNAct(p2_channels, c2, 1, 1)
        self.p3_proj = ConvBNAct(p3_channels, c2, 1, 1)
        self.detail_embed = nn.Sequential(
            ConvBNAct(1, gate_channels, 3, 1),
            nn.Conv2d(gate_channels, c2, 1, bias=True),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(c2, gate_channels, 1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, c2, 1, bias=True),
        )
        self.local_refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.out_proj = ConvBNAct(c2, c2, 1, 1)
        self.dga_map = None

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            p2, p3 = x
        else:
            p2, p3 = x, x

        p2_base = self.p2_proj(p2)
        p3_sem = self.p3_proj(p3)
        p3_sem = F.interpolate(p3_sem, size=p2_base.shape[-2:], mode="nearest")

        detail = self.gradient_probe(p2)
        gate = torch.sigmoid(self.detail_embed(detail) + self.semantic_gate(p3_sem))
        if self.training and self.export_gate:
            self.dga_map = gate.mean(dim=1, keepdim=True)
        else:
            self.dga_map = None

        refined = self.local_refine(p2_base)
        out = p2_base + refined * gate + self.semantic_weight * p3_sem
        return self.out_proj(out)


class CGRF(nn.Module):
    """Cross-Granularity Receptive-Field Fusion.

    CGRF injects the DSPR proxy into a deeper feature map through a gated
    bottleneck. It is intended for P3/P4 enhancement before the neck path.
    """

    def __init__(self, c1, c2, gate_ratio=0.5, export_gate=False):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            deep_channels, proxy_channels = c1[0], c1[1]
        else:
            deep_channels, proxy_channels = c1, c1

        gate_channels = max(int(c2 * gate_ratio), 16)
        self.export_gate = bool(export_gate)

        self.deep_proj = ConvBNAct(deep_channels, c2, 1, 1)
        self.proxy_proj = ConvBNAct(proxy_channels, c2, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(c2 * 2, gate_channels, 1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(c2 * 2, c2, 1, 1),
            ConvBNAct(c2, c2, 3, 1, g=c2),
            ConvBNAct(c2, c2, 1, 1),
        )
        self.dga_map = None

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            deep, proxy = x
        else:
            deep, proxy = x, x

        deep = self.deep_proj(deep)
        proxy = F.interpolate(proxy, size=deep.shape[-2:], mode="nearest")
        proxy = self.proxy_proj(proxy)
        gate = self.gate(torch.cat([deep, proxy], dim=1))
        if self.training and self.export_gate:
            self.dga_map = gate.mean(dim=1, keepdim=True)
        else:
            self.dga_map = None

        routed = proxy * gate
        return deep + self.fuse(torch.cat([deep, routed], dim=1))
