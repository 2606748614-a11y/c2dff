import torch
import torch.nn as nn
import torch.nn.functional as F


class BiSSM(nn.Module):
    """Pure PyTorch bidirectional Mamba-like block without causal-conv1d/mamba-ssm."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner_dim = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            bias=True,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv // 2,
        )

        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.x_proj = nn.Linear(self.inner_dim, self.inner_dim * 2, bias=False)

        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward_ssm(self, x):
        """Linear-memory selective scan approximation, implemented with stock torch ops."""
        gate, value = self.x_proj(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        state = torch.cumsum(gate * value, dim=1)
        normalizer = torch.cumsum(gate, dim=1).clamp_min(1e-4)
        y = state / normalizer
        return y + x * self.D

    def forward(self, x):
        _, length, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = F.silu(self.conv1d(x)[..., :length])
        x = x.transpose(1, 2)

        y_forward = self.forward_ssm(x)
        y_backward = self.forward_ssm(torch.flip(x, dims=[1]))
        y_backward = torch.flip(y_backward, dims=[1])

        y = (y_forward + y_backward) * F.silu(z)
        return self.out_proj(y)


class CrossModalMamba(nn.Module):
    """Cross-modal fusion module that can replace CGSA without compiled extensions."""

    def __init__(self, c1, c2, d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.c = c1

        self.ln_v = nn.LayerNorm(self.c)
        self.ln_i = nn.LayerNorm(self.c)

        self.mamba_interact = BiSSM(d_model=self.c, d_state=d_state, expand=expand)
        self.cv1 = nn.Conv2d(self.c * 2, self.c, 1, 1)

    def forward(self, x):
        x_v, x_i = x
        _, _, H, W = x_v.shape
        L = H * W

        feat_v = x_v.flatten(2).transpose(1, 2)
        feat_i = x_i.flatten(2).transpose(1, 2)
        feat_v = self.ln_v(feat_v)
        feat_i = self.ln_i(feat_i)

        joined_feat = torch.cat([feat_v, feat_i], dim=1)
        mamba_out = self.mamba_interact(joined_feat)
        out_v_seq, out_i_seq = torch.split(mamba_out, [L, L], dim=1)

        out_v = out_v_seq.transpose(1, 2).reshape_as(x_v)
        out_i = out_i_seq.transpose(1, 2).reshape_as(x_i)

        out = self.cv1(torch.cat([out_v + x_v, out_i + x_i], dim=1))
        return F.silu(out)


class AffineAlign(nn.Module):
    """Lightweight feature-level affine alignment for visible/infrared misregistration."""

    def __init__(
        self,
        channels,
        max_scale=0.10,
        max_shear=0.10,
        max_translate=0.10,
        delta_scale=1.0,
        residual_blend=1.0,
    ):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.max_scale = max_scale
        self.max_shear = max_shear
        self.max_translate = max_translate
        self.delta_scale = delta_scale
        self.residual_blend = residual_blend
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 6, 1),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).view(1, 2, 3))
        self.last_theta_mean = None
        self.last_delta_abs_mean = None

    def forward(self, reference, moving):
        delta = self.fc(self.pool(torch.cat([reference, moving], dim=1))).flatten(1).view(-1, 2, 3)
        d00 = delta[:, 0:1, 0:1].clamp(-self.max_scale, self.max_scale)
        d01 = delta[:, 0:1, 1:2].clamp(-self.max_shear, self.max_shear)
        d02 = delta[:, 0:1, 2:3].clamp(-self.max_translate, self.max_translate)
        d10 = delta[:, 1:2, 0:1].clamp(-self.max_shear, self.max_shear)
        d11 = delta[:, 1:2, 1:2].clamp(-self.max_scale, self.max_scale)
        d12 = delta[:, 1:2, 2:3].clamp(-self.max_translate, self.max_translate)
        delta = torch.cat([torch.cat([d00, d01, d02], dim=2), torch.cat([d10, d11, d12], dim=2)], dim=1)
        delta = delta * self.delta_scale
        theta = self.identity.to(delta.dtype) + delta
        if not torch.jit.is_scripting():
            self.last_theta_mean = theta.detach().mean(dim=0).cpu()
            self.last_delta_abs_mean = delta.detach().abs().mean().cpu()
        grid = F.affine_grid(theta, moving.size(), align_corners=False)
        aligned = F.grid_sample(moving, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if self.residual_blend < 1.0:
            aligned = moving + self.residual_blend * (aligned - moving)
        return aligned


class AlignedLocalMambaFusion(nn.Module):
    """Local-global cross-modal fusion with affine alignment and gated 2D detail preservation."""

    def __init__(self, c1, c2, d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.align_i = AffineAlign(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        self.gate = nn.Sequential(
            nn.Conv2d(c1 * 3, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        x_v, x_i = x
        x_i_aligned = self.align_i(x_v, x_i)
        pair = torch.cat([x_v, x_i_aligned], dim=1)

        local_feat = self.local(pair)
        global_feat = self.global_mamba([x_v, x_i_aligned])
        gate = self.gate(torch.cat([local_feat, global_feat, torch.abs(x_v - x_i_aligned)], dim=1))
        fused = local_feat * (1.0 - gate) + global_feat * gate
        return self.out(fused + 0.5 * (x_v + x_i_aligned))


class FrequencyDifferenceBranch(nn.Module):
    """Frequency-domain difference branch for complementary modal cues."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x_v, x_i):
        dtype = x_v.dtype
        H, W = x_v.shape[-2:]
        freq_v = torch.fft.rfft2(x_v.float(), norm="ortho")
        freq_i = torch.fft.rfft2(x_i.float(), norm="ortho")
        amp_diff = torch.abs(torch.abs(freq_v) - torch.abs(freq_i))
        freq_diff = torch.complex(amp_diff, torch.zeros_like(amp_diff))
        spatial_diff = torch.fft.irfft2(freq_diff, s=(H, W), norm="ortho").to(dtype)
        return self.proj(spatial_diff)


class ModalityReliabilityGate(nn.Module):
    """Predicts visible/infrared reliability maps from local and frequency differences."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1),
        )
        self.last_weights_mean = None

    def forward(self, x_v, x_i, freq_feat):
        logits = self.net(torch.cat([x_v, x_i, freq_feat], dim=1))
        weights = torch.softmax(logits, dim=1)
        if not torch.jit.is_scripting():
            self.last_weights_mean = weights.detach().mean(dim=(0, 2, 3)).cpu()
        return weights


class SpatialConsistencyGate(nn.Module):
    """Position-wise cross-modal consistency mask for suppressing unreliable fusion."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4 + 1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[-2].bias, -0.4)
        self.last_consistency_mean = None

    def forward(self, x_v, x_i, freq_feat):
        v_norm = F.normalize(x_v, dim=1)
        i_norm = F.normalize(x_i, dim=1)
        cosine = (v_norm * i_norm).sum(dim=1, keepdim=True)
        diff = torch.abs(x_v - x_i)
        mask = self.net(torch.cat([x_v, x_i, diff, cosine, freq_feat], dim=1))
        consistency = 0.25 + 0.75 * mask
        if not torch.jit.is_scripting():
            self.last_consistency_mean = consistency.detach().mean().cpu()
        return consistency


class ReliabilityAwareAlignedMambaFusion(nn.Module):
    """Reliability-aware, frequency-guided local-global fusion for multispectral detection."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.align_i = AffineAlign(c1)
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(c1 * 4, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 3, 1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None

    def _stage_prior(self, local_feat, global_feat, freq_feat):
        if self.stage in {"p3", "small", "detail"}:
            return local_feat + 0.35 * freq_feat + 0.15 * global_feat
        if self.stage in {"p5", "large", "semantic"}:
            return global_feat + 0.25 * local_feat + 0.20 * freq_feat
        return 0.45 * local_feat + 0.40 * global_feat + 0.15 * freq_feat

    def forward(self, x):
        x_v, x_i = x
        x_i_aligned = self.align_i(x_v, x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned

        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        gate_logits = self.fusion_gate(torch.cat([local_feat, global_feat, freq_feat, reliable_base], dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        if not torch.jit.is_scripting():
            self.last_fusion_gate_mean = gate.detach().mean(dim=(0, 2, 3)).cpu()
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
        fused = gate[:, 0:1] * local_feat + gate[:, 1:2] * global_feat + gate[:, 2:3] * freq_feat
        fused = fused + self._stage_prior(local_feat, global_feat, freq_feat) + reliable_base
        return self.out(fused)


class StabilizedRAAMambaFusion(nn.Module):
    """Stabilized RAA-Mamba with non-competitive residual fusion for better localization."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.align_i = AffineAlign(c1, max_scale=0.05, max_shear=0.05, max_translate=0.05)
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        self.branch_scale = nn.Parameter(torch.zeros(3))
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None

    def _stage_weights(self):
        if self.stage in {"p3", "small", "detail"}:
            return (1.00, 0.20, 0.30)
        if self.stage in {"p5", "large", "semantic"}:
            return (0.35, 1.00, 0.20)
        return (0.70, 0.55, 0.25)

    def forward(self, x):
        x_v, x_i = x
        x_i_aligned = self.align_i(x_v, x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned

        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        local_w, global_w, freq_w = self._stage_weights()
        learned = 0.5 + torch.sigmoid(self.branch_scale)
        fused = (
            local_w * learned[0] * local_feat
            + global_w * learned[1] * global_feat
            + freq_w * learned[2] * freq_feat
            + reliable_base
        )
        if not torch.jit.is_scripting():
            weights = torch.tensor(
                [local_w * float(learned[0]), global_w * float(learned[1]), freq_w * float(learned[2])]
            )
            self.last_fusion_gate_mean = weights / weights.sum().clamp_min(1e-6)
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
        return self.out(fused)


class GeometryPreservingRAAMambaFusion(nn.Module):
    """RAA-Mamba variant that keeps raw geometry as the anchor and uses weak residual alignment."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.schedule_progress = 1.0
        self.align_i = AffineAlign(
            c1,
            max_scale=0.04,
            max_shear=0.04,
            max_translate=0.04,
            delta_scale=0.25,
            residual_blend=0.35,
        )
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c1 * 3, max(c1 // 4, 16), 1, bias=False),
            nn.BatchNorm2d(max(c1 // 4, 16)),
            nn.SiLU(),
            nn.Conv2d(max(c1 // 4, 16), 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.detail_gate[-2].bias, -1.0)

        self.branch_gain = nn.Parameter(torch.tensor([-1.0, -1.4, -1.8]))
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None
        self.last_detail_gate_mean = None

    def _stage_weights(self):
        if self.stage in {"p3", "small", "detail"}:
            return (0.55, 0.08, 0.12)
        if self.stage in {"p5", "large", "semantic"}:
            return (0.20, 0.55, 0.08)
        return (0.40, 0.25, 0.10)

    def forward(self, x):
        x_v, x_i = x
        x_i_aligned = self.align_i(x_v, x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned
        raw_base = 0.5 * (x_v + x_i)

        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        detail_gate = self.detail_gate(torch.cat([local_feat, freq_feat, torch.abs(x_v - x_i_aligned)], dim=1))

        local_w, global_w, freq_w = self._stage_weights()
        learned = torch.sigmoid(self.branch_gain)
        detail_residual = (
            local_w * learned[0] * local_feat
            + global_w * learned[1] * global_feat
            + freq_w * learned[2] * freq_feat
        )
        fused = 0.60 * raw_base + 0.40 * reliable_base + detail_gate * detail_residual

        if not torch.jit.is_scripting():
            weights = torch.tensor(
                [local_w * float(learned[0]), global_w * float(learned[1]), freq_w * float(learned[2])]
            )
            self.last_fusion_gate_mean = weights / weights.sum().clamp_min(1e-6)
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
            self.last_detail_gate_mean = detail_gate.detach().mean().cpu()
        return self.out(fused)


class BoostedGeometryRAAMambaFusion(nn.Module):
    """Less conservative geometry-preserving fusion with stronger detail residuals."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.schedule_progress = 1.0
        self.align_i = AffineAlign(
            c1,
            max_scale=0.04,
            max_shear=0.04,
            max_translate=0.04,
            delta_scale=0.25,
            residual_blend=0.35,
        )
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        hidden = max(c1 // 4, 16)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c1 * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.detail_gate[-2].bias, -0.25)

        self.branch_gain = nn.Parameter(torch.tensor([-0.25, -0.70, -1.10]))
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None
        self.last_detail_gate_mean = None

    def _stage_weights(self):
        if self.stage in {"p3", "small", "detail"}:
            return (0.70, 0.12, 0.16)
        if self.stage in {"p5", "large", "semantic"}:
            return (0.25, 0.70, 0.10)
        return (0.50, 0.36, 0.14)

    def forward(self, x):
        x_v, x_i = x
        x_i_aligned = self.align_i(x_v, x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned
        raw_base = 0.5 * (x_v + x_i)

        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        detail_gate = self.detail_gate(torch.cat([local_feat, freq_feat, torch.abs(x_v - x_i_aligned)], dim=1))

        local_w, global_w, freq_w = self._stage_weights()
        learned = torch.sigmoid(self.branch_gain)
        detail_residual = (
            local_w * learned[0] * local_feat
            + global_w * learned[1] * global_feat
            + freq_w * learned[2] * freq_feat
        )
        fused = 0.35 * raw_base + 0.45 * reliable_base + 1.25 * detail_gate * detail_residual

        if not torch.jit.is_scripting():
            weights = torch.tensor(
                [local_w * float(learned[0]), global_w * float(learned[1]), freq_w * float(learned[2])]
            )
            self.last_fusion_gate_mean = weights / weights.sum().clamp_min(1e-6)
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
            self.last_detail_gate_mean = detail_gate.detach().mean().cpu()
        return self.out(fused)


class ConsistencyGuidedRAAMambaFusion(nn.Module):
    """Geometry-preserving RAA-Mamba with spatial consistency guided fusion suppression."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.schedule_progress = 1.0
        self.align_i = AffineAlign(
            c1,
            max_scale=0.035,
            max_shear=0.035,
            max_translate=0.035,
            delta_scale=0.20,
            residual_blend=0.30,
        )
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)
        self.consistency = SpatialConsistencyGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        hidden = max(c1 // 4, 16)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c1 * 4, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.detail_gate[-2].bias, -0.45)

        self.branch_gain = nn.Parameter(torch.tensor([-0.35, -0.80, -1.05]))
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None
        self.last_detail_gate_mean = None
        self.last_consistency_mean = None
        self.last_schedule_progress = None

    def set_schedule_progress(self, progress):
        self.schedule_progress = float(max(0.0, min(1.0, progress)))

    def _stage_weights(self):
        if self.stage in {"p3", "small", "detail"}:
            return (0.74, 0.10, 0.16)
        if self.stage in {"p5", "large", "semantic"}:
            return (0.24, 0.72, 0.09)
        return (0.52, 0.36, 0.13)

    def forward(self, x):
        x_v, x_i = x
        progress = float(self.schedule_progress)
        residual_strength = 0.35 + 0.80 * progress
        local_fallback_strength = 0.12 + 0.08 * progress
        x_i_aligned = self.align_i(x_v, x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned
        raw_base = 0.5 * (x_v + x_i)
        safe_base = 0.45 * raw_base + 0.55 * reliable_base

        consistency = self.consistency(x_v, x_i_aligned, freq_feat)
        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        detail_gate = self.detail_gate(
            torch.cat([local_feat, freq_feat, torch.abs(x_v - x_i_aligned), consistency.expand_as(local_feat)], dim=1)
        )

        local_w, global_w, freq_w = self._stage_weights()
        learned = torch.sigmoid(self.branch_gain)
        cross_detail = (
            local_w * learned[0] * local_feat
            + global_w * learned[1] * global_feat
            + freq_w * learned[2] * freq_feat
        )
        fused = safe_base + consistency * residual_strength * detail_gate * cross_detail
        fused = fused + (1.0 - consistency) * local_fallback_strength * local_feat

        if not torch.jit.is_scripting():
            weights = torch.tensor(
                [local_w * float(learned[0]), global_w * float(learned[1]), freq_w * float(learned[2])]
            )
            self.last_fusion_gate_mean = weights / weights.sum().clamp_min(1e-6)
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
            self.last_detail_gate_mean = detail_gate.detach().mean().cpu()
            self.last_consistency_mean = consistency.detach().mean().cpu()
            self.last_schedule_progress = progress
        return self.out(fused)


class ResidualConsistencyMambaFusion(nn.Module):
    """Small-data friendly fusion that learns residual cross-modal corrections from a safe raw anchor."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.schedule_progress = 1.0
        self.align_i = AffineAlign(
            c1,
            max_scale=0.025,
            max_shear=0.025,
            max_translate=0.025,
            delta_scale=0.15,
            residual_blend=0.25,
        )
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)
        self.consistency = SpatialConsistencyGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        hidden = max(c1 // 4, 16)
        self.residual_gate = nn.Sequential(
            nn.Conv2d(c1 * 4 + 1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.residual_gate[-2].bias, -2.0)
        self.residual_proj = nn.Conv2d(c1, c2, 1, bias=False)
        nn.init.zeros_(self.residual_proj.weight)
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None
        self.last_detail_gate_mean = None
        self.last_consistency_mean = None
        self.last_schedule_progress = None

    def set_schedule_progress(self, progress):
        self.schedule_progress = float(max(0.0, min(1.0, progress)))

    def _stage_weights(self):
        if self.stage in {"p3", "small", "detail"}:
            return (0.80, 0.08, 0.12)
        if self.stage in {"p5", "large", "semantic"}:
            return (0.20, 0.72, 0.08)
        return (0.55, 0.32, 0.13)

    def forward(self, x):
        x_v, x_i = x
        progress = float(self.schedule_progress)
        x_i_aligned = self.align_i(x_v, x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned
        raw_base = 0.5 * (x_v + x_i)
        anchor = 0.65 * raw_base + 0.35 * reliable_base

        consistency = self.consistency(x_v, x_i_aligned, freq_feat)
        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        gate_input = torch.cat(
            [local_feat, global_feat, freq_feat, torch.abs(x_v - x_i_aligned), consistency],
            dim=1,
        )
        gate = self.residual_gate(gate_input)

        local_w, global_w, freq_w = self._stage_weights()
        residual = (
            local_w * gate[:, 0:1] * local_feat
            + global_w * gate[:, 1:2] * global_feat
            + freq_w * gate[:, 2:3] * freq_feat
        )
        residual_strength = 0.15 + 0.70 * progress
        fused = anchor + self.residual_proj(consistency * residual_strength * residual)

        if not torch.jit.is_scripting():
            learned = gate.detach().mean(dim=(0, 2, 3)).cpu()
            weights = torch.tensor(
                [local_w * float(learned[0]), global_w * float(learned[1]), freq_w * float(learned[2])]
            )
            self.last_fusion_gate_mean = weights / weights.sum().clamp_min(1e-6)
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
            self.last_detail_gate_mean = learned.mean()
            self.last_consistency_mean = consistency.detach().mean().cpu()
            self.last_schedule_progress = progress
        return self.out(fused)


class ScheduledBoostedGeometryRAAMambaFusion(nn.Module):
    """Boosted geometry fusion with epoch-aware residual warmup to reduce validation spikes."""

    def __init__(self, c1, c2, stage="p4", d_state=16, expand=1):
        super().__init__()
        assert c1 == c2, "Input channels for both modalities must be equal."
        self.stage = str(stage).lower()
        self.schedule_progress = 1.0
        self.align_i = AffineAlign(
            c1,
            max_scale=0.035,
            max_shear=0.035,
            max_translate=0.035,
            delta_scale=0.20,
            residual_blend=1.0,
        )
        self.freq = FrequencyDifferenceBranch(c1)
        self.reliability = ModalityReliabilityGate(c1)

        self.local = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 * 2, 3, padding=1, groups=c1 * 2, bias=False),
            nn.BatchNorm2d(c1 * 2),
            nn.SiLU(),
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.global_mamba = CrossModalMamba(c1, c2, d_state=d_state, expand=expand)
        hidden = max(c1 // 4, 16)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c1 * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.detail_gate[-2].bias, -0.55)

        self.branch_gain = nn.Parameter(torch.tensor([-0.45, -0.85, -1.20]))
        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.last_fusion_gate_mean = None
        self.last_reliability_mean = None
        self.last_theta_mean = None
        self.last_theta_delta_abs_mean = None
        self.last_detail_gate_mean = None
        self.last_schedule_progress = None

    def set_schedule_progress(self, progress):
        self.schedule_progress = float(max(0.0, min(1.0, progress)))

    def _stage_weights(self):
        if self.stage in {"p3", "small", "detail"}:
            return (0.72, 0.10, 0.14)
        if self.stage in {"p5", "large", "semantic"}:
            return (0.23, 0.72, 0.08)
        return (0.52, 0.34, 0.12)

    def forward(self, x):
        x_v, x_i = x
        progress = float(self.schedule_progress)
        align_strength = 0.20 + 0.18 * progress
        residual_strength = 0.70 + 0.55 * progress
        raw_weight = 0.50 - 0.15 * progress
        reliable_weight = 0.45

        x_i_warped = self.align_i(x_v, x_i)
        x_i_aligned = x_i + align_strength * (x_i_warped - x_i)
        freq_feat = self.freq(x_v, x_i_aligned)
        reliability = self.reliability(x_v, x_i_aligned, freq_feat)
        reliable_base = reliability[:, 0:1] * x_v + reliability[:, 1:2] * x_i_aligned
        raw_base = 0.5 * (x_v + x_i)

        local_feat = self.local(torch.cat([x_v, x_i_aligned], dim=1))
        global_feat = self.global_mamba([x_v, x_i_aligned])
        detail_gate = self.detail_gate(torch.cat([local_feat, freq_feat, torch.abs(x_v - x_i_aligned)], dim=1))

        local_w, global_w, freq_w = self._stage_weights()
        learned = torch.sigmoid(self.branch_gain)
        detail_residual = (
            local_w * learned[0] * local_feat
            + global_w * learned[1] * global_feat
            + freq_w * learned[2] * freq_feat
        )
        fused = raw_weight * raw_base + reliable_weight * reliable_base + residual_strength * detail_gate * detail_residual

        if not torch.jit.is_scripting():
            weights = torch.tensor(
                [local_w * float(learned[0]), global_w * float(learned[1]), freq_w * float(learned[2])]
            )
            self.last_fusion_gate_mean = weights / weights.sum().clamp_min(1e-6)
            self.last_reliability_mean = self.reliability.last_weights_mean
            self.last_theta_mean = self.align_i.last_theta_mean
            self.last_theta_delta_abs_mean = self.align_i.last_delta_abs_mean
            self.last_detail_gate_mean = detail_gate.detach().mean().cpu()
            self.last_schedule_progress = progress
        return self.out(fused)


CGSA_Mamba = CrossModalMamba
CGSA = CrossModalMamba
