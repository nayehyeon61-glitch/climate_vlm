"""Stage 1: same-time downscaling with CorrDiff from physicsnemo (Mardani et al. 2025, plan ref. [1]).

    x~_s^(m) = D_mu(x_s^LR) + D_res(x_s^LR, eps_m),   s <= t

D_mu   physicsnemo ``CorrDiffRegressionUNet`` (SongUNetPosEmbd), trained with physicsnemo ``RegressionLoss``
D_res  optional: physicsnemo ``EDMPrecondSuperResolution`` trained on the residual with ``ResidualLoss``
       (hr_mean_conditioning, D_mu frozen) and sampled with ``stochastic_sampler`` (EDM, Algorithm 2)

As in CorrDiff, the conditioning image is the LR state interpolated onto the HR grid (here with the HR
orography and latitude features concatenated). The only change to the physicsnemo models: the global
grid is periodic in longitude, so every UNet call sees ``lon_pad`` extra columns copied from the other
side and the output is cropped back (``LonPeriodic``); CorrDiff's regional domains need no such padding.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from physicsnemo.diffusion.metrics import RegressionLoss, ResidualLoss
from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution
from physicsnemo.diffusion.samplers import stochastic_sampler
from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet


def bpad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Reflect in latitude, circular in longitude."""
    return F.pad(F.pad(x, (0, 0, p, p), mode="reflect"), (p, p, 0, 0), mode="circular")


def interp(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Bilinear (..., C, h, w) -> (..., C, h*f, w*f) for fine cells that split coarse cells; periodic in lon."""
    if factor == 1:
        return x
    shp = x.shape
    y = F.interpolate(bpad(x.reshape(-1, *shp[-3:]), 1), scale_factor=factor, mode="bilinear", align_corners=False)
    y = y[..., factor:-factor, factor:-factor]
    return y.reshape(*shp[:-2], *y.shape[-2:])


class Conv(nn.Module):
    """Conv2d with reflect (lat) / circular (lon) padding."""

    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv, self.p = nn.Conv2d(c_in, c_out, k), k // 2

    def forward(self, x):
        return self.conv(bpad(x, self.p) if self.p else x)


class LonPeriodic(nn.Module):
    """Circular longitude padding around a physicsnemo UNet / preconditioner: f(x, img_lr, ...) on a global grid."""

    def __init__(self, net: nn.Module, pad: int):
        super().__init__()
        self.net, self.pad = net, pad

    def _p(self, x):
        return F.pad(x, (self.pad, self.pad, 0, 0), mode="circular") if self.pad and x is not None else x

    def forward(self, x, img_lr=None, *args, **kwargs):
        y = self.net(self._p(x), self._p(img_lr), *args, **kwargs)
        return y[..., self.pad:-self.pad] if self.pad else y

    # attributes the physicsnemo samplers read from the preconditioner
    @property
    def sigma_min(self):
        return self.net.sigma_min

    @property
    def sigma_max(self):
        return self.net.sigma_max

    def round_sigma(self, sigma):
        return self.net.round_sigma(sigma)


class Downscaler(nn.Module):
    """D = D_mu (+ optional D_res) on normalised fields. ``tokens`` builds the Mean / Tendency / Spread inputs."""

    def __init__(self, channels: int, factor: int, hr_shape: Sequence[int], c_static: int = 3, lon_pad: int = 16,
                 unet: Optional[Dict] = None, residual: bool = False, res_unet: Optional[Dict] = None,
                 sampler_steps: int = 18):
        super().__init__()
        H, W = hr_shape
        self.factor, self.channels, self.steps = factor, channels, sampler_steps
        res_px = [H, W + 2 * lon_pad]
        unet = dict(unet or {})
        n_grid = unet.setdefault("N_grid_channels", 4)
        unet.setdefault("embedding_type", "zero")
        cond = channels + c_static
        self.mean = LonPeriodic(CorrDiffRegressionUNet(
            img_resolution=res_px, img_in_channels=cond + n_grid, img_out_channels=channels,
            model_type="SongUNetPosEmbd", **unet), lon_pad)
        self.res = None
        if residual:
            ru = dict(res_unet or {})
            n_grid_r = ru.setdefault("N_grid_channels", 4)
            ru.setdefault("gridtype", "sinusoidal")
            ru.setdefault("embedding_type", "zero")
            self.res = LonPeriodic(EDMPrecondSuperResolution(
                img_resolution=res_px, img_in_channels=cond + channels + n_grid_r, img_out_channels=channels,
                model_type="SongUNetPosEmbd", **ru), lon_pad)

    def condition(self, x_lr: torch.Tensor, s_hr: torch.Tensor) -> torch.Tensor:
        """CorrDiff conditioning image: LR state interpolated to the HR grid + HR static fields."""
        return torch.cat([interp(x_lr, self.factor), s_hr.expand(x_lr.shape[0], -1, -1, -1)], 1)

    def forward(self, x_lr: torch.Tensor, s_hr: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) normalised LR -> (B, C, H, W) conditional mean."""
        img_lr = self.condition(x_lr, s_hr)
        B, _, H, W = img_lr.shape
        return self.mean(torch.zeros(B, self.channels, H, W, device=img_lr.device, dtype=img_lr.dtype), img_lr)

    # ---------------------------------------------------------------------------------- stage 1 losses
    def regression_loss(self, x_lr, x_hr, s_hr) -> torch.Tensor:
        """physicsnemo RegressionLoss, elementwise (B, C, H, W)."""
        return RegressionLoss()(self.mean, x_hr, self.condition(x_lr, s_hr))

    def residual_loss(self, x_lr, x_hr, s_hr) -> torch.Tensor:
        """physicsnemo ResidualLoss with the HR mean as extra conditioning, elementwise (B, C, H, W)."""
        if self.res is None:
            raise RuntimeError("downscaler built without the residual model")
        loss = ResidualLoss(regression_net=self.mean, hr_mean_conditioning=True)
        return loss(self.res, x_hr, self.condition(x_lr, s_hr))

    @torch.no_grad()
    def sample(self, x_lr: torch.Tensor, s_hr: torch.Tensor, members: int,
               mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, C, h, w) -> (B, M, C, H, W) samples D_mu + D_res via physicsnemo's EDM stochastic sampler."""
        img_lr = self.condition(x_lr, s_hr)
        mean = self(x_lr, s_hr) if mean is None else mean
        B = img_lr.shape[0]
        lr = img_lr.repeat_interleave(members, 0)
        mu = mean.repeat_interleave(members, 0)
        latents = torch.randn_like(mu)
        res = stochastic_sampler(self.res, latents, lr, mean_hr=mu, num_steps=self.steps)
        return (mu + res).view(B, members, *mean.shape[1:])

    # ---------------------------------------------------------------------------------- vision-path inputs
    def tokens(self, x_lr: torch.Tensor, s_hr: torch.Tensor, history_h: Sequence[int],
               members: int = 0) -> Dict[str, torch.Tensor]:
        """x_lr (B, S, C, h, w) history -> dict of (B, S, C, H, W): lr (interpolated input), mean, tendency,
        and spread when D_res is present and members > 1. Tendency = mean difference / real dt in days, from the
        deterministic mean (plan p.4: differencing independent samples would create spurious motion); its first
        slot is zero."""
        B, S = x_lr.shape[:2]
        mean = self(x_lr.flatten(0, 1), s_hr)
        mean = mean.view(B, S, *mean.shape[1:])
        dt = torch.tensor([b - a for a, b in zip(history_h[:-1], history_h[1:])], dtype=mean.dtype,
                          device=mean.device).view(1, -1, 1, 1, 1) / 24.0
        out = {"lr": interp(x_lr, self.factor), "mean": mean,
               "tendency": torch.cat([torch.zeros_like(mean[:, :1]), (mean[:, 1:] - mean[:, :-1]) / dt], 1)}
        if self.res is not None and members > 1:
            ens = self.sample(x_lr.flatten(0, 1), s_hr, members, mean=mean.flatten(0, 1))
            out["spread"] = ens.std(1).view_as(mean)
        return out
