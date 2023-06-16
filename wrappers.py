from dataclasses import dataclass
import torch
from diffusers import UNet2DModel, DDPMScheduler

## scheduler must provide the following fields:
# timesteps T to 0
# betas

@dataclass
class CustomScheduler:
    def __init__(self, timesteps: torch.Tensor, betas: torch.Tensor):
        assert len(timesteps) == len(betas)
        self.timesteps = timesteps
        self.betas = betas
        # TODO verify this
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.roll(self.alphas_cumprod, 1)
        self.alphas_cumprod_prev[0] = 1.0

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    @classmethod
    def from_DDPMScheduler(cls, ddpm_scheduler: DDPMScheduler):
        return cls(ddpm_scheduler.timesteps, ddpm_scheduler.betas)

class Model:
    def __init__(self, model: UNet2DModel):
        self.model = model

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def __call__(self, x, t):
        """
        :param x: input image
        :param t: timestep
        :return: predicted noise at the current timestep, to be subtracted
        """
        return self.model(x, t)["sample"]
