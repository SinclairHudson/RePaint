"""
implements the main repaint idea
"""
from torchvision import transforms
import torch
import PIL.Image
from diffusers import UNet2DModel, DDPMScheduler
from dataclasses import dataclass
from tqdm import tqdm

## scheduler must provide the following fields:
# timesteps 999 to 0
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


@torch.no_grad()
def single_reverse_step(model: Model, x: torch.Tensor, t: int, S: CustomScheduler) -> torch.Tensor:
    """
    applies the model to go from timestep t to t-1
    :param model: the model that predicts the noise
    :param x: the data at timestep t
    :param t: the current timestep
    :param scheduler: class that provides the variance schedule
    :return: the data at diffusion timestep t-1
    """
    mean = S.sqrt_recip_alphas[t] * (x - S.betas[t] * model(x, t) / S.sqrt_one_minus_alphas_cumprod[t])
    if t == 0:
        return mean
    else:
        noise = torch.randn_like(x) * torch.sqrt(S.posterior_variance[t])
        return mean + noise

@torch.no_grad()
def zero_to_t(x_0: torch.Tensor, t: int, S: CustomScheduler):
    if t == 0:
        return x_0
    else:
        return torch.sqrt(S.alphas_cumprod[t]) * x_0 + \
                torch.sqrt(1.0 - S.alphas_cumprod[t]) * torch.randn_like(x_0)



@torch.no_grad()
def repaint(original_data: torch.Tensor, keep_mask: torch.Tensor,
            model: Model, scheduler: CustomScheduler) -> torch.Tensor:
    """
    repaints that which isn't in the mask using the provided diffusion model
    :param original_image: the original data to repaint. Must be in the range that the model expects (usually [-1, 1])
    :param keep_mask: the mask of the image to keep. All values must be 0 or 1
    :param model: the diffusion model to use
    :param scheduler: the scheduler to use, must be compatible with the model
    """

    device = original_data.device
    sample = torch.randn_like(original_data).to(device) # sample is x_t+1
    for t in tqdm(scheduler.timesteps):
        x_known = zero_to_t(original_data, t, scheduler)
        x_unknown = single_reverse_step(model, sample, t, scheduler)
        sample = keep_mask * x_known + (1-keep_mask) * x_unknown

    return sample

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "google/ddpm-celebahq-256"
    model = UNet2DModel.from_pretrained("goodle/ddpm-celebahq-256")
    scheduler = DDPMScheduler.from_config(model_id)


    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = PIL.Image.open("celeba_01.jpg")
    mask = PIL.Image.open("half_mask.png")

    model = 

    result = repaint()
