"""
implements the main repaint algorithm
"""
from torchvision import transforms
import torch
import PIL.Image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from wrappers import Model, CustomScheduler


sample_to_pil = transforms.Compose([
        transforms.Lambda(lambda t: t.squeeze(0)), # CHW to HWC
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: (t + 1) * 127.5), # [-1, 1] to [0, 255]
        transforms.Lambda(lambda t: torch.clamp(t, 0, 255)),
        transforms.Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])


@torch.no_grad()
def single_reverse_step(model: Model, x: torch.Tensor, t: int, S: CustomScheduler) -> torch.Tensor:
    """
    applies the model to go from timestep t to t-1
    See Algorithm 2 of https://arxiv.org/pdf/2006.11239.pdf
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
def zero_to_t(x_0: torch.Tensor, t: int, S: CustomScheduler) -> torch.Tensor:
    if t == 0:
        return x_0
    else:
        return torch.sqrt(S.alphas_cumprod[t]) * x_0 + \
                torch.sqrt(1.0 - S.alphas_cumprod[t]) * torch.randn_like(x_0)

@torch.no_grad()
def forward_j_steps(x_t: torch.Tensor, t: int, j: int, S: CustomScheduler)-> torch.Tensor:
    partial_alpha_cumprod = S.alphas_cumprod[t+j]/S.alphas_cumprod[t]
    return torch.sqrt(partial_alpha_cumprod) * x_t + \
            torch.sqrt(1.0 - partial_alpha_cumprod) * torch.randn_like(x_t)

def get_jumps(timesteps, jumps_every:int=100, r:int=5) -> List[int]:
    jumps = []
    for i in range(0, torch.max(timesteps), jumps_every):
        jumps.extend([i] * r)
    jumps.reverse()  # must be in descending order
    return jumps

@torch.no_grad()
def repaint(original_data: torch.Tensor, keep_mask: torch.Tensor,
            model: Model, scheduler: CustomScheduler, j:int=10, r:int=5) -> torch.Tensor:
    """
    repaints that which isn't in the mask using the provided diffusion model
    :param original_image: the original data to repaint. Must be in the range that the model expects (usually [-1, 1])
    :param keep_mask: the mask of the image to keep. All values must be 0 or 1
    :param model: the diffusion model to use
    :param scheduler: the scheduler to use, must be compatible with the model
    """

    jumps = get_jumps(scheduler.timesteps, r=r)

    device = original_data.device
    sample = torch.randn_like(original_data).to(device) # sample is x_t+1
    print("beginning repaint")
    for t in tqdm(scheduler.timesteps):
        # the following loop handles the bouts of resampling
        while len(jumps) > 0 and jumps[0] == t:
            jumps = jumps[1:]
            sample = forward_j_steps(sample, t, j, scheduler)
            # after the resample, come back down to the current timestep
            for override_t in range(t + j, t, -1):
                sample = single_reverse_step(model, sample, override_t, scheduler)

        x_known = zero_to_t(original_data, t, scheduler)
        x_unknown = single_reverse_step(model, sample, t, scheduler)
        sample = keep_mask * x_known + (1-keep_mask) * x_unknown

    return sample

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "google/ddpm-celebahq-256"
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256")
    scheduler = DDPMScheduler.from_pretrained(model_id)


    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
        transforms.Lambda(lambda t: t.unsqueeze(0))
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.unsqueeze(0))
    ])

    image = PIL.Image.open("img/celeba_01.jpg")
    mask = PIL.Image.open("img/half_mask.png")

    model = Model(model).to(device)
    scheduler = CustomScheduler.from_DDPMScheduler(scheduler)

    result = repaint(data_transform(image).to(device),
                     mask_transform(mask).to(device),
                     model, scheduler)
    plt.imshow(sample_to_pil(result))
    plt.show()
