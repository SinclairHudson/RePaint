"""
implements the main repaint idea
"""
from torchvision import transforms
import torch
import PIL.Image
from diffusers import UNet2DModel, DDPMScheduler

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

## scheduler must provide the following fields:
# timesteps 999 to 0
# betas


@torch.no_grad()
def repaint(original_data: torch.Tensor, keep_mask: torch.Tensor, model, scheduler) -> torch.Tensor:
    """
    repaints that which isn't in the mask using the provided diffusion model
    :param original_image: the original data to repaint. Must be in the range that the model expects (usually [-1, 1])
    :param keep_mask: the mask of the image to keep. All values must be 0 or 1
    :param model: the diffusion model to use
    :param scheduler: the scheduler to use, must be compatible with the model
    """

    for i, t in enumerate(scheduler.timesteps):

    pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "google/ddpm-celebahq-256"
    model = UNet2DModel.from_pretrained("goodle/ddpm-celebahq-256")
    scheduler = DDPMScheduler.from_config(model_id)

    image = PIL.Image.open("celeba_01.jpg")
    mask = PIL.Image.open("half_mask.png")

    repaint()
