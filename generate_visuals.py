from torchvision import transforms
from typing import List
import torch
import PIL.Image
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

from repaint import sample_to_pil, repaint
from wrappers import Model, CustomScheduler

def plot_10_images(list_of_images: List[PIL.Image.Image]):
    plt.tight_layout()
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.axis('off')
        ax.imshow(list_of_images[i])
    plt.show()

def generate_n_samples(image: torch.Tensor, mask: torch.Tensor,
                       model: Model, scheduler: CustomScheduler, n:int=10):

    images = []
    for _ in range(n):
        repainted_image = repaint(image, mask, model, scheduler)
        images.append(sample_to_pil(repainted_image))

    return images

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

    images = generate_n_samples(data_transform(image).to(device),
                                mask_transform(mask).to(device),
                                model, scheduler, n=10)
    plot_10_images(images)
