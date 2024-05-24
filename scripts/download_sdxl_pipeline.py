from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

text2img = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
text2img.save_pretrained("sdxl_text2img")

#img2img = StableDiffusionXLImg2ImgPipeline(**text2img.components)
#img2img.save_pretrained("sdxl_img2img")
