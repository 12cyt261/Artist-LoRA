import argparse

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from utils import BLOCKS, filter_lora, scale_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="Artist-LoRA prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--content_LoRA", type=str, default=None, help="path for the content Artist-LoRA"
    )
    parser.add_argument(
        "--style_LoRA", type=str, default=None, help="path for the style Artist-LoRA"
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1., help="alpha parameter to scale the content Artist-LoRA weights"
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1., help="alpha parameter to scale the style Artist-LoRA weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")
    

    # Get Content B-LoRA SD
    if args.content_LoRA is not None:
        content_LoRA_sd, _ = pipeline.lora_state_dict(args.content_LoRA)
        content_LoRA = filter_lora(content_LoRA_sd, BLOCKS['content'])
        content_LoRA = scale_lora(content_LoRA, args.content_alpha)
    else:
        content_LoRA = {}

    # Get Style B-LoRA SD
    if args.style_LoRA is not None:
        style_LoRA_sd, _ = pipeline.lora_state_dict(args.style_LoRA)
        style_LoRA = filter_lora(style_LoRA_sd, BLOCKS['style'])
        style_LoRA = scale_lora(style_LoRA, args.style_alpha)
    else:
        style_LoRA = {}

    # Merge B-LoRAs SD
    res_lora = {**content_LoRA, **style_LoRA}

    # Load
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

    seed = 4
    for i in range(0, seed):
        generator = torch.Generator(device="cuda").manual_seed(i)
        img = pipeline(args.prompt, generator=generator, num_images_per_prompt=args.num_images_per_prompt).images[0]
        img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
