"""
Inference script for InternVL3.5 family models.
Adapted for inverse scaling experiments.
"""
import os
import json
import argparse
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from typing import Dict, Any, List
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from infer_utils import (
    process_prompt,
    get_dataset_config,
    get_prompt_suffix,
    save_results,
    load_existing_results,
    setup_output_path,
    add_common_args,
    load_dataset_from_args,
)

MAX_RETRY = 5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INTERNVL_MODELS = [
    "OpenGVLab/InternVL3_5-1B-Instruct",
    "OpenGVLab/InternVL3_5-2B-Instruct",
    "OpenGVLab/InternVL3_5-4B-Instruct",
    "OpenGVLab/InternVL3_5-8B-Instruct",
    "OpenGVLab/InternVL3_5-14B-Instruct",
    "OpenGVLab/InternVL3_5-30B-A3B-Instruct",  # MoE (3B active)
    "OpenGVLab/InternVL3_5-38B-Instruct",
    "OpenGVLab/InternVL3_5-241B-A28B-Instruct", # MoE (28B active)
]

MULTI_GPU_MODELS = [
    "OpenGVLab/InternVL3_5-241B-A28B-Instruct",
    # 38B weights (76GB bf16) + vision-tile activations exceed one 80GB card
    # once MMBench images load correctly; shard across the allocated GPUs.
    "OpenGVLab/InternVL3_5-38B-Instruct",
]


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image, input_size=448, max_num=12):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def run_inference(dataset, model, tokenizer, prompt_suffix, args, dataset_config):
    """Run inference on dataset using InternVL3.5 model."""
    output_path = setup_output_path(args)
    results = load_existing_results(output_path)
    start_idx = len(results)
    if start_idx >= len(dataset):
        print(f"All {len(dataset)} samples already processed, skipping.")
        return results
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{len(dataset)}")

    generation_config = dict(max_new_tokens=2048, do_sample=False)

    for idx, data in enumerate(tqdm(dataset, desc="Processing", initial=start_idx, total=len(dataset))):
        if idx < start_idx:
            continue
        prompt, images = process_prompt(data, dataset_config, prompt_suffix)

        # Process images
        pixel_values_list = []
        for img in images:
            pv = load_image(img, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pv)

        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

        # Construct question with image tokens
        image_tokens = "<image>\n" * len(images)
        question = f"{image_tokens}{prompt}"

        retry_count = 0
        generated_text = ""

        while not generated_text and retry_count < MAX_RETRY:
            try:
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                generated_text = response
                print(f"[{idx}] {generated_text[:100]}...")

                if not generated_text:
                    retry_count += 1
                    print(f"Retry {retry_count}/{MAX_RETRY} due to empty output.")
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{MAX_RETRY} due to error: {e}")

        if not generated_text:
            print(f"Failed after {MAX_RETRY} retries.")

        result_sample = {"response": generated_text, **data}
        result_sample = {k: v for k, v in result_sample.items() if not k.startswith("image") and not k.startswith("decoded_image")}
        results.append(result_sample)

        if args.save_every and (idx + 1) % args.save_every == 0:
            output_path = setup_output_path(args)
            save_results(results, output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="InternVL3.5 Inference for Inverse Scaling")
    add_common_args(parser)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    args = parser.parse_args()

    # Dataset
    dataset_config = get_dataset_config(args.dataset)
    dataset = load_dataset_from_args(args)

    # Prompt
    prompt_suffix = args.prompt_suffix or get_prompt_suffix(args.mode)

    # Model
    print(f"Loading model {args.model}...")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
    )
    if args.model in MULTI_GPU_MODELS:
        model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(args.model, **model_kwargs).eval()
    else:
        model = AutoModel.from_pretrained(args.model, **model_kwargs).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False
    )
    print("Model loaded.")

    # Run
    results = run_inference(dataset, model, tokenizer, prompt_suffix, args, dataset_config)

    # Save
    output_path = setup_output_path(args)
    save_results(results, output_path)


if __name__ == "__main__":
    main()
