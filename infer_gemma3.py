"""
Inference script for Gemma 3 family models.
Adapted for inverse scaling experiments.
"""
import os
import json
import argparse
import torch
from typing import Dict, Any, List
from PIL import Image
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

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

GEMMA3_MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]


def prepare_messages(prompt: str, images: List[Image.Image]) -> List[Dict]:
    """Prepare messages in Gemma 3 format."""
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_inference(dataset, model, processor, prompt_suffix, args, dataset_config):
    """Run inference on dataset using Gemma 3 model."""
    output_path = setup_output_path(args)
    results = load_existing_results(output_path)
    start_idx = len(results)
    if start_idx >= len(dataset):
        print(f"All {len(dataset)} samples already processed, skipping.")
        return results
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{len(dataset)}")

    for idx, data in enumerate(tqdm(dataset, desc="Processing", initial=start_idx, total=len(dataset))):
        if idx < start_idx:
            continue
        prompt, images = process_prompt(data, dataset_config, prompt_suffix)

        pil_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            pil_images.append(img.convert("RGB"))

        messages = prepare_messages(prompt, pil_images)

        retry_count = 0
        generated_text = ""

        while not generated_text and retry_count < MAX_RETRY:
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=2048, do_sample=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                generated_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

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
    parser = argparse.ArgumentParser(description="Gemma 3 Inference for Inverse Scaling")
    add_common_args(parser)
    args = parser.parse_args()

    # Dataset
    dataset_config = get_dataset_config(args.dataset)
    dataset = load_dataset_from_args(args)

    # Prompt
    prompt_suffix = args.prompt_suffix or get_prompt_suffix(args.mode)

    # Model
    print(f"Loading model {args.model}...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print("Model loaded.")

    # Run
    results = run_inference(dataset, model, processor, prompt_suffix, args, dataset_config)

    # Save
    output_path = setup_output_path(args)
    save_results(results, output_path)


if __name__ == "__main__":
    main()
