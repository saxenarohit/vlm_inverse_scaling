"""
Shared utilities for VLM inference scripts.
Adapted from robustbench for inverse scaling experiments.

Supports: MMT-Bench, MMMU, ScienceQA via flexible field mapping.
"""
import os
import re
import ast
import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class DatasetConfig:
    """Configuration for dataset field mapping."""
    question_field: str = "question"
    image_field: str = "image"
    answer_field: str = "answer"
    options_field: Optional[str] = "options"
    category_field: Optional[str] = "subject"
    multi_image: bool = False
    image_in_question: bool = False
    answer_is_list: bool = False
    task_type: str = "multiple_choice"


DATASET_CONFIGS = {
    "mmt_bench": DatasetConfig(
        question_field="question",
        image_field="image",
        answer_field="answer",
        options_field="options",
        category_field="l2-category",
        multi_image=False,
        image_in_question=False,
        task_type="multiple_choice",
    ),
    "mmmu": DatasetConfig(
        question_field="question",
        image_field="image",
        answer_field="answer",
        options_field="options",
        category_field="subfield",
        multi_image=True,
        image_in_question=True,
        task_type="multiple_choice",
    ),
    "scienceqa": DatasetConfig(
        question_field="question",
        image_field="image",
        answer_field="answer",
        options_field="choices",
        category_field="category",
        multi_image=False,
        image_in_question=False,
        task_type="multiple_choice",
    ),
    "mme": DatasetConfig(
        question_field="question",
        image_field="image",
        answer_field="answer",
        options_field=None,
        category_field="category",
        multi_image=False,
        image_in_question=False,
        task_type="yes_no",
    ),
    "ai2d": DatasetConfig(
        question_field="question",
        image_field="image",
        answer_field="answer",
        options_field=None,
        category_field="category",
        multi_image=False,
        image_in_question=False,
        task_type="preformatted",
    ),
    "mathvista": DatasetConfig(
        question_field="query",
        image_field="decoded_image",
        answer_field="answer",
        options_field=None,
        category_field="task",
        multi_image=False,
        image_in_question=False,
        task_type="preformatted",
    ),
    "blink": DatasetConfig(
        question_field="prompt",
        image_field="image",
        answer_field="answer",
        options_field=None,
        category_field="sub_task",
        multi_image=True,
        image_in_question=False,
        task_type="preformatted",
    ),
    "mmbench": DatasetConfig(
        question_field="question",
        image_field="image",
        answer_field="answer",
        options_field="options",
        category_field="l2-category",
        multi_image=False,
        image_in_question=False,
        task_type="multiple_choice",
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get a dataset config by name."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]


def parse_options(options):
    """Parse multiple choice options into formatted string."""
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            options = [options]
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{letter}. {option}" for letter, option in zip(option_letters, options)])
    return choices_str


def extract_images(data: Dict, config: DatasetConfig) -> List[Image.Image]:
    """Extract images from dataset sample based on config."""
    images = []

    if config.multi_image and config.image_in_question:
        question = data.get(config.question_field, "")
        image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", question)]
        for idx in image_order:
            img_key = f"{config.image_field}_{idx}"
            if img_key in data and data[img_key] is not None:
                images.append(data[img_key])
    elif config.multi_image:
        idx = 1
        while f"{config.image_field}_{idx}" in data:
            img = data[f"{config.image_field}_{idx}"]
            if img is not None:
                images.append(img)
            idx += 1
    else:
        if config.image_field in data and data[config.image_field] is not None:
            images.append(data[config.image_field])

    return images


def construct_prompt(data: Dict, config: DatasetConfig, prompt_suffix: str = "") -> str:
    """Construct prompt from dataset sample using config."""
    question = data.get(config.question_field, "")

    # Preformatted: question already has options + suffix (e.g., VLMEvalKit AI2D)
    if config.task_type == "preformatted":
        return question

    if config.image_in_question:
        question = re.sub(r"<image\s+\d+>", "[image]", question)

    # Handle MMBench hint field
    hint = data.get("hint", "")
    if hint and str(hint).lower() not in ("nan", "none", ""):
        question = f"{hint}\n{question}"

    # Detect open-ended questions (MMMU fill-in-the-blank: options=[] or question_type=open)
    is_open = data.get("question_type") == "open"
    if not is_open and config.options_field and config.options_field in data:
        options = data[config.options_field]
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except (ValueError, SyntaxError):
                options = [options]
        is_open = not options or options == ["", "", "", ""]

    # Add options (skip for yes/no tasks like MME and open-ended)
    if not is_open and config.task_type != "yes_no" and config.options_field and config.options_field in data:
        options = data[config.options_field]
        options_str = parse_options(options)
        question = f"{question}\n{options_str}"

    if config.task_type != "yes_no" and prompt_suffix:
        if is_open:
            question = f"{question}\nAnswer concisely with just the final value or term. Answer:"
        else:
            question = f"{question}\n{prompt_suffix}"

    return question


def process_prompt(data: Dict, config: DatasetConfig, prompt_suffix: str = "") -> tuple:
    """Process dataset sample into (prompt, images) tuple."""
    prompt = construct_prompt(data, config, prompt_suffix)
    images = extract_images(data, config)
    return (prompt, images)


def get_prompt_suffix(mode: str, prompts_path: str = None) -> str:
    """Get prompt suffix from prompts.yaml based on mode."""
    if prompts_path is None:
        prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.yaml")
    with open(prompts_path, "r") as f:
        prompt_config = yaml.safe_load(f)[mode]
    return prompt_config.get("standard", "")


def setup_output_path(args) -> str:
    """Setup output path: output_dir/{model_name}/{dataset}/{split}_{mode}.jsonl"""
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset
    split = args.split
    mode = getattr(args, "mode", "direct")
    output_dir = getattr(args, "output_dir", "./output")

    if not os.path.isabs(output_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, output_dir.lstrip("./"))

    filename = f"{split}_{mode}.jsonl"
    output_path = os.path.join(output_dir, model_name, dataset_name, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def _json_default(obj):
    """Handle numpy/pandas types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_results(results: List[Dict], output_path: str):
    """Save results to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + "\n")
    print(f"Saved {len(results)} results to {output_path}")


def load_existing_results(output_path: str) -> List[Dict]:
    """Load existing results from JSONL file for resume support."""
    if not os.path.exists(output_path):
        return []
    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    if results:
        print(f"Loaded {len(results)} existing results from {output_path}")
    return results


def add_common_args(parser):
    """Add common arguments shared by all inference scripts."""
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to evaluate")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (default: mmt_bench=VAL, mmmu=dev_val, scienceqa=test)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override default data directory")
    parser.add_argument("--images_only", action="store_true",
                        help="ScienceQA: only use samples with images")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "cot"],
                        help="Prompt mode")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                        help="Run on first 100 samples only")
    parser.add_argument("--prompt_suffix", type=str, default=None,
                        help="Custom prompt suffix (overrides prompts.yaml)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save intermediate results every N samples")


def load_dataset_from_args(args):
    """Load dataset based on CLI args, using dataset_loaders."""
    from dataset_loaders import load_dataset_by_name

    defaults = {"mmt_bench": "VAL", "mmmu": "dev_val", "scienceqa": "test", "mme": "test", "ai2d": "test", "mathvista": "testmini", "mmbench": "DEV_EN", "blink": "val"}
    if args.split is None:
        args.split = defaults.get(args.dataset, "test")

    dataset = load_dataset_by_name(
        args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        images_only=args.images_only,
    )

    if args.debug:
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(min(100, len(dataset))))
        else:
            # List-like (MMT-Bench)
            class SlicedDataset:
                def __init__(self, ds, n):
                    self._ds = ds
                    self._n = min(n, len(ds))
                def __len__(self):
                    return self._n
                def __getitem__(self, idx):
                    return self._ds[idx]
            dataset = SlicedDataset(dataset, 100)
        print(f"Debug mode: using first {len(dataset)} samples")

    return dataset
