# Inverse Scaling in Vision-Language Models

Code for the paper *On the Limits of Scale: Inverse Scaling in Vision-Language Models*. Provided for reviewer reference only. **Please do not share or redistribute.**

## Setup

```bash
pip install torch transformers datasets accelerate pillow pyyaml tqdm
pip install flash-attn  # for InternVL
pip install qwen-vl-utils  # for Qwen3-VL
```

Download benchmark datasets:
```bash
python setup_data.py
```

By default, `setup_data.py`, the inference scripts and the analysis scripts use the `data/`, `output/` and `analysis/` directories located one level above this repository (the parent of the checkout). Pass `--output_dir` / `--csv_dir` to override.

## Running Inference

Each model family has its own inference script. All use zero-shot direct prompting with greedy decoding.

```bash
# Example: run Qwen3-VL-8B on BLINK
python infer_qwen3vl.py --model Qwen/Qwen3-VL-8B-Instruct --dataset blink --mode direct

# Example: run InternVL3.5-14B on MMMU
python infer_internvl.py --model OpenGVLab/InternVL3_5-14B-Instruct --dataset mmmu --mode direct

# Gemma 3
python infer_gemma3.py --model google/gemma-3-12b-it --dataset mmt_bench --mode direct

# Molmo 2
python infer_molmo2.py --model allenai/Molmo2-8B --dataset mmbench --mode direct
```

Results are saved to `output/{model_name}/{dataset}/{split}_direct.jsonl` (see the note on default directories above).

Available datasets: `mmt_bench`, `mmmu`, `scienceqa`, `mme`, `ai2d`, `mathvista`, `blink`, `mmbench`.

ScienceQA is filtered to the 2,017 questions that have images (text-only questions excluded).

## Analysis

Find inverse scaling cases using pairwise flip analysis:

```bash
python analyze_flips.py --family all --dataset all
```

This outputs CSVs to `analysis/` (default location as above) with all subtask-model pairs where a smaller model outperforms a larger one (Net >= 5, n >= 20).

To view per-subtask accuracy breakdowns:
```bash
python find_inverse_scaling.py --family qwen --dataset blink
```

## Answer Verification

Responses are scored using deterministic pattern matching (letter extraction for MCQ, numeric matching for open-ended). For the ~0.4% of responses where extraction fails, GPT-5.4 adjudicates. See `evaluate.py` for details.

## Pre-computed Results

The `results/` directory contains the 375 pairwise inverse scaling cases reported in the paper:

- `results/inverse_scaling_qwen.csv` — Qwen3-VL (122 cases)
- `results/inverse_scaling_internvl.csv` — InternVL3.5 (207 cases)
- `results/inverse_scaling_gemma3.csv` — Gemma 3 (26 cases)
- `results/inverse_scaling_molmo2.csv` — Molmo 2 (20 cases)

Each CSV contains: dataset, subtask, small\_model, large\_model, n, acc\_small, acc\_large, delta\_pp, flip\_inv, flip\_norm, net, both\_right, both\_wrong.

## File Overview

- `infer_*.py` — inference scripts for each model family
- `infer_utils.py` — shared prompt construction and dataset loading
- `evaluate.py` — answer extraction and matching
- `find_inverse_scaling.py` — per-subtask accuracy and scoring
- `analyze_flips.py` — pairwise flip analysis (produces the main results)
- `llm_judge.py` — GPT fallback judge for unextractable responses
- `dataset_loaders.py` — benchmark dataset loaders
- `prompts.yaml` — prompt templates
- `setup_data.py` — dataset download script
- `results/` — pre-computed inverse scaling cases (375 total)
