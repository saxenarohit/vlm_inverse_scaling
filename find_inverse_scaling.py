"""
Find inverse scaling patterns across model sizes and datasets.

Usage:
    python find_inverse_scaling.py --family qwen
    python find_inverse_scaling.py --family qwen --dataset mmt_bench
    python find_inverse_scaling.py --family qwen --min_n 10 --min_delta 10
"""
import json
import math
import os
import re
import argparse
from collections import defaultdict
from evaluate import extract_answer_letter, extract_numbers

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ScienceQA image filter: skip text-only questions (no image in source dataset)
_scienceqa_has_image = None

def _get_scienceqa_image_set():
    """Build set of ScienceQA indices that have images. Cached after first call."""
    global _scienceqa_has_image
    if _scienceqa_has_image is not None:
        return _scienceqa_has_image
    from datasets import load_from_disk
    ds_path = os.path.join(DATA_DIR, "scienceqa", "test")
    if not os.path.exists(ds_path):
        _scienceqa_has_image = None
        return None
    ds = load_from_disk(ds_path)
    _scienceqa_has_image = set()
    for i in range(len(ds)):
        if ds[i].get("image") is not None:
            _scienceqa_has_image.add(i)
    return _scienceqa_has_image


def _skip_question(dataset, line_idx):
    """Return True if this question should be skipped (e.g. text-only ScienceQA)."""
    if dataset != "scienceqa":
        return False
    img_set = _get_scienceqa_image_set()
    if img_set is None:
        return False
    return line_idx not in img_set

FAMILIES = {
    "qwen": {
        "models": ["2B", "4B", "8B", "30B-A3B", "32B", "235B-A22B"],
        "name_template": "Qwen3-VL-{}-Instruct",
    },
    "internvl": {
        "models": ["1B", "2B", "4B", "8B", "14B", "30B-A3B", "38B", "241B-A28B"],
        "name_template": "InternVL3_5-{}-Instruct",
    },
    "gemma3": {
        "models": ["4b", "12b", "27b"],
        "name_template": "gemma-3-{}-it",
    },
    "molmo2": {
        "models": ["4B", "8B"],
        "name_template": "Molmo2-{}",
    },
}

DATASETS = {
    "mmt_bench": {"file": "VAL_direct.jsonl", "category_field": "l2-category"},
    "mmmu": {"file": "dev_val_direct.jsonl", "category_field": "subfield"},
    "mmmu_test": {"file": "test_direct.jsonl", "category_field": "subfield", "dir": "mmmu"},
    "scienceqa": {"file": "test_direct.jsonl", "category_field": "category"},
    "mme": {"file": "test_direct.jsonl", "category_field": "category"},
    "ai2d": {"file": "test_direct.jsonl", "category_field": "category"},
    "mathvista": {"file": "testmini_direct.jsonl", "category_field": "task"},
    "blink": {"file": "val_direct.jsonl", "category_field": "sub_task"},
    "mmbench": {"file": "DEV_EN_direct.jsonl", "category_field": "l2-category"},
}


def _is_open_ended(r):
    """Check if a sample is an open-ended (non-MCQ) question."""
    if r.get("question_type") == "open":
        return True
    ans = r.get("answer", "").strip()
    if ans and len(ans) > 1 and ans[0].upper() not in "ABCDEFGHI":
        return True
    return False


def _extract_last_number(text):
    """Extract the last number from a response, handling commas and boxed answers."""
    # Try boxed answer first: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        try:
            return float(boxed[-1].replace(",", "").strip())
        except ValueError:
            pass
    # Try "Final Answer: X" or "answer is X" patterns (supports comma-separated numbers)
    num_pat = r'([+-]?\d[\d,]*(?:\.\d+)?)'
    final_patterns = [
        r'[Ff]inal [Aa]nswer[:\s]*' + num_pat,
        r'[Aa]nswer[:\s]*' + num_pat + r'\s*$',
        r'[Tt]herefore[,\s]+.*?' + num_pat + r'\s*$',
        r'=\s*' + num_pat + r'\s*$',
    ]
    for pat in final_patterns:
        matches = re.findall(pat, text, re.MULTILINE)
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except ValueError:
                pass
    # Fall back to last comma-aware number in the text
    # Match numbers like 9,606 or 3.14 or 42
    all_nums = re.findall(r'[+-]?\d[\d,]*(?:\.\d+)?', text)
    if all_nums:
        try:
            return float(all_nums[-1].replace(",", ""))
        except (ValueError, TypeError):
            pass
    return None


def _match_open_ended(resp, ans):
    """Match open-ended response against ground truth (numeric or string)."""
    resp_clean = resp.strip().rstrip(".")
    ans = ans.strip()
    # Exact match
    if resp_clean.lower() == ans.lower():
        return True
    # If answer is a single letter, treat as MCQ (some MMMU open questions have letter answers)
    if len(ans) == 1 and ans.upper() in "ABCDEFGHI":
        letter = extract_answer_letter(resp)
        if letter:
            return letter == ans.upper()
        resp_letter = resp_clean[0].upper() if resp_clean and resp_clean[0].upper() in "ABCDEFGHI" else resp_clean
        return resp_letter == ans.upper()
    # Try numeric match: extract last number from response
    try:
        a_val = float(ans.replace(",", ""))
    except ValueError:
        # Non-numeric answer: check if answer appears in response
        return ans.lower() in resp_clean.lower()
    # First try parsing the full response as a number
    try:
        r_val = float(resp_clean.replace(",", ""))
    except ValueError:
        # Extract last number from response
        r_val = _extract_last_number(resp)
    if r_val is not None:
        if a_val == 0:
            return abs(r_val) < 0.01
        # Integer answers: exact match. Float answers: 5% tolerance.
        if not (math.isinf(a_val) or math.isnan(a_val)) and a_val == int(a_val):
            if not (math.isinf(r_val) or math.isnan(r_val)) and r_val == int(r_val):
                return int(r_val) == int(a_val)
            return False
        return abs(r_val - a_val) / max(abs(a_val), 1e-9) < 0.05  # 5% tolerance for floats
    return False


def _match_response(resp, ans, dataset, record=None):
    """Match model response against ground truth answer."""
    if dataset == "mme":
        # Yes/No matching
        resp_lower = resp.lower().strip()
        ans_lower = ans.lower().strip()
        # Extract yes/no from response
        if "yes" in resp_lower:
            pred = "yes"
        elif "no" in resp_lower:
            pred = "no"
        else:
            pred = resp_lower
        return pred == ans_lower
    # Open-ended (MMMU fill-in-the-blank, MathVista free-form)
    if record and _is_open_ended(record):
        return _match_open_ended(resp, ans)
    # MCQ letter matching — use robust extraction
    resp_letter = extract_answer_letter(resp)
    if resp_letter:
        return resp_letter == ans
    # Fallback: first character
    if resp and resp[0].upper() in "ABCDEFGHI":
        return resp[0].upper() == ans
    # Fallback: match boxed/dollar/bold values against option text
    if record:
        options = record.get("options", record.get("choices", []))
        if isinstance(options, list) and options:
            # Extract value from \boxed{...}, $...$, or **...**
            value = None
            m = re.search(r'\\boxed\{([^}]+)\}', resp)
            if m:
                value = m.group(1).strip()
            elif re.search(r'\$([^$]+)\$\s*$', resp.strip()):
                value = re.search(r'\$([^$]+)\$\s*$', resp.strip()).group(1).strip()
            elif re.search(r'\*\*([^*]+)\*\*\s*$', resp.strip()):
                value = re.search(r'\*\*([^*]+)\*\*\s*$', resp.strip()).group(1).strip()
            if value:
                for i, opt in enumerate(options):
                    opt_clean = str(opt).strip().rstrip('°').strip()
                    if value == opt_clean or value.rstrip('°').strip() == opt_clean:
                        matched_letter = chr(65 + i)
                        return matched_letter == ans
    # GPT judge fallback for unextractable MCQ responses
    if record and "judge_correct" in record and record.get("judge_model") == "gpt-5.4":
        return record["judge_correct"]
    return False


def _dataset_path(output_dir, model_dir, dataset):
    """Get file path for a dataset, using 'dir' override if present."""
    ds_dir = DATASETS[dataset].get("dir", dataset)
    return os.path.join(output_dir, model_dir, ds_dir, DATASETS[dataset]["file"])


def load_per_category_accuracy(output_dir, model_dir, dataset, category_field):
    """Load results and compute per-category accuracy."""
    path = _dataset_path(output_dir, model_dir, dataset)
    if not os.path.exists(path):
        return None

    cats = defaultdict(lambda: {"correct": 0, "total": 0})
    with open(path) as f:
        for line_idx, line in enumerate(f):
            if _skip_question(dataset, line_idx):
                continue
            r = json.loads(line)
            cat = r.get(category_field, "unknown") or "unknown"
            resp = r.get("response", "").strip()
            ans = r.get("answer", "").strip()
            cats[cat]["total"] += 1
            if resp and _match_response(resp, ans, dataset, record=r):
                cats[cat]["correct"] += 1

    return {cat: d["correct"] / d["total"] * 100 for cat, d in cats.items() if d["total"] > 0}, \
           {cat: d["total"] for cat, d in cats.items()}


def find_inverse_scaling(family_config, dataset, output_dir, min_n=5, min_delta=5):
    """Find subtasks where a smaller model outperforms the largest model."""
    models = family_config["models"]
    name_template = family_config["name_template"]
    cat_field = DATASETS[dataset]["category_field"]
    largest = models[-1]

    # Load per-category accuracy for each model
    model_accs = {}
    model_ns = {}
    for m in models:
        model_dir = name_template.format(m)
        result = load_per_category_accuracy(output_dir, model_dir, dataset, cat_field)
        if result is None:
            continue
        accs, ns = result
        model_accs[m] = accs
        model_ns[m] = ns

    if largest not in model_accs:
        return [], model_accs

    # Compute overall accuracy
    overall = {}
    for m in models:
        if m in model_accs:
            model_dir = name_template.format(m)
            path = _dataset_path(output_dir, model_dir, dataset)
            total = correct = 0
            with open(path) as f:
                for line_idx, line in enumerate(f):
                    if _skip_question(dataset, line_idx):
                        continue
                    r = json.loads(line)
                    resp = r.get("response", "").strip()
                    ans = r.get("answer", "").strip()
                    total += 1
                    if _match_response(resp, ans, dataset, record=r):
                        correct += 1
            overall[m] = (correct, total, correct / total * 100 if total else 0)

    # Find all categories
    all_cats = set()
    for m in model_accs:
        all_cats.update(model_accs[m].keys())

    inverse_cases = []
    for cat in sorted(all_cats):
        accs = {}
        ns = {}
        for m in models:
            if m in model_accs and cat in model_accs[m]:
                accs[m] = model_accs[m][cat]
                ns[m] = model_ns[m][cat]

        if len(accs) < 2 or largest not in accs:
            continue
        if ns.get(largest, 0) < min_n:
            continue

        acc_largest = accs[largest]
        smaller_models = [m for m in models[:-1] if m in accs]
        if not smaller_models:
            continue

        max_smaller_acc = max(accs[m] for m in smaller_models)
        if max_smaller_acc > acc_largest + min_delta:
            best_small = [m for m in smaller_models if accs[m] == max_smaller_acc][0]
            delta = max_smaller_acc - acc_largest
            n = ns[largest]
            acc_str = " | ".join([f"{m}={accs[m]:.1f}" for m in models if m in accs])
            inverse_cases.append((delta, cat, n, best_small, acc_str))

    inverse_cases.sort(reverse=True)
    return inverse_cases, overall


def main():
    parser = argparse.ArgumentParser(description="Find inverse scaling patterns")
    parser.add_argument("--family", required=True, choices=list(FAMILIES.keys()) + ["all"])
    parser.add_argument("--dataset", default="all", choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--min_n", type=int, default=5, help="Minimum samples per subtask")
    parser.add_argument("--min_delta", type=float, default=5.0, help="Minimum accuracy gap (pp)")
    args = parser.parse_args()

    families = list(FAMILIES.keys()) if args.family == "all" else [args.family]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for family in families:
        family_config = FAMILIES[family]
        print(f"\n{'#'*80}")
        print(f"# Family: {family} | Models: {', '.join(family_config['models'])}")
        print(f"{'#'*80}")

        for dataset in datasets:
            inverse_cases, overall = find_inverse_scaling(
                family_config, dataset, args.output_dir,
                min_n=args.min_n, min_delta=args.min_delta,
            )

            print(f"\n{'='*80}")
            print(f"Dataset: {dataset}")
            print(f"{'='*80}")

            if overall:
                print("\nOverall accuracy:")
                for m in family_config["models"]:
                    if m in overall:
                        correct, total, acc = overall[m]
                        print(f"  {m:>8}: {acc:.2f}% ({correct}/{total})")
            else:
                print("  No results found.")
                continue

            if inverse_cases:
                print(f"\nInverse scaling (smaller beats largest by >{args.min_delta}pts, n>={args.min_n}):")
                print(f"  {'Subtask':<45} {'n':>4} {'Best':>5} {'Delta':>6}  Scores")
                print(f"  {'-'*110}")
                for delta, cat, n, best, acc_str in inverse_cases:
                    print(f"  {cat:<45} {n:>4} {best:>5} {delta:>+6.1f}  {acc_str}")
            else:
                print(f"\n  No inverse scaling found (>{args.min_delta}pt threshold, n>={args.min_n})")


if __name__ == "__main__":
    main()
