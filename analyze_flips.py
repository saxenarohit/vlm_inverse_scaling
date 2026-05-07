"""
Per-question flip analysis for inverse scaling — all pairwise comparisons.

For every pair of models (smaller, larger) within a family, and for every subtask,
computes how many individual questions the smaller model got right that the larger
got wrong (flip_inv) and vice versa (flip_norm). Reports net = flip_inv - flip_norm.

Filters: n >= 20 AND net >= 5. Delta is computed for viewing, not as a filter.
Outputs separate CSV per family.

Usage:
    python analyze_flips.py --family all --dataset all
    python analyze_flips.py --family internvl --dataset blink
"""
import csv
import json
import os
import argparse
from collections import defaultdict
from find_inverse_scaling import (
    FAMILIES, DATASETS, OUTPUT_DIR,
    _dataset_path, _match_response, _skip_question,
)

# Models to exclude per family (e.g., incomplete results)
EXCLUDE_MODELS = {
}


def get_question_id(record, line_idx):
    """Get a unique question ID from a record."""
    for key in ("idx", "index", "id"):
        if key in record:
            return record[key]
    return line_idx


def load_per_question_results(output_dir, model_dir, dataset, cat_field):
    """Load per-question correctness grouped by category.

    Returns: {category: {question_id: correct_bool}}
    """
    path = _dataset_path(output_dir, model_dir, dataset)
    if not os.path.exists(path):
        return None

    results = defaultdict(dict)
    with open(path) as f:
        for line_idx, line in enumerate(f):
            if _skip_question(dataset, line_idx):
                continue
            r = json.loads(line)
            cat = r.get(cat_field, "unknown") or "unknown"
            resp = r.get("response", "").strip()
            ans = r.get("answer", "").strip()
            qid = get_question_id(r, line_idx)
            if not resp:
                correct = False  # Empty response treated as incorrect
            else:
                correct = bool(_match_response(resp, ans, dataset, record=r))
            results[cat][qid] = correct

    return results


def compute_flips(qs_small, qs_large):
    """Compute flip counts between two models on shared questions."""
    common_qids = set(qs_small.keys()) & set(qs_large.keys())
    both_right = 0
    both_wrong = 0
    flip_inv = 0   # smaller right, larger wrong
    flip_norm = 0  # smaller wrong, larger right

    for qid in common_qids:
        s = qs_small[qid]
        l = qs_large[qid]
        if s and l:
            both_right += 1
        elif s and not l:
            flip_inv += 1
        elif not s and l:
            flip_norm += 1
        else:
            both_wrong += 1

    return flip_inv, flip_norm, both_right, both_wrong, len(common_qids)


def analyze_flips_pairwise(family, family_config, dataset, output_dir, min_n=20, min_net=5):
    """For all model pairs, find subtasks where smaller model has net >= min_net flips."""
    excluded = EXCLUDE_MODELS.get(family, set())
    models = [m for m in family_config["models"] if m not in excluded]
    name_template = family_config["name_template"]
    cat_field = DATASETS[dataset]["category_field"]

    # Load per-question results for all models
    model_results = {}
    for m in models:
        model_dir = name_template.format(m)
        result = load_per_question_results(output_dir, model_dir, dataset, cat_field)
        if result is None:
            continue
        model_results[m] = result

    if len(model_results) < 2:
        return []

    # All subtasks
    all_cats = set()
    for m in model_results:
        all_cats.update(model_results[m].keys())

    flip_cases = []
    # All pairs where smaller_idx < larger_idx in the model list
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            small_m = models[i]
            large_m = models[j]
            if small_m not in model_results or large_m not in model_results:
                continue

            for cat in sorted(all_cats):
                qs_small = model_results[small_m].get(cat, {})
                qs_large = model_results[large_m].get(cat, {})
                if not qs_small or not qs_large:
                    continue

                flip_inv, flip_norm, both_right, both_wrong, matched = compute_flips(qs_small, qs_large)
                net = flip_inv - flip_norm

                # Apply filters
                if matched < min_n:
                    continue
                if net < min_net:
                    continue

                # Compute accuracies on matched questions
                acc_small = (both_right + flip_inv) / matched * 100 if matched else 0
                acc_large = (both_right + flip_norm) / matched * 100 if matched else 0
                delta = acc_small - acc_large

                flip_cases.append({
                    "dataset": dataset,
                    "subtask": cat,
                    "small_model": small_m,
                    "large_model": large_m,
                    "n": matched,
                    "acc_small": round(acc_small, 1),
                    "acc_large": round(acc_large, 1),
                    "delta_pp": round(delta, 1),
                    "flip_inv": flip_inv,
                    "flip_norm": flip_norm,
                    "net": net,
                    "both_right": both_right,
                    "both_wrong": both_wrong,
                })

    flip_cases.sort(key=lambda x: -x["net"])
    return flip_cases


CSV_FIELDS = [
    "dataset", "subtask", "small_model", "large_model", "n",
    "acc_small", "acc_large", "delta_pp",
    "flip_inv", "flip_norm", "net",
    "both_right", "both_wrong",
]


def main():
    parser = argparse.ArgumentParser(description="Per-question pairwise flip analysis")
    parser.add_argument("--family", required=True, choices=list(FAMILIES.keys()) + ["all"])
    parser.add_argument("--dataset", default="all", choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--min_n", type=int, default=20, help="Minimum samples per subtask")
    parser.add_argument("--min_net", type=int, default=5, help="Minimum net flips (flip_inv - flip_norm)")
    parser.add_argument("--csv_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"),
                        help="Directory for CSV output")
    args = parser.parse_args()

    families = list(FAMILIES.keys()) if args.family == "all" else [args.family]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    os.makedirs(args.csv_dir, exist_ok=True)

    grand_total = 0
    for family in families:
        family_config = FAMILIES[family]
        excluded = EXCLUDE_MODELS.get(family, set())
        active_models = [m for m in family_config["models"] if m not in excluded]

        print(f"\n{'#'*80}")
        print(f"# Family: {family} | Models: {', '.join(active_models)}")
        if excluded:
            print(f"# Excluded: {', '.join(excluded)}")
        print(f"{'#'*80}")

        all_cases = []
        for dataset in datasets:
            cases = analyze_flips_pairwise(
                family, family_config, dataset, args.output_dir,
                min_n=args.min_n, min_net=args.min_net,
            )

            if not cases:
                continue

            all_cases.extend(cases)
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset}  ({len(cases)} cases)")
            print(f"{'='*80}")
            print(f"  {'Subtask':<35} {'Small':>8} {'Large':>10} {'n':>4} {'Delta':>6} {'FI':>4} {'FN':>4} {'Net':>5}")
            print(f"  {'-'*85}")
            for c in cases:
                print(f"  {c['subtask']:<35} {c['small_model']:>8} {c['large_model']:>10} {c['n']:>4} {c['delta_pp']:>+6.1f} {c['flip_inv']:>4} {c['flip_norm']:>4} {c['net']:>+5}")

        # Write CSV for this family
        if all_cases:
            csv_path = os.path.join(args.csv_dir, f"inverse_scaling_{family}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(all_cases)
            print(f"\n  -> Saved {len(all_cases)} cases to {csv_path}")

        grand_total += len(all_cases)
        print(f"\n  Family total: {len(all_cases)} cases")

    print(f"\n{'='*80}")
    print(f"Grand total (n>={args.min_n}, net>={args.min_net}): {grand_total}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
