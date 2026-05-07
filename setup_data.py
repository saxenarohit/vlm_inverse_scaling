"""
Download and set up datasets for inverse scaling experiments.

Usage:
    python setup_data.py --dataset all
    python setup_data.py --dataset mmmu
    python setup_data.py --dataset scienceqa
    python setup_data.py --dataset mmt_bench
    python setup_data.py --dataset mme

Data directory structure after setup:
    data/
    ├── mmmu/           # HuggingFace Arrow files
    │   ├── dev/
    │   ├── validation/
    │   └── test/
    ├── scienceqa/      # HuggingFace Arrow files
    │   ├── test/
    │   └── validation/
    ├── mmt_bench/      # TSV with base64 images
    │   └── MMT-Bench_VAL.tsv
    └── mme/            # Images + text QA pairs
        └── MME_Benchmark_release_version/
            └── MME_Benchmark/
                ├── existence/
                ├── count/
                └── ...
"""
import os
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def setup_mmmu():
    """Download MMMU from HuggingFace and save as Arrow files.

    Source: MMMU/MMMU on HuggingFace
    Splits: dev (150), validation (900), test (10500)
    dev_val = dev + validation = 1050 (matches MMMU leaderboard DEV_VAL)
    """
    from datasets import load_dataset

    out_dir = os.path.join(DATA_DIR, "mmmu")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["dev", "validation", "test"]:
        split_dir = os.path.join(out_dir, split)
        if os.path.exists(split_dir):
            print(f"  MMMU {split}: already exists at {split_dir}, skipping")
            continue
        print(f"  Downloading MMMU {split}...")
        ds = load_dataset("MMMU/MMMU", split=split)
        ds.save_to_disk(split_dir)
        print(f"  Saved {len(ds)} samples to {split_dir}")

    print(f"MMMU setup complete: {out_dir}")


def setup_scienceqa():
    """Download ScienceQA from HuggingFace and save as Arrow files.

    Source: derek-thomas/ScienceQA on HuggingFace
    We use images_only=True during inference (4241 samples with images in test).
    """
    from datasets import load_dataset

    out_dir = os.path.join(DATA_DIR, "scienceqa")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["test", "validation"]:
        split_dir = os.path.join(out_dir, split)
        if os.path.exists(split_dir):
            print(f"  ScienceQA {split}: already exists at {split_dir}, skipping")
            continue
        print(f"  Downloading ScienceQA {split}...")
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        ds.save_to_disk(split_dir)
        print(f"  Saved {len(ds)} samples to {split_dir}")

    print(f"ScienceQA setup complete: {out_dir}")


def setup_mmt_bench():
    """Download MMT-Bench TSV file.

    Source: https://huggingface.co/datasets/OpenGVLab/MMT-Bench/tree/main
    The VAL split TSV (~602MB) contains base64-encoded images inline.
    Downloaded directly as a single file (not via datasets library).
    """
    out_dir = os.path.join(DATA_DIR, "mmt_bench")
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = os.path.join(out_dir, "MMT-Bench_VAL.tsv")
    if os.path.exists(tsv_path):
        print(f"  MMT-Bench VAL: already exists at {tsv_path}, skipping")
        print(f"MMT-Bench setup complete: {out_dir}")
        return

    print("  Downloading MMT-Bench VAL...")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id="OpenGVLab/MMT-Bench",
        filename="MMT-Bench_VAL.tsv",
        repo_type="dataset",
        local_dir=out_dir,
    )
    print(f"  Saved to {downloaded}")
    print(f"MMT-Bench setup complete: {out_dir}")


def setup_ai2d():
    """Download AI2D_TEST (VLMEvalKit-processed with masked diagram text).

    Source: OpenGVLab/InternVL on HuggingFace (model repo)
    Files: AI2D_TEST.zip (112MB, masked images) + ai2d_test_vlmevalkit.jsonl (1MB, metadata)
    Split: test (3088 samples), 15 categories
    This is the "AI2D_test" variant reported in papers (Qwen3, InternVL).
    Text in diagrams is masked and replaced with letter labels.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    out_dir = os.path.join(DATA_DIR, "ai2d")
    image_dir = os.path.join(out_dir, "AI2D_TEST")

    if os.path.isdir(image_dir) and os.path.exists(os.path.join(out_dir, "ai2d_test_vlmevalkit.jsonl")):
        print(f"  AI2D_TEST: already exists at {out_dir}, skipping")
        print(f"AI2D setup complete: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("  Downloading ai2d_test_vlmevalkit.jsonl...")
    hf_hub_download(
        repo_id="OpenGVLab/InternVL",
        filename="ai2d_test_vlmevalkit.jsonl",
        repo_type="model",
        local_dir=out_dir,
    )

    print("  Downloading AI2D_TEST.zip...")
    zip_path = hf_hub_download(
        repo_id="OpenGVLab/InternVL",
        filename="AI2D_TEST.zip",
        repo_type="model",
        local_dir=out_dir,
    )

    print("  Extracting images...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print(f"AI2D setup complete: {out_dir}")


def setup_mme():
    """Set up MME benchmark.

    Source: https://huggingface.co/datasets/lmms-lab/MME
    MME requires downloading the benchmark release and extracting it.

    The expected structure is:
        data/mme/MME_Benchmark_release_version/MME_Benchmark/
            ├── existence/    (flat: NNNN.jpg + NNNN.txt)
            ├── count/
            ├── position/
            ├── color/
            ├── posters/      (images/ + questions_answers_YN/)
            ├── celebrity/
            ├── scene/
            ├── landmark/
            ├── artwork/
            ├── OCR/
            ├── commonsense_reasoning/
            ├── numerical_calculation/
            ├── text_translation/
            └── code_reasoning/
    """
    out_dir = os.path.join(DATA_DIR, "mme")
    bench_dir = os.path.join(out_dir, "MME_Benchmark_release_version", "MME_Benchmark")

    if os.path.isdir(bench_dir):
        print(f"  MME: already exists at {bench_dir}, skipping")
        print(f"MME setup complete: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print("  Downloading MME benchmark...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="lmms-lab/MME",
            repo_type="dataset",
            local_dir=out_dir,
        )
        print(f"  Downloaded to {out_dir}")
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        print(f"  Manual setup:")
        print(f"    1. Download from https://huggingface.co/datasets/lmms-lab/MME")
        print(f"    2. Extract to {out_dir}/MME_Benchmark_release_version/MME_Benchmark/")

    print(f"MME setup complete: {out_dir}")


def setup_mathvista():
    """Download MathVista from HuggingFace and save as Arrow files.

    Source: AI4Math/MathVista on HuggingFace
    Splits: testmini (1000), test (5141 -- answers NOT public)
    We use testmini, which is the standard split papers report.
    Mixed format: 54% MCQ + 46% free-form (integer/float/text).
    5 task types: FQA, GPS, MWP, TQA, VQA.
    """
    from datasets import load_dataset

    out_dir = os.path.join(DATA_DIR, "mathvista")
    os.makedirs(out_dir, exist_ok=True)

    split_dir = os.path.join(out_dir, "testmini")
    if os.path.exists(split_dir):
        print(f"  MathVista testmini: already exists at {split_dir}, skipping")
        print(f"MathVista setup complete: {out_dir}")
        return

    print("  Downloading MathVista testmini...")
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    ds.save_to_disk(split_dir)
    print(f"  Saved {len(ds)} samples to {split_dir}")
    print(f"MathVista setup complete: {out_dir}")


def setup_blink():
    """Download BLINK from HuggingFace and save as Arrow files.

    Source: BLINK-Benchmark/BLINK on HuggingFace
    Split: val (1901 samples), 14 subtasks
    Multi-image MCQ with pre-formatted prompts.
    """
    from datasets import load_dataset, concatenate_datasets

    out_dir = os.path.join(DATA_DIR, "blink")
    val_dir = os.path.join(out_dir, "val")
    if os.path.exists(val_dir):
        print(f"  BLINK val: already exists at {val_dir}, skipping")
        print(f"BLINK setup complete: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    configs = ['Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence',
               'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
               'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
               'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']

    all_val = []
    for cfg in configs:
        print(f"  Loading {cfg}...")
        ds = load_dataset("BLINK-Benchmark/BLINK", name=cfg, split="val")
        all_val.append(ds)

    combined = concatenate_datasets(all_val)
    combined.save_to_disk(val_dir)
    print(f"  Saved {len(combined)} val samples to {val_dir}")
    print(f"BLINK setup complete: {out_dir}")


def setup_mmbench():
    """Download MMBench DEV from OpenCompass.

    Source: http://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
    Split: DEV_EN (4329 samples), all MCQ, 6 l2-categories
    TSV with base64-encoded images (same format as MMT-Bench).
    Has hint field for additional context (322/4329 samples).
    """
    import urllib.request

    out_dir = os.path.join(DATA_DIR, "mmbench")
    os.makedirs(out_dir, exist_ok=True)

    tsv_path = os.path.join(out_dir, "MMBench_DEV_EN.tsv")
    if os.path.exists(tsv_path):
        print(f"  MMBench DEV_EN: already exists at {tsv_path}, skipping")
        print(f"MMBench setup complete: {out_dir}")
        return

    url = "http://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv"
    print(f"  Downloading MMBench DEV_EN from {url}...")
    urllib.request.urlretrieve(url, tsv_path)
    print(f"  Saved to {tsv_path}")
    print(f"MMBench setup complete: {out_dir}")


SETUP_FNS = {
    "mmmu": setup_mmmu,
    "scienceqa": setup_scienceqa,
    "mmt_bench": setup_mmt_bench,
    "mme": setup_mme,
    "ai2d": setup_ai2d,
    "mathvista": setup_mathvista,
    "blink": setup_blink,
    "mmbench": setup_mmbench,
}


def main():
    parser = argparse.ArgumentParser(description="Download and set up datasets")
    parser.add_argument("--dataset", required=True,
                        choices=list(SETUP_FNS.keys()) + ["all"],
                        help="Dataset to set up")
    args = parser.parse_args()

    datasets = list(SETUP_FNS.keys()) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Setting up: {name}")
        print(f"{'='*60}")
        SETUP_FNS[name]()

    print(f"\nDone. Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
