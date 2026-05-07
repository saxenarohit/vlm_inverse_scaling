"""
Unified dataset loaders for inverse scaling experiments.

Loads MMT-Bench, MMMU, and ScienceQA from local cached files
and normalizes them to a common format compatible with infer_utils.DatasetConfig.
"""
import os
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
from datasets import load_from_disk, concatenate_datasets


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def decode_base64_image(base64_str):
    """Decode base64-encoded image string to PIL Image."""
    if pd.isna(base64_str) or not base64_str:
        return None
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    try:
        # Fix padding if needed
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to decode image: {e}")
        return None


class MMTBenchDataset:
    """Lazy-loading MMT-Bench dataset from TSV with base64 images."""

    def __init__(self, data_dir=None, split="VAL"):
        data_dir = data_dir or os.path.join(DATA_DIR, "mmt_bench")
        tsv_path = os.path.join(data_dir, f"MMT-Bench_{split}.tsv")
        print(f"Loading MMT-Bench {split} from {tsv_path}...")
        self.df = pd.read_csv(tsv_path, sep="\t")
        print(f"Loaded {len(self.df)} samples, {self.df['l2-category'].nunique()} subtasks")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        options = []
        for letter in "ABCDEFGHI":
            if letter in row and pd.notna(row[letter]) and str(row[letter]).strip():
                options.append(str(row[letter]))
        image = decode_base64_image(row.get("image", ""))
        return {
            "index": int(row["index"]) if "index" in row else idx,
            "question": str(row.get("question", "")),
            "options": options,
            "answer": str(row.get("answer", "")),
            "image": image,
            "category": str(row.get("category", "")),
            "l2-category": str(row.get("l2-category", "")),
        }


def load_mmt_bench(data_dir=None, split="VAL"):
    """Load MMT-Bench. Returns an iterable with __len__ and __getitem__."""
    return MMTBenchDataset(data_dir=data_dir, split=split)


def load_mmmu(data_dir=None, split="dev_val"):
    """
    Load MMMU from local Arrow files.

    Splits:
      - "dev_val": dev + validation = 1050 samples (matches MMMU_DEV_VAL leaderboard)
      - "dev": 150 samples
      - "validation": 900 samples
      - "test": 10500 samples
    """
    data_dir = data_dir or os.path.join(DATA_DIR, "mmmu")
    if split == "dev_val":
        dev = load_from_disk(os.path.join(data_dir, "dev"))
        val = load_from_disk(os.path.join(data_dir, "validation"))
        dataset = concatenate_datasets([dev, val])
    else:
        dataset = load_from_disk(os.path.join(data_dir, split))
    print(f"Loaded MMMU {split}: {len(dataset)} samples")
    return dataset


def load_scienceqa(data_dir=None, split="test", images_only=False):
    """
    Load ScienceQA from local Arrow files.
    Converts integer answer (0,1,2,...) to letter (A,B,C,...).
    """
    data_dir = data_dir or os.path.join(DATA_DIR, "scienceqa")
    dataset = load_from_disk(os.path.join(data_dir, split))

    def convert_answer(example):
        example["answer"] = chr(ord("A") + example["answer"])
        return example

    dataset = dataset.map(convert_answer)

    if images_only:
        dataset = dataset.filter(lambda x: x["image"] is not None)
        print(f"Loaded ScienceQA {split} (images only): {len(dataset)} samples")
    else:
        print(f"Loaded ScienceQA {split}: {len(dataset)} samples")
    return dataset


class MMEDataset:
    """MME benchmark dataset loader.

    Handles two folder formats:
      - Format A: subtask/images/ + subtask/questions_answers_YN/  (artwork, celebrity, landmark, posters, scene)
      - Format B: subtask/NNNN.jpg + subtask/NNNN.txt  (flat pairs)

    Each text file has 2 lines (yes question + no question), tab-separated: question\tanswer
    """

    PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "OCR", "artwork"]
    COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]

    def __init__(self, data_dir=None):
        data_dir = data_dir or os.path.join(DATA_DIR, "mme")
        # Find the benchmark root (may be nested)
        bench_dir = os.path.join(data_dir, "MME_Benchmark_release_version", "MME_Benchmark")
        if not os.path.isdir(bench_dir):
            bench_dir = data_dir
        self.bench_dir = bench_dir
        self.samples = []
        self._load()
        print(f"Loaded MME: {len(self.samples)} questions across {len(set(s['category'] for s in self.samples))} subtasks")

    def _load(self):
        all_subtasks = self.PERCEPTION + self.COGNITION
        for subtask in all_subtasks:
            subtask_dir = os.path.join(self.bench_dir, subtask)
            if not os.path.isdir(subtask_dir):
                continue

            images_dir = os.path.join(subtask_dir, "images")
            qa_dir = os.path.join(subtask_dir, "questions_answers_YN")

            if os.path.isdir(images_dir) and os.path.isdir(qa_dir):
                # Format A: images/ + questions_answers_YN/
                for txt_file in sorted(os.listdir(qa_dir)):
                    if not txt_file.endswith(".txt"):
                        continue
                    stem = txt_file[:-4]
                    # Find matching image
                    img_path = None
                    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                        candidate = os.path.join(images_dir, stem + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    if img_path is None:
                        continue
                    # Read questions
                    with open(os.path.join(qa_dir, txt_file), "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) != 2:
                                continue
                            question, answer = parts
                            self.samples.append({
                                "question": question.strip(),
                                "answer": answer.strip(),
                                "image_path": img_path,
                                "category": subtask,
                            })
            else:
                # Format B: flat img+txt pairs
                txt_files = sorted([f for f in os.listdir(subtask_dir) if f.endswith(".txt")])
                for txt_file in txt_files:
                    stem = txt_file[:-4]
                    img_path = None
                    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                        candidate = os.path.join(subtask_dir, stem + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    if img_path is None:
                        continue
                    with open(os.path.join(subtask_dir, txt_file), "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) != 2:
                                continue
                            question, answer = parts
                            self.samples.append({
                                "question": question.strip(),
                                "answer": answer.strip(),
                                "image_path": img_path,
                                "category": subtask,
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "question": sample["question"],
            "answer": sample["answer"],
            "image": image,
            "category": sample["category"],
        }


def load_mme(data_dir=None):
    """Load MME benchmark. Returns an iterable with __len__ and __getitem__."""
    return MMEDataset(data_dir=data_dir)


class AI2DDataset:
    """AI2D_TEST dataset (VLMEvalKit-processed with masked diagram text).

    Loads from ai2d_test_vlmevalkit.jsonl + AI2D_TEST/ images.
    Question field already contains options and prompt suffix.
    """

    def __init__(self, data_dir=None):
        import json
        data_dir = data_dir or os.path.join(DATA_DIR, "ai2d")
        jsonl_path = os.path.join(data_dir, "ai2d_test_vlmevalkit.jsonl")
        self.image_dir = os.path.join(data_dir, "AI2D_TEST")
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f]
        cats = len(set(s["category"] for s in self.samples))
        print(f"Loaded AI2D_TEST: {len(self.samples)} samples, {cats} categories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_name = os.path.basename(s["image"])
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return {
            "question": s["question"],
            "answer": s["answer"],
            "image": image,
            "category": s["category"],
        }


def load_ai2d(data_dir=None, split="test"):
    """Load AI2D_TEST (VLMEvalKit-processed, masked diagrams)."""
    return AI2DDataset(data_dir=data_dir)


def load_mathvista(data_dir=None, split="testmini"):
    """
    Load MathVista from local Arrow files.

    Converts MCQ answer text to letter (A/B/C/D).
    Extracts task type from metadata into top-level 'task' field.
    Maps question_type: 'free_form' -> 'open', 'multi_choice' -> 'multi_choice'.
    Uses 'query' field as the pre-formatted prompt (includes hint, question, choices).
    """
    data_dir = data_dir or os.path.join(DATA_DIR, "mathvista")
    dataset = load_from_disk(os.path.join(data_dir, split))

    def process_sample(example):
        # Extract task from metadata
        example["task"] = example["metadata"]["task"]

        # Convert MCQ answer from text to letter
        if example["question_type"] == "multi_choice" and example["choices"]:
            choices = example["choices"]
            answer_text = example["answer"]
            try:
                idx = choices.index(answer_text)
                example["answer"] = chr(ord("A") + idx)
            except ValueError:
                pass  # keep original if not found

        # Map question_type for our eval pipeline
        if example["question_type"] == "free_form":
            example["question_type"] = "open"

        return example

    dataset = dataset.map(process_sample)
    print(f"Loaded MathVista {split}: {len(dataset)} samples")
    return dataset


class MMBenchDataset:
    """Lazy-loading MMBench dataset from TSV with base64 images.

    Very similar to MMT-Bench. Has a 'hint' field for context.
    Some samples have empty C/D options (2-3 way MCQ).
    """

    def __init__(self, data_dir=None, split="DEV_EN"):
        data_dir = data_dir or os.path.join(DATA_DIR, "mmbench")
        tsv_path = os.path.join(data_dir, f"MMBench_{split}.tsv")
        print(f"Loading MMBench {split} from {tsv_path}...")
        self.df = pd.read_csv(tsv_path, sep="\t")
        # Build index->base64 map: some rows have base64 blobs, others have
        # numeric IDs that reference those rows
        self._img_b64 = {}
        for _, row in self.df.iterrows():
            img = str(row.get("image", ""))
            if len(img) > 100:  # base64 blob
                self._img_b64[str(int(row["index"]))] = img
        print(f"Loaded {len(self.df)} samples, {self.df['l2-category'].nunique()} subtasks, {len(self._img_b64)} unique images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        options = []
        for letter in "ABCD":
            if letter in row and pd.notna(row[letter]) and str(row[letter]).strip():
                options.append(str(row[letter]))
        # Resolve image: either base64 blob or numeric ID referencing another row
        img_val = str(row.get("image", ""))
        if img_val.isdigit() and img_val in self._img_b64:
            image = decode_base64_image(self._img_b64[img_val])
        else:
            image = decode_base64_image(img_val)
        hint = str(row.get("hint", "")) if pd.notna(row.get("hint", "")) else ""
        return {
            "index": int(row["index"]) if "index" in row else idx,
            "question": str(row.get("question", "")),
            "hint": hint,
            "options": options,
            "answer": str(row.get("answer", "")),
            "image": image,
            "category": str(row.get("category", "")),
            "l2-category": str(row.get("l2-category", "")),
        }


def load_mmbench(data_dir=None, split="DEV_EN"):
    """Load MMBench. Returns an iterable with __len__ and __getitem__."""
    return MMBenchDataset(data_dir=data_dir, split=split)


def load_blink(data_dir=None, split="val"):
    """
    Load BLINK from local Arrow files.

    14 subtasks (Art Style, Counting, etc.), 1901 val samples.
    Multi-image (1-4 images per sample), pre-formatted prompts.
    Normalizes answer from '(A)' or 'A' to just 'A'.
    """
    data_dir = data_dir or os.path.join(DATA_DIR, "blink")
    dataset = load_from_disk(os.path.join(data_dir, split))

    def normalize_answer(example):
        ans = example["answer"].strip()
        # Strip parentheses: (A) -> A
        if ans.startswith("(") and ans.endswith(")"):
            ans = ans[1:-1]
        example["answer"] = ans
        return example

    dataset = dataset.map(normalize_answer)
    print(f"Loaded BLINK {split}: {len(dataset)} samples")
    return dataset


def load_dataset_by_name(name, data_dir=None, split=None, images_only=False):
    """Dispatch to the right loader based on dataset name."""
    defaults = {"mmt_bench": "VAL", "mmmu": "dev_val", "scienceqa": "test", "mme": "test", "ai2d": "test", "mathvista": "testmini", "mmbench": "DEV_EN", "blink": "val"}
    split = split or defaults.get(name, "test")

    if name == "mmt_bench":
        return load_mmt_bench(data_dir=data_dir, split=split)
    elif name == "mmmu":
        return load_mmmu(data_dir=data_dir, split=split)
    elif name == "scienceqa":
        return load_scienceqa(data_dir=data_dir, split=split, images_only=images_only)
    elif name == "mme":
        return load_mme(data_dir=data_dir)
    elif name == "ai2d":
        return load_ai2d(data_dir=data_dir, split=split)
    elif name == "mathvista":
        return load_mathvista(data_dir=data_dir, split=split)
    elif name == "mmbench":
        return load_mmbench(data_dir=data_dir, split=split)
    elif name == "blink":
        return load_blink(data_dir=data_dir, split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")
