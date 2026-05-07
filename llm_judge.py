"""
LLM judge for evaluating open-ended and ambiguous VLM responses.

Adds a 'judge_correct' field to JSONL result files. Only judges samples where
simple matching is uncertain (open-ended questions, non-letter MCQ responses).

Usage:
    # Judge a single file
    python llm_judge.py output/Qwen3-VL-8B-Instruct/mmmu/dev_val_direct.jsonl

    # Judge all MMMU results
    python llm_judge.py --pattern "output/*/mmmu/*_direct.jsonl"

    # Dry run (show what would be judged, don't call API)
    python llm_judge.py --dry-run output/Qwen3-VL-8B-Instruct/mmmu/dev_val_direct.jsonl

    # Use a specific model
    python llm_judge.py --model gpt-4o-mini output/...
"""
import json
import os
import re
import sys
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluate import extract_answer_letter, extract_numbers


JUDGE_PROMPT = """You are an answer-checking judge. Given a question, the ground truth answer, and a model's response, determine if the model's response is correct.

Question: {question}
Ground truth answer: {answer}
Model response: {response}

Rules:
- For numeric answers: accept equivalent forms (e.g., 120 = 120.0 = 1.2e2), and values within 5% relative tolerance.
- For text answers: accept equivalent meaning, abbreviations, or minor formatting differences.
- For MCQ: if the model selected a letter (A/B/C/D) and it matches the correct option, it's correct.
- Ignore extra explanation — only judge whether the core answer is correct.

Reply with ONLY one word: "correct" or "incorrect"."""


def _is_open_ended(r):
    """Check if a sample is an open-ended (non-MCQ) question."""
    if r.get("question_type") == "open":
        return True
    ans = r.get("answer", "").strip()
    if ans and len(ans) > 1 and ans[0].upper() not in "ABCDEFGHI":
        return True
    return False


def _extract_last_number(text):
    """Extract the last number from a response (models often put answer at end)."""
    # Try boxed answer first: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        try:
            return float(boxed[-1].replace(",", "").strip())
        except ValueError:
            pass
    # Try "Final Answer: X" or "answer is X" patterns
    final_patterns = [
        r'[Ff]inal [Aa]nswer[:\s]*([+-]?\d+(?:\.\d+)?)',
        r'[Aa]nswer[:\s]*([+-]?\d+(?:\.\d+)?)\s*$',
        r'[Tt]herefore[,\s]+.*?([+-]?\d+(?:\.\d+)?)\s*$',
        r'=\s*([+-]?\d+(?:\.\d+)?)\s*$',
    ]
    for pat in final_patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    # Fall back to last number in the text
    nums = extract_numbers(text)
    if nums:
        try:
            return float(nums[-1].replace(",", "") if isinstance(nums[-1], str) else nums[-1])
        except (ValueError, TypeError):
            pass
    return None


def _simple_match(resp, ans, record):
    """Quick deterministic match. Returns True/False/None (None = uncertain)."""
    resp = resp.strip()
    ans = ans.strip()
    if not resp:
        return False

    # MCQ: extract letter using robust extraction
    if not _is_open_ended(record):
        letter = extract_answer_letter(resp)
        if letter:
            return letter == ans.upper()
        # Fallback: first character
        if resp[0].upper() in "ABCDEFGHI":
            return resp[0].upper() == ans.upper()
        # Model didn't give a letter for MCQ — needs judge
        return None

    # Open-ended: exact match
    resp_clean = resp.rstrip(".")
    if resp_clean.lower() == ans.lower():
        return True

    # Single-letter answer in open question
    if len(ans) == 1 and ans.upper() in "ABCDEFGHI":
        letter = extract_answer_letter(resp)
        if letter:
            return letter == ans.upper()
        if resp[0].upper() in "ABCDEFGHI":
            return resp[0].upper() == ans.upper()

    # Numeric match — try parsing full response first, then extract last number
    try:
        a_val = float(ans.replace(",", ""))
    except ValueError:
        # Non-numeric answer — needs judge for text matching
        return None

    # Try parsing full response as number
    r_val = None
    try:
        r_val = float(resp_clean.replace(",", ""))
    except ValueError:
        # Extract last number from long response
        r_val = _extract_last_number(resp)

    if r_val is not None:
        if a_val == 0:
            if abs(r_val) < 0.01:
                return True
            return False
        if abs(r_val - a_val) / max(abs(a_val), 1e-9) < 0.05:
            return True
        # Extracted a number but it doesn't match — likely wrong
        # Still send to judge for borderline cases
        return None

    # Model gave a letter for an open-ended question (confused by MCQ prompt)
    if len(resp) <= 2 and resp[0].upper() in "ABCDEFGHI":
        return False  # Hallucinated letter, definitely wrong

    # Can't determine — needs judge
    return None


def call_judge(client, model, question, answer, response):
    """Call LLM judge API."""
    prompt = JUDGE_PROMPT.format(
        question=question[:2000],
        answer=answer,
        response=response[:1000],
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0,
        )
        verdict = completion.choices[0].message.content.strip().lower()
        if verdict == "correct":
            return True
        if verdict == "incorrect":
            return False
        return None
    except Exception as e:
        print(f"  Judge API error: {e}")
        return None


def process_file(filepath, client, model, dry_run=False, force=False):
    """Process a JSONL file, adding judge_correct field where needed."""
    records = []
    needs_judge = []

    with open(filepath) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            records.append(r)

            # Skip if already judged (unless force)
            if "judge_correct" in r and not force:
                continue

            resp = r.get("response", "").strip()
            ans = r.get("answer", "").strip()

            simple = _simple_match(resp, ans, r)
            if simple is not None:
                r["judge_correct"] = simple
            else:
                needs_judge.append((i, r))

    if not needs_judge:
        print(f"  {filepath}: {len(records)} records, 0 need judging")
        return 0

    print(f"  {filepath}: {len(records)} records, {len(needs_judge)} need judging")

    if dry_run:
        for idx, r in needs_judge[:5]:
            print(f"    [{idx}] ans={repr(r.get('answer','')[:30])} resp={repr(r.get('response','')[:50])}")
        if len(needs_judge) > 5:
            print(f"    ... and {len(needs_judge) - 5} more")
        return len(needs_judge)

    # Call judge API with rate limiting
    judged = 0
    for idx, r in needs_judge:
        question = r.get("question", "")
        answer = r.get("answer", "")
        response = r.get("response", "")

        result = call_judge(client, model, question, answer, response)
        if result is not None:
            r["judge_correct"] = result
            judged += 1

        # Rate limit: ~50 req/s for gpt-4o-mini
        time.sleep(0.02)

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"    Judged {judged}/{len(needs_judge)} samples")
    return judged


def main():
    parser = argparse.ArgumentParser(description="LLM judge for open-ended VLM responses")
    parser.add_argument("files", nargs="*", help="JSONL files to judge")
    parser.add_argument("--pattern", type=str, help="Glob pattern for files")
    parser.add_argument("--model", type=str, default="gpt-5.1-2025-11-13", help="Judge model")
    parser.add_argument("--dry-run", action="store_true", help="Show what needs judging without calling API")
    parser.add_argument("--force", action="store_true", help="Re-judge even if judge_correct exists")
    args = parser.parse_args()

    # Collect files
    files = list(args.files) if args.files else []
    if args.pattern:
        files.extend(glob.glob(args.pattern, recursive=True))
    if not files:
        parser.error("No files specified. Use positional args or --pattern")

    files = sorted(set(files))
    print(f"Processing {len(files)} files with judge model: {args.model}")

    # Init OpenAI client
    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI()

    total_judged = 0
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"  SKIP (not found): {filepath}")
            continue
        total_judged += process_file(filepath, client, args.model, args.dry_run, args.force)

    print(f"\nDone. Total judged: {total_judged}")


if __name__ == "__main__":
    main()
