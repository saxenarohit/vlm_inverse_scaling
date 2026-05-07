"""
Evaluation utilities for VLM robustness benchmarking.

Supports multiple evaluation modes:
- Multiple choice (MMMU, AI2D, MathVista): Parse letter answer from response
- Exact match (OCRBench, ChartQA): String equality comparison
- Open-ended (DocVQA, TextVQA): Check if response contains any valid answer
"""
from collections import defaultdict
import re
import ast
import numpy as np
import os
import json
import sys


# =============================================================================
# Task-specific evaluation functions
# =============================================================================

def evaluate_multiple_choice(response: str, answer: str, options: list = None) -> bool:
    """Evaluate multiple choice response.

    Parses the response to extract an answer letter (A, B, C, etc.)
    and compares with the ground truth.
    """
    if options:
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except:
                options = None

        if options:
            index2ans, all_choices = get_multi_choice_info(options)
            parsed_pred = parse_multi_choice_response(response, all_choices, index2ans)
            return parsed_pred == answer

    # Fallback: direct letter comparison
    # Extract letter from response
    pred_letter = extract_answer_letter(response)
    return pred_letter == answer


def evaluate_exact_match(response: str, answer: str) -> bool:
    """Evaluate exact match (case-insensitive, whitespace-normalized)."""
    response_clean = response.strip().lower()
    answer_clean = str(answer).strip().lower()
    return response_clean == answer_clean


def evaluate_contains_answer(response: str, answers) -> bool:
    """Evaluate if response contains any of the valid answers.

    Used for open-ended VQA tasks where multiple answers are acceptable.
    """
    if isinstance(answers, str):
        try:
            answers = ast.literal_eval(answers)
        except:
            answers = [answers]

    if not isinstance(answers, (list, tuple)):
        answers = [answers]

    response_lower = response.strip().lower()
    for ans in answers:
        ans_str = str(ans).strip().lower()
        if ans_str in response_lower:
            return True
    return False


def extract_answer_letter(response: str) -> str:
    """Extract answer letter (A-J) from a model response."""
    # For long responses, check explicit answer declarations first (last match)
    if len(response) > 500:
        matches = re.findall(
            r'(?:[Ff]inal\s+)?(?:[Cc]orrect\s+)?(?:[Bb]est\s+)?(?:[Aa]nswer|[Oo]ption|[Cc]hoice)[:\s]+.*?(?:\(?|\\boxed\{|\$)([A-J])(?:\)?|\}|\$)',
            response
        )
        if matches:
            return matches[-1].upper()

    # Look for \boxed{X} (common in math model outputs)
    match = re.search(r'\\boxed\{([A-J])\}', response)
    if match:
        return match.group(1).upper()

    # Look for letter in parentheses: (A), (B), etc.
    match = re.search(r'\(([A-J])\)', response)
    if match:
        return match.group(1).upper()

    # Look for "Answer: X", "answer is X", "option is: A)" patterns

    match = re.search(r'(?:[Aa]nswer|[Oo]ption|[Cc]hoice)(?:\s+is)?[:\s]*(?:\(?|\\boxed\{|\$)([A-J])(?:\)?|\}|\$|[\.\s,;:!?*])', response)
    if match:
        return match.group(1).upper()

    # Look for letter at the START of response (e.g., "A. Something" or "A) Something")
    match = re.match(r'^([A-J])[\.\)\s]', response.strip())
    if match:
        return match.group(1).upper()

    # Look for markdown bold letter: **A** or *A*
    match = re.search(r'\*{1,2}([A-J])\*{1,2}\s*$', response.strip())
    if match:
        return match.group(1).upper()

    # Look for Chinese answer patterns: "故选：C" or "答案是 C" or "选C"
    match = re.search(r'(?:故选[：:]?\s*|答案[是为][：:\s]*)([A-J])', response)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end (allow trailing punctuation incl. CJK periods)
    match = re.search(r'\b([A-J])\b[.,;:!?\s*。．，、]*$', response.strip())
    if match:
        return match.group(1).upper()

    return ""


# =============================================================================
# Generic evaluation interface
# =============================================================================

def process_result_generic(result: dict, task_type: str = "multiple_choice",
                           answer_field: str = "answer",
                           options_field: str = "options") -> dict:
    """Process a single result with flexible field mapping.

    Args:
        result: Result dict with 'response' and ground truth fields
        task_type: One of 'multiple_choice', 'exact_match', 'open_ended'
        answer_field: Name of the ground truth answer field
        options_field: Name of the options field (for multiple choice)

    Returns:
        Result dict with 'parsed_pred' and 'if_right' fields added
    """
    response = result.get('response', '')
    if isinstance(response, dict):
        response = ''

    answer = result.get(answer_field, '')
    options = result.get(options_field) if options_field else None

    # Evaluate based on task type
    if task_type == "multiple_choice":
        is_correct = evaluate_multiple_choice(response, answer, options)
    elif task_type == "exact_match":
        is_correct = evaluate_exact_match(response, answer)
    elif task_type == "open_ended":
        is_correct = evaluate_contains_answer(response, answer)
    else:
        # Default to multiple choice
        is_correct = evaluate_multiple_choice(response, answer, options)

    result['parsed_pred'] = extract_answer_letter(response) if task_type == "multiple_choice" else response.strip()[:100]
    result['if_right'] = is_correct

    return result


def evaluate_results(results: list, task_type: str = "multiple_choice",
                    answer_field: str = "answer",
                    options_field: str = "options",
                    category_field: str = None) -> dict:
    """Evaluate a list of results.

    Args:
        results: List of result dicts
        task_type: Evaluation type
        answer_field: Ground truth field name
        options_field: Options field name (for MCQ)
        category_field: Field for grouping results (optional)

    Returns:
        Dict with 'overall' accuracy and per-category breakdown
    """
    processed = []
    correct = 0
    total = 0

    # Per-category tracking
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for result in results:
        proc = process_result_generic(result, task_type, answer_field, options_field)
        processed.append(proc)

        if proc['if_right']:
            correct += 1
        total += 1

        # Track by category if available
        if category_field and category_field in result:
            cat = result[category_field]
            if proc['if_right']:
                category_correct[cat] += 1
            category_total[cat] += 1

    overall_acc = (correct / total * 100) if total > 0 else 0

    eval_result = {
        'overall': {
            'accuracy': overall_acc,
            'correct': correct,
            'total': total,
        },
        'processed_results': processed,
    }

    # Add category breakdown
    if category_field:
        categories = {}
        for cat in category_total:
            cat_acc = (category_correct[cat] / category_total[cat] * 100) if category_total[cat] > 0 else 0
            categories[cat] = {
                'accuracy': cat_acc,
                'correct': category_correct[cat],
                'total': category_total[cat],
            }
        eval_result['categories'] = categories

    return eval_result


# =============================================================================
# MMMU-specific functions (legacy support)
# =============================================================================

def mmmu_process_results(results):
    pred = results['response']
    if isinstance(pred, dict):
        pred = ''

    index2ans, all_choices = get_multi_choice_info(ast.literal_eval(str(results["options"])))
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)

    id = results["id"]
    # Track unparsable responses separately
    if parsed_pred is None:
        if_right = False
        unparsable = True
    elif parsed_pred == results["answer"]:
        if_right = True
        unparsable = False
    else:
        if_right = False
        unparsable = False

    results['pred_indexs'] = parsed_pred
    results['if_right'] = if_right
    results['unparsable'] = unparsable

    return results


def extract_subset_name(input_string):
    # Define a regex pattern to match "validation_" at the beginning and "_<number>" at the end
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


def mmmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subject"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_mmmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 3),
        }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 3),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 3),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]


##################
# Helper functions written by official MMMU repo.
##################


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate_mmmu(samples):
    """
    Batch evaluation for multiple choice and open questions.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L219
    """
    pred_correct = 0
    unparsable_count = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answers"]
        pred_i = sample["parsed_pred"]

        # Track unparsable responses
        if pred_i is None:
            unparsable_count += 1
            judge_dict[sample["id"]] = "Unparsable"
            continue

        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {
        "acc": pred_correct / len(samples),
        "unparsable": unparsable_count,
        "unparsable_rate": unparsable_count / len(samples),
    }


def parse_multi_choice_responses(response):
    pred_indexs = []
    return pred_indexs

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index, e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        # Extract the string after "Answer:"
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        
        # Find a unique match in the options
        matching_options = [option for option in all_choices if option in answer_str]
        
        # If a unique match is found, return that option
        if len(matching_options) == 1:
            return matching_options[0]

        
    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        print (response)
        response = ""
    

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, return None (unparsable)
        return None
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbersz
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    """
    Check if the given string a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L122
    """

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def check_files(input_dir, expected_num=None, save_processed=False):
    """
    Evaluate JSONL result files in a directory.

    Args:
        input_dir: Directory containing JSONL result files
        expected_num: Expected number of samples per file (None to skip check)
        save_processed: If True, save processed results back to files (adds pred_indexs, if_right fields)
    """
    pattern = re.compile(r"(?P<model_name>.+)_(?P<setting>standard|vision)_(?P<method>cot|direct)(?:_(?P<aug>[\w\-]+))?\.jsonl")

    all_results = []

    for file_name in sorted(os.listdir(input_dir)):
        match = pattern.match(file_name)
        if match:
            model_name = match.group("model_name")
            method = match.group("method")
            setting = match.group("setting")
            aug = match.group("aug") or "none"
            file_path = os.path.join(input_dir, file_name)

            with open(file_path, 'r', encoding='utf-8') as infile:
                results = [json.loads(data) for data in infile]

            if expected_num is not None and len(results) != expected_num:
                print(f"Warning: {file_path} has {len(results)} results, expected {expected_num}")

            if len(results) == 0:
                print(f"Skipping empty file: {file_path}")
                continue

            processed_results = []
            true_nums = 0
            false_nums = 0

            for i, result in enumerate(results):
                new_data = mmmu_process_results(result)
                processed_results.append(new_data)

                if new_data['if_right']:
                    true_nums += 1
                else:
                    false_nums += 1

            # Calculate and output the accuracy
            total = true_nums + false_nums
            acc = true_nums / total * 100 if total > 0 else 0
            print(f"Model: {model_name:<20} Method: {method:<8} Setting: {setting:<8} Aug: {aug:<20} - Accuracy: {acc:>6.2f}% ({true_nums}/{total})")

            all_results.append({
                "model": model_name,
                "method": method,
                "setting": setting,
                "aug": aug,
                "accuracy": acc,
                "correct": true_nums,
                "total": total,
            })

            # Optionally write the processed results back to the file
            if save_processed:
                with open(file_path, 'w', encoding='utf-8') as outfile:
                    for item in processed_results:
                        outfile.write(json.dumps(item) + '\n')

    return all_results


def evaluate_single_file(file_path):
    """Evaluate a single JSONL file and print results."""
    with open(file_path, 'r', encoding='utf-8') as infile:
        results = [json.loads(data) for data in infile]

    if len(results) == 0:
        print(f"Empty file: {file_path}")
        return

    true_nums = 0
    false_nums = 0

    for result in results:
        new_data = mmmu_process_results(result)
        if new_data['if_right']:
            true_nums += 1
        else:
            false_nums += 1

    total = true_nums + false_nums
    acc = true_nums / total * 100 if total > 0 else 0
    print(f"File: {file_path}")
    print(f"Accuracy: {acc:.2f}% ({true_nums}/{total})")

    return {"accuracy": acc, "correct": true_nums, "total": total}


def evaluate_file_generic(file_path: str, task_type: str = "multiple_choice",
                          answer_field: str = "answer", options_field: str = "options",
                          category_field: str = None, save_processed: bool = False) -> dict:
    """Evaluate a single JSONL file with flexible field mapping.

    Args:
        file_path: Path to JSONL file
        task_type: 'multiple_choice', 'exact_match', or 'open_ended'
        answer_field: Ground truth field name
        options_field: Options field name (for MCQ, None to skip)
        category_field: Category field for grouping (optional)
        save_processed: If True, save processed results back to file

    Returns:
        Evaluation results dict
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f if line.strip()]

    if len(results) == 0:
        print(f"Empty file: {file_path}")
        return None

    eval_result = evaluate_results(
        results,
        task_type=task_type,
        answer_field=answer_field,
        options_field=options_field,
        category_field=category_field,
    )

    overall = eval_result['overall']
    print(f"File: {file_path}")
    print(f"Accuracy: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")

    if 'categories' in eval_result and eval_result['categories']:
        print("Per-category breakdown:")
        for cat, stats in sorted(eval_result['categories'].items()):
            print(f"  {cat}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

    if save_processed:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in eval_result['processed_results']:
                f.write(json.dumps(item) + '\n')
        print(f"Saved processed results to {file_path}")

    return eval_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate VLM inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MMMU Pro results (legacy mode)
  python evaluate.py --input_dir output/MMMU_Pro

  # Evaluate DocVQA results (open-ended)
  python evaluate.py --input_file results.jsonl --task_type open_ended --answer_field answers

  # Evaluate ChartQA results (exact match)
  python evaluate.py --input_file results.jsonl --task_type exact_match --answer_field label

  # Evaluate with category breakdown
  python evaluate.py --input_file results.jsonl --category_field subject
        """
    )

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing JSONL result files")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Single JSONL file to evaluate")
    parser.add_argument("--expected_num", type=int, default=None,
                        help="Expected number of samples per file (optional)")
    parser.add_argument("--save_processed", action="store_true",
                        help="Save processed results back to files (adds evaluation fields)")

    # Generic evaluation options
    parser.add_argument("--task_type", type=str, default="multiple_choice",
                        choices=["multiple_choice", "exact_match", "open_ended"],
                        help="Evaluation task type")
    parser.add_argument("--answer_field", type=str, default="answer",
                        help="Field name for ground truth answer")
    parser.add_argument("--options_field", type=str, default="options",
                        help="Field name for MCQ options (use 'none' to disable)")
    parser.add_argument("--category_field", type=str, default=None,
                        help="Field name for category grouping (e.g., 'subject', 'category')")

    # Output options
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save evaluation summary to JSON file")

    args = parser.parse_args()

    # Handle 'none' as None for options_field
    if args.options_field == 'none':
        args.options_field = None

    if args.input_file:
        # Single file evaluation with generic options
        result = evaluate_file_generic(
            args.input_file,
            task_type=args.task_type,
            answer_field=args.answer_field,
            options_field=args.options_field,
            category_field=args.category_field,
            save_processed=args.save_processed,
        )
        if args.output_json and result:
            # Remove processed_results for summary output
            summary = {k: v for k, v in result.items() if k != 'processed_results'}
            with open(args.output_json, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to {args.output_json}")

    elif args.input_dir:
        # Directory evaluation (legacy MMMU mode)
        all_results = check_files(args.input_dir, args.expected_num, args.save_processed)
        if args.output_json and all_results:
            with open(args.output_json, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved summary to {args.output_json}")

    else:
        # Default behavior for backwards compatibility
        default_dir = "./output/MMMU_Pro"
        if os.path.exists(default_dir):
            check_files(default_dir)
        else:
            parser.print_help()
            print(f"\nError: No input specified and default directory '{default_dir}' not found.")
            sys.exit(1)
