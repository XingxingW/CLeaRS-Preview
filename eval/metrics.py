"""
CLeaRS Evaluation Metrics and Task Scorers (Pure Rule-based & Offline)
Author: Ruifeng Ni et al. (CLeaRS Benchmark)
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# CLeaRS subset to task mapping
CLEARS_SUBSETS: Dict[str, Dict[str, str]] = {
    'FireRisk': {'type': 'CLS', 'desc': 'Fire risk scene classification'},
    'AID': {'type': 'CLS', 'desc': 'Aerial scene classification'},
    'RescueNet': {'type': 'vqa', 'desc': 'Disaster rescue visual QA'},
    'IR-VQA': {'type': 'vqa', 'desc': 'Infrared imagery visual QA'},
    'SAR-VQA': {'type': 'vqa', 'desc': 'SAR imagery visual QA'},
    'VRS-VQA': {'type': 'vqa', 'desc': 'Optical remote sensing visual QA'},
    'VRS-Cap': {'type': 'caption', 'desc': 'Remote sensing image captioning'},
    'IR-VG': {'type': 'vg', 'desc': 'Infrared visual grounding'},
    'SAR-VG': {'type': 'vg', 'desc': 'SAR visual grounding'},
    'VRS-VG': {'type': 'vg', 'desc': 'Optical visual grounding'},
}

# Recommended system/prefix prompts for model inference
CLEARS_PROMPTS: Dict[str, str] = {
    'AID': 'Answer with exactly one word from the given choices, no extra text.Do not respond with numbers.',
    'FireRisk': 'Answer with exactly one word from the given choices, no extra text.Do not respond with numbers.',
    'RescueNet': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'IR-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'SAR-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'VRS-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'IR-VG': 'Please output unique bounding box and do not include any extra content.',
    'SAR-VG': 'Please output unique bounding box and do not include any extra content.',
    'VRS-VG': 'Please output unique bounding box and do not include any extra content.',
    'VRS-Cap': '',
}


def normalize_text(text: Any) -> str:
    return str(text).strip().lower()


def is_pure_number(text: str) -> bool:
    """Check if the string is purely numeric."""
    return re.fullmatch(r"\d+", text) is not None


def _check_synonyms(gt: str, pred: str) -> bool:
    """Rule-based synonym matching for remote sensing attributes."""
    synonym_groups = [
        {'circular', 'circle'},
        {'grey', 'gray'},
    ]
    for group in synonym_groups:
        if gt in group and pred in group:
            return True
    return False


def _check_antonyms(gt: str, pred: str) -> bool:
    """Rule-based antonym / distinct entity collision prevention."""
    antonym_groups = [
        {'freight car', 'car'},
        {'very_low', 'low'},
        {'undamaged', 'damaged'},
        {'high', 'very_high'},
    ]
    for group in antonym_groups:
        if gt in group and pred in group:
            return True
    return False


def parse_bbox(text: Any) -> Optional[List[float]]:
    """
    Parse bounding box from various model output formats:
    - [x1, y1, x2, y2]
    - {<x1><y1><x2><y2>}
    - comma/space separated coordinates
    Returns [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2.
    """
    if text is None or pd.isna(text):
        return None
    s = str(text).strip()
    if len(s) > 500:
        s = s[:500]

    candidates = []
    # 1. Bracket format: [x1, y1, x2, y2]
    candidates += re.findall(
        r'\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]',
        s
    )
    # 2. Angle brackets / curly braces: {<x1><y1><x2><y2>}
    candidates += re.findall(
        r'\{<[^<>]+><[^<>]+><[^<>]+><[^<>]+>(?:\|<[^<>]+>)*\}',
        s
    )

    for cand in candidates:
        nums = re.findall(r'-?\d+\.?\d*', cand)
        if len(nums) >= 4:
            try:
                x1, y1, x2, y2 = map(float, nums[:4])
                return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            except Exception:
                continue

    # Fallback: scan for any 4 sequential numbers in the string
    all_nums = re.findall(r'-?\d+\.?\d*', s)
    if len(all_nums) >= 4:
        try:
            x1, y1, x2, y2 = map(float, all_nums[:4])
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        except Exception:
            return None

    return None


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x1_g, y1_g, x2_g, y2_g = bbox2

    inter_x1 = max(x1, x1_g)
    inter_y1 = max(y1, y1_g)
    inter_x2 = min(x2, x2_g)
    inter_y2 = min(y2, y2_g)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    box1_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box2_area = max(0.0, x2_g - x1_g) * max(0.0, y2_g - y1_g)

    union_area = box1_area + box2_area - inter_area
    return (inter_area / union_area) if union_area > 0 else 0.0


def extract_mcq_answer_strict(text: Any) -> Optional[str]:
    """Strictly extract multiple-choice option letter (A-H)."""
    if text is None or pd.isna(text):
        return None
    s = str(text).strip()

    # 1. Beginning of string
    m = re.match(r'^\(?([A-H])\)?[.\s)]', s)
    if m:
        return m.group(1).upper()

    # 2. Choice indicators
    patterns = [
        r'(?:answer is|correct answer is|the answer is)\s*\(?([A-H])\)?',
        r'(?:choose|select|pick|option)\s*\(?([A-H])\)?',
        r'^\s*\(?([A-H])\)?\s*$',
    ]
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None


# -------------------------------------------------------------------------
# Task-specific Evaluators
# -------------------------------------------------------------------------

def evaluate_cls(subset_df: pd.DataFrame) -> Tuple[Dict[str, Any], List[int]]:
    """
    Evaluate Classification (CLS) task.
    Supports AID and FireRisk.
    Validates single category prediction and extracts candidate categories from question.
    """
    total = len(subset_df)
    per_sample: List[int] = []
    invalid = 0

    for _, row in tqdm(subset_df.iterrows(), total=total, desc="Evaluating CLS", leave=False):
        flag = 0
        try:
            predicted = str(row['prediction']).strip()
            ground_truth = str(row['answer']).strip()
            question = str(row.get('question', ''))

            pred_lower = predicted.lower()
            gt_lower = ground_truth.lower()

            if pred_lower == gt_lower:
                flag = 1
            elif gt_lower in pred_lower:
                # Extract candidate categories from question brackets [...]
                categories = []
                match = re.search(r"(\[.*?\])", question)
                if match:
                    try:
                        categories = ast.literal_eval(match.group(1))
                    except Exception:
                        pass
                if not categories:
                    match2 = re.search(r"\[(.*?)\]", question)
                    if match2:
                        categories = [c.strip().strip("'\"") for c in match2.group(1).split(',')]

                if not categories:
                    categories = [ground_truth]

                mentioned_categories = set()
                for cat in categories:
                    cat_lower = str(cat).lower()
                    pattern = r'\b' + re.escape(cat_lower) + r'\b'
                    if re.search(pattern, pred_lower):
                        mentioned_categories.add(cat_lower)

                # Exactly one category mentioned and matches ground truth
                if gt_lower in mentioned_categories and len(mentioned_categories) == 1:
                    flag = 1
                else:
                    flag = 0
            else:
                flag = 0
        except Exception:
            invalid += 1
            flag = 0

        per_sample.append(flag)

    correct = sum(per_sample)
    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    return {
        'accuracy': round(accuracy, 2),
        'correct': correct,
        'total': total,
        'invalid': invalid,
    }, per_sample


def evaluate_vqa(subset_df: pd.DataFrame) -> Tuple[Dict[str, Any], List[int]]:
    """
    Evaluate Visual Question Answering (VQA) task.
    Supports RescueNet, IR-VQA, SAR-VQA, VRS-VQA.
    100% deterministic, rule-based matching.
    """
    total = len(subset_df)
    per_sample: List[int] = []
    invalid = 0

    for _, row in tqdm(subset_df.iterrows(), total=total, desc="Evaluating VQA", leave=False):
        flag = 0
        try:
            predicted = normalize_text(row['prediction'])
            ground_truth = normalize_text(row['answer'])

            if predicted == ground_truth:
                flag = 1
            elif re.fullmatch(r"\d+\s*-\s*\d+|\d+\+", ground_truth):
                pred_nums = re.findall(r"\d+", predicted)
                if pred_nums:
                    pred_num = int(pred_nums[0])
                    if "+" in ground_truth:
                        low = int(ground_truth.replace("+", ""))
                        flag = 1 if pred_num >= low else 0
                    else:
                        low, high = map(int, ground_truth.split("-"))
                        flag = 1 if low <= pred_num <= high else 0
                else:
                    flag = 0
            elif is_pure_number(ground_truth):
                if is_pure_number(predicted):
                    flag = 1 if predicted == ground_truth else 0
                else:
                    pred_nums = re.findall(r"\d+", predicted)
                    flag = 1 if (pred_nums and pred_nums[0] == ground_truth) else 0
            elif ground_truth in ["yes", "no"] and predicted in ["yes", "no"]:
                flag = 1 if predicted == ground_truth else 0
            elif _check_antonyms(ground_truth, predicted):
                flag = 0
            elif _check_synonyms(ground_truth, predicted):
                flag = 1
            elif ground_truth in predicted or predicted in ground_truth:
                flag = 1
            else:
                flag = 0

        except Exception:
            invalid += 1
            flag = 0

        per_sample.append(flag)

    correct = sum(per_sample)
    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    return {
        'accuracy': round(accuracy, 2),
        'correct': correct,
        'total': total,
        'invalid': invalid,
    }, per_sample


def evaluate_vg(subset_df: pd.DataFrame) -> Tuple[Dict[str, Any], List[int]]:
    """
    Evaluate Visual Grounding (VG) task.
    Supports IR-VG, SAR-VG, VRS-VG.
    Calculates Acc@0.5 (IoU >= 0.5) and Mean IoU.
    """
    total = len(subset_df)
    iou_scores: List[float] = []
    acc_flags: List[int] = []
    per_sample: List[int] = []
    invalid_count = 0

    for _, row in tqdm(subset_df.iterrows(), total=total, desc="Evaluating VG", leave=False):
        flag = 0
        try:
            pred_bbox = parse_bbox(row['prediction'])
            true_bbox = parse_bbox(row['answer'])
            if pred_bbox and true_bbox:
                iou = calculate_iou(pred_bbox, true_bbox)
                iou_scores.append(iou)
                flag = 1 if iou >= 0.5 else 0
                acc_flags.append(flag)
            else:
                invalid_count += 1
                flag = 0
        except Exception:
            invalid_count += 1
            flag = 0

        per_sample.append(flag)

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    acc_05 = (float(np.mean(acc_flags)) * 100.0) if acc_flags else 0.0

    return {
        'acc_@0.5': round(acc_05, 2),
        'mean_iou': round(mean_iou, 4),
        'valid_samples': len(iou_scores),
        'invalid_samples': invalid_count,
        'total': total,
    }, per_sample


def evaluate_caption(subset_df: pd.DataFrame) -> Tuple[Dict[str, Any], List[float]]:
    """
    Evaluate Image Captioning task (VRS-Cap).
    Computes BLEU-1~4, ROUGE_L, CIDEr.
    Falls back gracefully if pycocoevalcap is missing.
    """
    total = len(subset_df)
    ref: Dict[str, List[str]] = {}
    gt: Dict[str, List[str]] = {}

    for i in range(total):
        line = subset_df.iloc[i]
        ref[str(i)] = [str(line['prediction']).strip()]
        ans = line['answer']
        if isinstance(ans, list):
            gt[str(i)] = [str(x).strip() for x in ans]
        else:
            gt[str(i)] = [str(ans).strip()]

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]

        scores_dict: Dict[str, Any] = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(gt, ref)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    scores_dict[m] = round(sc * 100.0, 2)
            else:
                scores_dict[method] = round(score * 100.0, 2)

        if 'CIDEr' in scores_dict:
            raw_cider = scores_dict['CIDEr']
            cider_upper_bound = 33.9
            normalized_cider = round((raw_cider / cider_upper_bound) * 100.0, 2)
            scores_dict['normalized_CIDEr'] = normalized_cider
            scores_dict['raw_CIDEr'] = raw_cider
            scores_dict['cider_normalization_note'] = (
                "For VRS-Cap, we normalize the CIDEr scores using the highest CIDEr value (33.9) reported on the original VRSBench-Cap as the upper bound"
                ", and then compute the continual learning metrics."
            )

        return scores_dict, []

    except ImportError:
        warning_msg = (
            "pycocoevalcap is not installed. Run `pip install pycocoevalcap` "
            "to compute full COCO Caption metrics (CIDEr, ROUGE_L, BLEU-1~4)."
        )
        print(f"[Warning] {warning_msg}")
        return {'warning': warning_msg, 'total': total}, []
    except Exception as e:
        print(f"[Error in caption evaluation] {e}")
        return {'error': str(e), 'total': total}, []


def evaluate_mcq(subset_df: pd.DataFrame) -> Tuple[Dict[str, Any], List[int]]:
    """Evaluate Multiple Choice (MCQ) task if present."""
    total = len(subset_df)
    correct = 0
    invalid = 0
    per_sample: List[int] = []

    for _, row in tqdm(subset_df.iterrows(), total=total, desc="Evaluating MCQ", leave=False):
        flag = 0
        try:
            pred_opt = extract_mcq_answer_strict(row['prediction'])
            gt_opt = str(row['answer']).strip().upper()
            if pred_opt is None:
                invalid += 1
            elif pred_opt == gt_opt:
                correct += 1
                flag = 1
        except Exception:
            invalid += 1

        per_sample.append(flag)

    acc = (correct / total * 100.0) if total > 0 else 0.0
    return {
        'accuracy': round(acc, 2),
        'correct': correct,
        'total': total,
        'invalid': invalid,
    }, per_sample


# -------------------------------------------------------------------------
# Master CLeaRS Benchmark Evaluator
# -------------------------------------------------------------------------

def evaluate_clears_df(
    df: pd.DataFrame
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Main evaluation pipeline for CLeaRS predictions.

    Args:
        df: DataFrame containing at least 'subset', 'prediction', and 'answer'.

    Returns:
        (summary_results_dict, annotated_dataframe)
    """
    # Validation
    required_cols = {'subset', 'prediction', 'answer'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    eval_df = df.copy()
    eval_df['eval_result'] = np.nan

    subsets_in_data = list(eval_df['subset'].unique())
    by_subset: Dict[str, Any] = {}

    for subset_name in subsets_in_data:
        subset_mask = eval_df['subset'] == subset_name
        subset_preds = eval_df[subset_mask].copy()

        # Determine task type
        task_type = 'unknown'
        if 'task_type' in subset_preds.columns and not pd.isna(subset_preds['task_type'].iloc[0]):
            task_type = str(subset_preds['task_type'].iloc[0])
        elif subset_name in CLEARS_SUBSETS:
            task_type = CLEARS_SUBSETS[subset_name]['type']

        task_type_lower = task_type.lower()
        res: Dict[str, Any] = {}
        per_sample: List[Any] = []

        if task_type_lower == 'cls':
            res, per_sample = evaluate_cls(subset_preds)
        elif task_type_lower == 'vqa':
            res, per_sample = evaluate_vqa(subset_preds)
        elif task_type_lower == 'vg':
            res, per_sample = evaluate_vg(subset_preds)
        elif task_type_lower == 'caption':
            res, per_sample = evaluate_caption(subset_preds)
        elif task_type_lower == 'mcq':
            res, per_sample = evaluate_mcq(subset_preds)
        else:
            print(f"[Warning] Unknown task type '{task_type}' for subset '{subset_name}'.")
            res = {'error': f"Unknown task type: {task_type}"}

        by_subset[subset_name] = {
            'task_type': task_type,
            'metrics': res,
            'samples': len(subset_preds),
        }

        if per_sample and len(per_sample) == len(subset_preds):
            eval_df.loc[subset_mask, 'eval_result'] = per_sample

    summary: Dict[str, Any] = {
        'benchmark': 'CLeaRS',
        'total_samples': len(eval_df),
        'subsets_evaluated': subsets_in_data,
        'by_subset': by_subset,
    }

    return summary, eval_df
