#!/usr/bin/env python3
"""
CLeaRS Evaluation Tool (Pure Rule-based & Offline)
Benchmark: Continual Vision-Language Learning for Remote Sensing (CLeaRS)

Usage:
    python evaluate.py --pred path/to/predictions.xlsx
    python evaluate.py --pred path/to/predictions.jsonl --output-dir results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add current folder to sys.path to ensure metrics module can be imported
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from metrics import CLEARS_SUBSETS, evaluate_clears_df


def load_predictions(file_path: Path) -> pd.DataFrame:
    """Load model predictions from various tabular / json file formats."""
    suffix = file_path.suffix.lower()
    if suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif suffix == '.csv':
        df = pd.read_csv(file_path)
    elif suffix in ['.tsv', '.tab']:
        df = pd.read_csv(file_path, sep='\t')
    elif suffix == '.jsonl':
        df = pd.read_json(file_path, lines=True)
    elif suffix == '.json':
        try:
            df = pd.read_json(file_path)
        except Exception:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported extensions: .xlsx, .xls, .csv, .tsv, .json, .jsonl"
        )
    return df


def print_results_table(summary: Dict[str, Any]) -> None:
    """Print beautifully formatted evaluation results to console."""
    print("\n" + "=" * 80)
    print("                     CLeaRS EVALUATION RESULTS")
    print("=" * 80)

    # Detailed Per-Subset Breakdown
    by_subset = summary.get('by_subset', {})
    print(f"\n[Subset-Level Breakdown ({len(by_subset)} subsets)]")
    print("-" * 80)
    print(f"{'Subset':<15} | {'Task':<8} | {'Samples':<8} | {'Primary Metrics'}")
    print("-" * 80)

    for subset_name, info in by_subset.items():
        task = info.get('task_type', '')
        samples = info.get('samples', 0)
        metrics = info.get('metrics', {})

        metric_strs = []
        if 'accuracy' in metrics:
            metric_strs.append(f"Acc: {metrics['accuracy']}% ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
        if 'acc_@0.5' in metrics:
            metric_strs.append(f"Acc@0.5: {metrics['acc_@0.5']}%")
        if 'mean_iou' in metrics:
            metric_strs.append(f"mIoU: {metrics['mean_iou']}")
        if 'normalized_CIDEr' in metrics:
            metric_strs.append(f"normalized_CIDEr: {metrics['normalized_CIDEr']}% (raw_CIDEr: {metrics.get('raw_CIDEr', '')})")
        elif 'CIDEr' in metrics:
            metric_strs.append(f"CIDEr: {metrics['CIDEr']}")
        if 'Bleu_4' in metrics:
            metric_strs.append(f"B-4: {metrics['Bleu_4']}")
        if 'ROUGE_L' in metrics:
            metric_strs.append(f"R-L: {metrics['ROUGE_L']}")
        if 'warning' in metrics:
            metric_strs.append("[Warning: missing pycocoevalcap]")
        if 'error' in metrics:
            metric_strs.append(f"[Error: {metrics['error']}]")

        metric_display = ", ".join(metric_strs) if metric_strs else json.dumps(metrics)
        print(f"{subset_name:<15} | {task:<8} | {samples:<8} | {metric_display}")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Evaluator for CLeaRS (Continual Vision-Language Learning for Remote Sensing)"
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to prediction file (.xlsx, .csv, .json, .jsonl, .tsv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results (default: 'eval_results')"
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default=None,
        help="Custom filename for score JSON (default: <pred_stem>_clears_score.json)"
    )
    parser.add_argument(
        "--no-save-annotated",
        action="store_true",
        help="Do not save the annotated prediction file with 'eval_result' column"
    )

    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        print(f"Error: Prediction file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from: {pred_path} ...")
    df = load_predictions(pred_path)
    print(f"Loaded {len(df)} samples.")
    print(f"Detected columns: {list(df.columns)}")

    # Run CLeaRS Evaluation
    summary, annotated_df = evaluate_clears_df(df)

    # Print summary table
    print_results_table(summary)

    # 1. Save JSON Scores
    stem = pred_path.stem
    if stem.endswith('_with_eval_result'):
        stem = stem[:-len('_with_eval_result')]
    json_filename = args.json_name or f"{stem}_clears_score.json"
    json_path = output_dir / json_filename

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ CLeaRS score saved to: {json_path}")

    # 2. Save Annotated Predictions (with eval_result column)
    if not args.no_save_annotated:
        suffix = pred_path.suffix.lower()
        annotated_filename = f"{stem}_with_eval_result{suffix}"
        annotated_path = output_dir / annotated_filename

        if suffix in ['.xlsx', '.xls']:
            annotated_df.to_excel(annotated_path, index=False)
        elif suffix == '.csv':
            annotated_df.to_csv(annotated_path, index=False)
        elif suffix in ['.tsv', '.tab']:
            annotated_df.to_csv(annotated_path, sep='\t', index=False)
        elif suffix == '.jsonl':
            annotated_df.to_json(annotated_path, orient='records', lines=True, force_ascii=False)
        else:
            annotated_df.to_json(annotated_path, orient='records', indent=2, force_ascii=False)

        print(f"✓ Annotated predictions saved to: {annotated_path}")


if __name__ == "__main__":
    main()
