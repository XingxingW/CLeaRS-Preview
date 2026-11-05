# CLeaRS Evaluation Toolkit

English | [简体中文](quickstart_zh.md)

This directory provides the standalone evaluation toolkit for **CLeaRS (Continual Vision-Language Learning for Remote Sensing)**.


## 1. Overview

CLeaRS covers diverse remote sensing sensor modalities (optical, infrared, SAR) across **10 evaluation subsets** grouped into 4 task types:

| Task Type | Subset | Modality | Primary Metric |
| :--- | :--- | :--- | :--- |
| **Classification (CLS)** | `AID` | Optical aerial | Accuracy (%) |
| **Classification (CLS)** | `FireRisk` | Multispectral | Accuracy (%) |
| **Visual QA (VQA)** | `RescueNet` | Post-disaster optical | Accuracy (%) |
| **Visual QA (VQA)** | `IR-VQA` | Thermal / IR drone | Accuracy (%) |
| **Visual QA (VQA)** | `SAR-VQA` | Synthetic Aperture Radar (SAR) | Accuracy (%) |
| **Visual QA (VQA)** | `VRS-VQA` | Optical satellite | Accuracy (%) |
| **Visual Grounding (VG)** | `IR-VG` | Thermal / IR drone | Acc@0.5 (%) |
| **Visual Grounding (VG)** | `SAR-VG` | Synthetic Aperture Radar (SAR) | Acc@0.5 (%) |
| **Visual Grounding (VG)** | `VRS-VG` | Optical satellite | Acc@0.5 (%) |
| **Image Captioning** | `VRS-Cap` | Optical satellite | normalized_CIDEr |

---

## 2. Inference Prompt Templates

To ensure a fair comparison across different models, we recommend prepending the corresponding task prompt prefix to `question` during model inference to standardize the output format:

```python
CLEARS_PROMPTS = {
    # Classification (CLS)
    'AID': 'Answer with exactly one word from the given choices, no extra text.Do not respond with numbers.',
    'FireRisk': 'Answer with exactly one word from the given choices, no extra text.Do not respond with numbers.',

    # Visual Question Answering (VQA)
    'RescueNet': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'IR-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'SAR-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'VRS-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',

    # Visual Grounding (VG)
    'IR-VG': 'Please output unique bounding box and do not include any extra content.',
    'SAR-VG': 'Please output unique bounding box and do not include any extra content.',
    'VRS-VG': 'Please output unique bounding box and do not include any extra content.',

    # Image Captioning
    'VRS-Cap': '',  # Use original question directly without extra prefix
}

# Inference prompt formatting
prefix = CLEARS_PROMPTS.get(subset_name, '')
full_prompt = f"{prefix}{question}" if prefix else question
```

---

## 3. Requirements

```bash
cd eval
pip install -r requirements.txt
```

---

## 4. Quick Start

### 4.1 Running Evaluation
Save your model's prediction outputs as an Excel (`.xlsx`), CSV, TSV, or JSONL file, and run:

```bash
# Evaluate prediction file (results are saved to eval_results/ by default)
python evaluate.py --pred /path/to/predictions.xlsx

# Or specify a custom output directory
python evaluate.py --pred /path/to/predictions.jsonl --output-dir custom_results/
```

### 4.2 Verifying with Built-in Example
We provide an example prediction file [example_predictions.jsonl](example_predictions.jsonl) containing samples from all 10 subsets:

```bash
python evaluate.py --pred example_predictions.jsonl
```

The console will display the formatted evaluation table:

```text
================================================================================
                     CLeaRS EVALUATION RESULTS
================================================================================

[Subset-Level Breakdown (10 subsets)]
--------------------------------------------------------------------------------
Subset          | Task     | Samples  | Primary Metrics
--------------------------------------------------------------------------------
AID             | CLS      | 2        | Acc: 100.0% (2/2)
FireRisk        | CLS      | 2        | Acc: 100.0% (2/2)
RescueNet       | vqa      | 2        | Acc: 100.0% (2/2)
IR-VQA          | vqa      | 1        | Acc: 100.0% (1/1)
SAR-VQA         | vqa      | 1        | Acc: 100.0% (1/1)
VRS-VQA         | vqa      | 1        | Acc: 100.0% (1/1)
VRS-Cap         | caption  | 1        | normalized_CIDEr: ...
IR-VG           | vg       | 1        | Acc@0.5: 100.0%, mIoU: 0.8931
SAR-VG          | vg       | 1        | Acc@0.5: 100.0%, mIoU: 0.8352
VRS-VG          | vg       | 1        | Acc@0.5: 100.0%, mIoU: 0.9258
--------------------------------------------------------------------------------
```

---

## 5. Input Format

Supported formats: `.xlsx`, `.xls`, `.csv`, `.tsv`, `.jsonl`, `.json`.

Each row represents one evaluation sample and must contain the following fields:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `subset` | string | **[Required]** Name of the evaluation subset | `AID`, `RescueNet`, `VRS-VG`, etc. |
| `answer` | string | **[Required]** Ground truth answer | `"airport"`, `"yes"`, `"[120, 85, 230, 195]"` |
| `prediction` | string | **[Required]** Model-generated prediction text | `"airport"`, `"[122, 88, 228, 192]"` |
| `question` | string | **[Required]** Question content (used for CLS candidate parsing) | `"Answer with ... ['airport', 'bridge']"` |
| `index` | string | *[Optional]* Unique sample ID | `"AID_0"` |
| `task_type` | string | *[Optional]* Task type (auto-inferred from `subset` if omitted) | `"CLS"`, `"vqa"`, `"vg"`, `"caption"` |

**JSONL Example**:
```json
{"index": "AID_0", "subset": "AID", "question": "Answer with ... ['airport', 'bridge']", "answer": "airport", "prediction": "airport"}
{"index": "IR-VG_0", "subset": "IR-VG", "question": "Detect the vehicle.", "answer": "[120, 85, 230, 195]", "prediction": "[122, 88, 228, 192]"}
```

---

## 6. Outputs

After evaluation finishes, results are saved to `eval_results/` (or your specified `--output-dir`):

1. **Score Summary File** (`<name>_clears_score.json`): Contains detailed metric scores and results for each subset.
2. **Sample-level Results File** (`<name>_with_eval_result.xlsx` / `.jsonl`): Appends an `eval_result` column (`1` for correct, `0` for incorrect) to the input data for error analysis.

---

## 7. Continual Learning Metrics

After completing evaluations across all continual learning stages (Checkpoints), organize the results into a summary CSV table (following the format in [`example_summary.csv`](example_summary.csv)):

* The first header row contains all task/subset names (`AID`, `VRS-Cap`, `VRS-VG`, ...).
* The second header row indicates the primary metric used for each task (`accuracy`, `CIDEr`, `acc_@0.5`).
* Each row records the performance of a checkpoint stage on all evaluated tasks.

Then run the provided computation script:

```bash
# Defaults to reading eval/example_summary.csv, or pass your own summary CSV path
python compute_cl_metrics.py [path/to/summary.csv]
```

The script computes and prints the 4 standard continual learning metrics: **MFT**, **MFN**, **MAA**, and **BWT**.
