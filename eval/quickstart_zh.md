# CLeaRS 评测工具

[English](quickstart.md) | 简体中文

本目录提供 **CLeaRS (Continual Vision-Language Learning for Remote Sensing)** 独立评测工具。


## 1. 基准概览 (Overview)

CLeaRS 涵盖遥感领域中多种传感模态（光学、红外、SAR）与任务类型，共包含 **10 个测试子集**，归属于 4 个任务类型：

| 任务类型 (Task) | 子集名称 (Subset) | 模态 (Modality) | 评测指标 (Primary Metric) |
| :--- | :--- | :--- | :--- |
| **分类 (CLS)** | `AID` | 光学航拍 | Accuracy (%) |
| **分类 (CLS)** | `FireRisk` | 多光谱遥感 | Accuracy (%) |
| **视觉问答 (VQA)** | `RescueNet` | 灾后光学影像 | Accuracy (%) |
| **视觉问答 (VQA)** | `IR-VQA` | 红外无人机 (Thermal/IR) | Accuracy (%) |
| **视觉问答 (VQA)** | `SAR-VQA` | 合成孔径雷达 (SAR) | Accuracy (%) |
| **视觉问答 (VQA)** | `VRS-VQA` | 光学卫星影像 | Accuracy (%) |
| **视觉定位 (VG)** | `IR-VG` | 红外无人机 (Thermal/IR) | Acc@0.5 (%) |
| **视觉定位 (VG)** | `SAR-VG` | 合成孔径雷达 (SAR) | Acc@0.5 (%) |
| **视觉定位 (VG)** | `VRS-VG` | 光学卫星影像 | Acc@0.5 (%) |
| **图像描述 (Caption)** | `VRS-Cap` | 光学卫星影像 | normalized_CIDEr |

---

## 2. 推理提示词模板 (Inference Prompt Templates)

为了保证不同模型之间评测的公平性，建议在模型推理时，在 `question` 前拼接对应子集的 Prompt 前缀规范输出格式：

```python
CLEARS_PROMPTS = {
    # 分类任务 (CLS)
    'AID': 'Answer with exactly one word from the given choices, no extra text.Do not respond with numbers.',
    'FireRisk': 'Answer with exactly one word from the given choices, no extra text.Do not respond with numbers.',

    # 视觉问答任务 (VQA)
    'RescueNet': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'IR-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'SAR-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',
    'VRS-VQA': 'Please answer concisely with short words or phrases when possible. Do not add any extra text.',

    # 视觉定位任务 (VG)
    'IR-VG': 'Please output unique bounding box and do not include any extra content.',
    'SAR-VG': 'Please output unique bounding box and do not include any extra content.',
    'VRS-VG': 'Please output unique bounding box and do not include any extra content.',

    # 图像描述任务 (Caption)
    'VRS-Cap': '',  # 使用原始提问
}

# 推理拼接示例
prefix = CLEARS_PROMPTS.get(subset_name, '')
full_prompt = f"{prefix}{question}" if prefix else question
```

---

## 3. 环境准备 (Requirements)

```bash
cd eval
pip install -r requirements.txt
```

---

## 4. 快速开始 (Quick Start)

### 4.1 运行评测
只需将模型的生成结果保存为表格或 JSONL 文件，然后运行：

```bash
# 评测预测文件（默认输出到 eval_results/ 目录）
python evaluate.py --pred /path/to/predictions.xlsx

# 也可以指定自定义输出目录
python evaluate.py --pred /path/to/predictions.jsonl --output-dir custom_results/
```

### 4.2 使用自带示例验证
我们提供了一份包含 10 个子集样例的测试文件 [example_predictions.jsonl](example_predictions.jsonl)，可直接运行验证：

```bash
python evaluate.py --pred example_predictions.jsonl
```

控制台将输出格式化的评测表格：

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

## 5. 预测文件格式要求 (Input Format)

支持格式：`.xlsx`, `.xls`, `.csv`, `.tsv`, `.jsonl`, `.json`。

文件中每一行代表一个样本的预测结果，包含以下字段：

| 字段名 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `subset` | string | **[必填]** 所属子集名称 | `AID`, `RescueNet`, `VRS-VG` 等 |
| `answer` | string | **[必填]** Ground Truth 真值 | `"airport"`, `"yes"`, `"[120, 85, 230, 195]"` |
| `prediction` | string | **[必填]** 模型推理生成的预测文本 | `"airport"`, `"[122, 88, 228, 192]"` |
| `question` | string | **[必填]** 提问内容（用于分类任务候选类别解析） | `"<cls>Answer with ... ['airport', 'bridge']"` |
| `index` | string | *[可选]* 样本唯一标识符 | `"AID_0"` |
| `task_type` | string | *[可选]* 任务类型（若未提供，将自动根据 `subset` 识别） | `"CLS"`, `"vqa"`, `"vg"`, `"caption"` |

**JSONL 格式示例**：
```json
{"index": "AID_0", "subset": "AID", "question": "Answer with ... ['airport', 'bridge']", "answer": "airport", "prediction": "airport"}
{"index": "IR-VG_0", "subset": "IR-VG", "question": "Detect the vehicle.", "answer": "[120, 85, 230, 195]", "prediction": "[122, 88, 228, 192]"}
```

---

## 6. 输出结果说明 (Outputs)

评测完成后，结果默认保存在 `eval_results/`（或指定的 `--output-dir`）目录下，生成两个文件：

1. **分数汇总文件** (`<name>_clears_score.json`)：包含各个子集的指标得分与判定结果。
2. **样本级结果** (`<name>_with_eval_result.xlsx` / `.jsonl`)：在输入数据基础上追加 `eval_result` 列（`1` 为正确，`0` 为错误），便于分析 Bad Case。

---

## 7. 持续学习指标计算 (Continual Learning Metrics)

完成持续学习所有阶段（Checkpoints）的评测后，可将测试结果整理成表格（格式参考 [`example_summary.csv`](example_summary.csv)）：

* 表头第一行包含所有评测任务/子集名称（如 `AID`, `VRS-Cap`, `VRS-VG`, ...）。
* 表头第二行指明各任务采用的主要指标（如 `accuracy`, `CIDEr`, `acc_@0.5`）。
* 每行记录一个 Checkpoint 评测阶段在各个已学任务上的表现。

整理完成后，运行提供的计算脚本：

```bash
# 默认读取 eval/example_summary.csv，也可以传入自己的汇总文件路径
python compute_cl_metrics.py [path/to/summary.csv]
```

脚本将计算并输出 4 个持续学习指标：MFT, MFN, MAA, BWT
