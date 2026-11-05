
import pandas as pd
import numpy as np
import sys
import os

csv_path = sys.argv[1] if len(sys.argv) > 1 else ("eval/example_summary.csv" if os.path.exists("eval/example_summary.csv") else "example_summary.csv")

primary_metrics = {
    "AID": "accuracy",
    "VRS-Cap": "CIDEr",
    "VRS-VG": "acc_@0.5",
    "VRS-VQA": "accuracy",
    "SAR-VG": "acc_@0.5",
    "SAR-VQA": "accuracy",
    "IR-VG": "acc_@0.5",
    "IR-VQA": "accuracy",
    "FireRisk": "accuracy",
    "RescueNet": "accuracy",
}

df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

tasks = list(primary_metrics.keys())
T = len(tasks)

A = np.full((T, T), np.nan)

for j in range(T):
    for i in range(j + 1):
        task = tasks[i]
        metric = primary_metrics[task]

        value = df.iloc[j][(task, metric)]
        A[j, i] = pd.to_numeric(value, errors="coerce")

MFT = np.mean([A[i, i] for i in range(T)])

MFN = np.mean(A[T - 1, :])

MAA = np.mean([
    np.mean(A[j, :j + 1])
    for j in range(T)
])

BWT = np.mean([
    A[T - 1, i] - A[i, i]
    for i in range(T - 1)
])

print(f"MFT = {MFT:.4f}")
print(f"MFN = {MFN:.4f}")
print(f"MAA = {MAA:.4f}")
print(f"BWT = {BWT:.4f}")
