"""Print the saved evaluation produced by train_model.py (all numbers come from the real held-out test run)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, config as C
m = json.load(open(C.MODEL_DIR / "model_metrics.json"))
print("Selected:", m["selected_model"], "| threshold", round(m["decision_threshold"], 3), "|", m["data_note"])
print(pd.DataFrame(m["model_comparison"])[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]].round(3).to_string(index=False))
print("Majority-class accuracy (do-nothing baseline):", round(m["majority_class_accuracy"], 3))
