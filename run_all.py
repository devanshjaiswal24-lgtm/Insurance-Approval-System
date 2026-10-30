"""One command to build everything:  python run_all.py   (then: python app.py)"""
import subprocess, sys
import config as C
steps = ["training/train_model.py", "explainability/build_global.py", "training/fairness_analysis.py", "training/evaluate_model.py"]
if C.DATA_SOURCE == "synthetic":
    steps.insert(0, "training/generate_data.py")
for s in steps:
    print(f"\n=== {s} ===", flush=True); subprocess.run([sys.executable, s], check=True)
print("\nDone. Start the web app with:  python app.py   and open http://127.0.0.1:5000")
