"""Run the whole pipeline, 01 to 09, in one go.

Every saved output and every file in output/ then comes from the same run. Running one
notebook by hand afterwards breaks that, and notebook 08 stops when it sees it.

Run it from inside this folder: python run_all.py
"""
import os
import shutil
import subprocess
import sys

NOTEBOOKS = [
    "01_explore.ipynb",
    "02_clean_and_split.ipynb",
    "03_features.ipynb",
    "04_baseline.ipynb",
    "05_train_model.ipynb",
    "06_evaluate_and_rank.ipynb",
    "07_generator.ipynb",
    "08_validate.ipynb",
    "09_demo.ipynb",
]

# Start from an empty output/, so no notebook picks up a file from an earlier run.
# The raw Kaggle file stays in ../data, it never changes
if os.path.isdir("output"):
    shutil.rmtree("output")
os.makedirs("output")

for notebook in NOTEBOOKS:
    print("Running", notebook)
    # A fresh kernel per notebook, outputs saved into the notebook itself. No time limit per cell
    result = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                             "--inplace", "--ExecutePreprocessor.timeout=-1", notebook])
    if result.returncode != 0:
        print("Stopped:", notebook, "failed. The notebooks after it were not run.")
        sys.exit(1)

print("All 9 notebooks ran, in order, in one run.")
