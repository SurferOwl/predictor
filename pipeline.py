"""
pipeline.py

Runs the full training pipeline in order:

  Step 1 — clean raw CSVs          (data/ → cleaned_data/)
  Step 2 — merge all cleaned CSVs  (cleaned_data/ → merged/)
  Step 3 — synonym reduction       (merged/ → merged/)
  Step 4 — train models            (merged/ → model/)

Usage:
    python pipeline.py              # run all steps
    python pipeline.py --from 3     # resume from step 3
    python pipeline.py --only 4     # run only step 4
"""

import sys
import time
import argparse
import subprocess

STEPS = [
    (1, "clean_data.py",          "Clean raw CSVs → cleaned_data/"),
    (2, "merge_data.py",          "Merge cleaned CSVs → merged/merged_dataset.csv"),
    (3, "normalize_diseases.py",  "Normalize disease name synonyms"),
    (4, "synonyms.py",            "Symptom synonym reduction"),
    (5, "train_model.py",         "Train models → model/disease_model.pkl"),
]


def run_step(num, script, description):
    print(f"\n{'='*60}")
    print(f"  STEP {num}: {description}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  Step {num} FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n✅  Step {num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",  dest="from_step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only",  dest="only_step", type=int, default=None,
                        help="Run only this step")
    args = parser.parse_args()

    steps_to_run = STEPS
    if args.only_step:
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
    elif args.from_step > 1:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]

    if not steps_to_run:
        print("No matching steps found.")
        sys.exit(1)

    total_start = time.time()
    print(f"\n🚀  Running {len(steps_to_run)} step(s): {[s[0] for s in steps_to_run]}")

    for num, script, description in steps_to_run:
        run_step(num, script, description)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✅  Pipeline complete in {total:.1f}s")
    print(f"{'='*60}")
    print("\nFolder summary:")
    print("  data/          ← your raw source CSVs (untouched)")
    print("  cleaned_data/  ← binary symptom matrices per source")
    print("  merged/        ← merged_dataset.csv + synonyms + symptom_list.pkl")
    print("  model/         ← disease_model.pkl + symptom_list.pkl + test_data.csv")


if __name__ == "__main__":
    main()