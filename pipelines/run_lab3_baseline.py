
import subprocess
import sys

print("=" * 55)
print("STUDENT PERFORMANCE - LAB 3 BASELINE PIPELINE")
print("=" * 55)

scripts = [
    "src/train.py",
    "src/evaluate.py"
]

for script in scripts:
    print(f"\n[INFO] Executing {script}...")

    result = subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"[SUCCESS] {script} completed!")

print("\n" + "=" * 55)
print("[SUCCESS] LAB 3 PIPELINE COMPLETED!")
print("=" * 55)
