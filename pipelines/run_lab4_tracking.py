
import subprocess
import sys

print("=" * 55)
print("STUDENT PERFORMANCE - LAB 4 MLFLOW TRACKING")
print("=" * 55)

subprocess.run(
    [sys.executable, "src/train_tracking.py"],
    check=True
)

print("\n[SUCCESS] LAB 4 COMPLETED!")
