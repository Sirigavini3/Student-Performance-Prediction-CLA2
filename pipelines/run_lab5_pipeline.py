import subprocess
import sys

print("=" * 60)
print("STUDENT PERFORMANCE - LAB 5 PRODUCTION PIPELINE")
print("=" * 60)

subprocess.run(
    [sys.executable, "src/production_pipeline.py"],
    check=True
)

print("\n[SUCCESS] Lab 5 completed successfully!")