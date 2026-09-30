import subprocess
import sys

print("=" * 60)
print("STUDENT PERFORMANCE - LAB 6 MODEL REGISTRY")
print("=" * 60)

subprocess.run(
    [sys.executable, "src/model_registry.py"],
    check=True
)

print("\n[SUCCESS] Lab 6 completed successfully!")