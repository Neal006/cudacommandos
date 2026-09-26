import subprocess
import os
import sys

run_dir = os.path.join(os.environ["SM_MODEL_DIR"], "run_007")

result = subprocess.run([
    sys.executable, "run_v2.py",
    "--sample", "150000",
    "--train-only",
    "--run-dir", run_dir,
], cwd=os.path.dirname(os.path.abspath(__file__)))

sys.exit(result.returncode)