"""Task 1.1: print reproducible environment and TensorFlow device evidence."""
import argparse
import getpass
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

parser = argparse.ArgumentParser()
parser.add_argument("--section", choices=["versions", "gpu", "all"], default="all")
args = parser.parse_args()
print("VisionGuard | 23L-0885_A2_Ops | Task 1.1", flush=True)
print("Recorded:", datetime.now(timezone(timedelta(hours=5))).isoformat(timespec="seconds"), "(Asia/Karachi)")
print("User:", getpass.getuser())
print("Project:", Path(__file__).resolve().parents[1])
print()

if args.section in ("versions", "all"):
    print("ENVIRONMENT AND EXACT PACKAGE VERSIONS")
    print("OS:", platform.platform())
    print("Python:", platform.python_version())
    print("Executable:", sys.executable)
    print("Virtual environment isolated:", sys.prefix != sys.base_prefix)
    print("Base Python:", sys.base_prefix)
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    for line in cfg.read_text().splitlines():
        if line.startswith("include-system-site-packages"):
            print(line)
    for name in ("pip", "tensorflow", "keras", "numpy", "pandas", "matplotlib", "seaborn", "pillow", "scikit-learn"):
        print(f"{name:16} {importlib.metadata.version(name)}")
    print()
    print("Command: python -m pip check")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    print(check.stdout.strip())
    if check.returncode:
        print(check.stderr)
        sys.exit(check.returncode)

if args.section in ("gpu", "all"):
    import tensorflow as tf
    print("TENSORFLOW IMPORT, DEVICE DETECTION AND COMPUTATION")
    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("CPU devices:", tf.config.list_physical_devices("CPU"))
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("GPU count:", len(tf.config.list_physical_devices("GPU")))
    result = tf.matmul(tf.constant([[1., 2.]]), tf.constant([[3.], [4.]]))
    print("Matrix multiplication [[1,2]] x [[3],[4]]:", result.numpy().tolist())
    print("Computation device:", result.device)
    assert result.numpy().tolist() == [[11.0]]
    print("TensorFlow smoke check: PASS")
    print()
    print("NVIDIA hardware check (separate from TensorFlow detection):")
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], capture_output=True, text=True)
        print(proc.stdout.strip() or proc.stderr.strip())
    except FileNotFoundError:
        print("nvidia-smi not available")
