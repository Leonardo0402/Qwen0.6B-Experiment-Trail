"""Wrapper to run boundary training with proper logging."""
import sys
import os
import traceback
from pathlib import Path

# Set working directory
os.chdir(Path(__file__).resolve().parent)

# Add to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure stdout
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

print("=== Boundary Training Wrapper ===")
print(f"Python: {sys.executable}")
print(f"Working dir: {os.getcwd()}")

try:
    from scripts.train_lora import run_training
    config_path = str(Path(__file__).resolve().parent / "configs" / "train_boundary.yaml")
    print(f"Config: {config_path}")
    print("Starting training...")
    sys.stdout.flush()
    result = run_training(config_path)
    print(f"Training completed with exit code: {result}")
except Exception:
    print("Training failed with exception:")
    traceback.print_exc()
    sys.exit(1)