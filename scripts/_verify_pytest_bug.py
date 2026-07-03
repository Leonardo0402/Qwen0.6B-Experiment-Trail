"""Verify the bug: test code has no 'from solution import' so functions are undefined."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

code = '''def count_char(str1, char):
    count = str1.count(char)
    return (count)
'''
tests_raw = '''assert count_char("Python",'o')==1
assert count_char("little",'t')==2
'''
tests_fixed = '''from solution import count_char
assert count_char("Python",'o')==1
assert count_char("little",'t')==2
'''

for label, tests in [("raw (no import)", tests_raw), ("fixed (with import)", tests_fixed)]:
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(tests)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--no-header"],
            cwd=tmpdir, capture_output=True, text=True, timeout=10,
        )
        print(f"\n=== {label} ===")
        print(f"  exit code: {proc.returncode}")
        print(f"  stdout: {proc.stdout.strip()[:400]}")
        print(f"  stderr (first 300): {proc.stderr.strip()[:300]}")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
