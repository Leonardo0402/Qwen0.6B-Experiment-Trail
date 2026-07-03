"""Manually verify that 'failed' Stage2 code is actually correct."""
import subprocess
import sys
import tempfile
from pathlib import Path

# These are Stage2 outcomes that "failed" but look correct
test_cases = [
    {
        "name": "mbpp_666 count_char",
        "code": '''def count_char(str1, char):
    count = str1.count(char)
    return (count)
''',
        "tests": '''
assert count_char("Python",'o')==1
assert count_char("little",'t')==2
assert count_char("assert",'s')==2
print("mbpp_666 PASS")
'''
    },
    {
        "name": "mbpp_678 remove_spaces",
        "code": '''def remove_spaces(text):
    text = text.replace(' ', "")
    return text
''',
        "tests": '''
assert remove_spaces("a b c") == "abc"
assert remove_spaces("1 2 3") == "123"
assert remove_spaces(" b c") == "bc"
print("mbpp_678 PASS")
'''
    },
    {
        "name": "mbpp_611 max_of_nth (Stage1 version)",
        "code": '''def max_of_nth(test_list, N):
    return (max( test_list[N-1] for x in test_list if isinstance(x, int) ))
''',
        "tests": '''
assert max_of_nth([(5, 6, 7), (1, 3, 5), (8, 9, 19)], 2) == 19
print("mbpp_611 PASS")
'''
    },
    {
        "name": "mbpp_611 max_of_nth (correct version)",
        "code": '''def max_of_nth(test_list, N):
    res = max([sub[N] for sub in test_list])
    return (res)
''',
        "tests": '''
assert max_of_nth([(5, 6, 7), (1, 3, 5), (8, 9, 19)], 2) == 19
assert max_of_nth([(6, 7, 8), (2, 4, 6), (9, 10, 20)], 1) == 10
assert max_of_nth([(7, 8, 9), (3, 5, 7), (10, 11, 21)], 1) == 11
print("mbpp_611 CORRECT PASS")
'''
    },
]

for tc in test_cases:
    print(f"\n=== {tc['name']} ===")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(tc["code"])
        f.write(tc["tests"])
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  exit code: {result.returncode}")
        print(f"  stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr.strip()[:500]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        Path(path).unlink(missing_ok=True)
