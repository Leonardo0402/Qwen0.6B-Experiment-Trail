"""Extract Python function signature from target_code.

Used to augment instruction with function signature so the model knows
the exact function name and parameters expected by tests.
"""
import re
from typing import Optional


def extract_function_signature(code: str) -> Optional[str]:
    """Extract the first function signature line from Python code.

    Returns the signature string like:
        "def max_of_nth(test_list, N):"
    or None if no function definition is found.
    """
    if not code:
        return None
    # Match: def function_name(params):
    # Handles optional type hints and multi-line params
    match = re.search(r'^def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:', code, re.MULTILINE)
    if match:
        return match.group(0)
    return None


def extract_function_name(code: str) -> Optional[str]:
    """Extract just the function name from Python code."""
    if not code:
        return None
    match = re.search(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def augment_instruction(instruction: str, target_code: str) -> str:
    """Augment instruction with function signature from target_code.

    If the instruction already contains the function name, no augmentation.
    Otherwise, append the function signature to the instruction.

    Example:
        instruction: "Write a function to find the maximum of nth column."
        target_code: "def max_of_nth(test_list, N):\\n  ..."
        result: "Write a function to find the maximum of nth column.

        Function signature: def max_of_nth(test_list, N):"
    """
    if not instruction or not target_code:
        return instruction

    sig = extract_function_signature(target_code)
    name = extract_function_name(target_code)
    if not sig or not name:
        return instruction

    # If instruction already mentions the function name, don't duplicate
    if name in instruction:
        return instruction

    return f"{instruction}\n\nFunction signature: {sig}"


if __name__ == "__main__":
    # Test
    code = "def max_of_nth(test_list, N):\n  res = max([sub[N] for sub in test_list])\n  return (res)"
    instr = "Write a function to find the maximum of nth column from the given tuple list."
    print("Original:", repr(instr))
    print("Augmented:", repr(augment_instruction(instr, code)))
