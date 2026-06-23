"""
Qwen2.5-Math official parser for extracting and comparing mathematical answers.
Adapted from: https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation

This module provides functions to:
1. Extract answers from \boxed{} format
2. Normalize mathematical strings
3. Compare mathematical answers (numerical and symbolic equality)
"""

import re
from math import isclose
from typing import Union

# regex is an optional enhanced regex library; fall back to re if not available
try:
    import regex
except ImportError:
    regex = re  # type: ignore

try:
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False

try:
    from word2number import w2n
    WORD2NUMBER_AVAILABLE = True
except ImportError:
    WORD2NUMBER_AVAILABLE = False


def find_box(pred_str: str) -> str:
    """Extract content from \\boxed{...} in the prediction string."""
    if "boxed" not in pred_str:
        return ""
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def extract_last_number(text: str) -> str:
    """Extract the last number from text (fallback when no boxed answer)."""
    # Try to find numbers (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return text.strip()


def convert_word_number(text: str) -> str:
    """Convert word numbers to digits."""
    if not WORD2NUMBER_AVAILABLE:
        return text
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


def _fix_fracs(string):
    """Fix LaTeX fractions like \\frac12 to \\frac{1}{2}."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
        string = new_str
    return string


def _fix_a_slash_b(string):
    """Convert a/b to \\frac{a}{b}."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    """Fix LaTeX sqrt like \\sqrt2 to \\sqrt{2}."""
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string, skip_unit=False):
    """Normalize a mathematical string."""
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")
    # right "."
    string = string.rstrip(".")
    
    # remove inverse spaces
    string = string.replace("\\!", "")
    
    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    
    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    
    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")
    
    # Remove unit: texts
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    
    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    
    # convert word number to digit
    string = convert_word_number(string)
    
    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")
    
    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")  # raw string for literal backslash-percent
    string = string.replace("%", "")
    
    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]
    
    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    
    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    
    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)
    
    # quote
    string = string.replace("'", "")
    string = string.replace('"', "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    
    # replace a.000b where b is not number or b is end, with ab
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    
    # get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    
    return string


def parse_digits(num):
    """Parse a string to float."""
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
        if num.endswith("\\"):
            num = num[:-1]
        try:
            return float(num) / 100
        except:
            pass
    return None


def is_digit(num):
    """Check if a string can be parsed as a number."""
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    """Check if two numbers are equal within tolerance."""
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b) -> bool:
    """Check if two expressions are symbolically equal."""
    if not SYMPY_AVAILABLE:
        return False
    
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        if LATEX2SYMPY_AVAILABLE:
            try:
                return latex2sympy(s.replace("\\\\", "\\"))
            except:
                try:
                    return latex2sympy(s)
                except:
                    pass
        return s
    
    a = _parse(a)
    b = _parse(b)
    
    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass
    
    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass
    
    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass
    
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass
    
    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """
    Check if prediction equals reference mathematically.
    Handles numerical equality and symbolic equality.
    """
    if prediction is None or reference is None:
        return False
    
    pred_str = str(prediction).strip().lower()
    ref_str = str(reference).strip().lower()
    
    # Direct string match
    if pred_str == ref_str:
        return True
    
    # Try numerical comparison
    try:
        if is_digit(prediction) and is_digit(reference):
            pred_num = parse_digits(prediction)
            ref_num = parse_digits(reference)
            # Ensure both are valid numbers (not None)
            if pred_num is not None and ref_num is not None:
                if include_percentage:
                    gt_result = [ref_num / 100, ref_num, ref_num * 100]
                else:
                    gt_result = [ref_num]
                for item in gt_result:
                    try:
                        if is_close:
                            if numeric_equal(pred_num, item):
                                return True
                        else:
                            if item == pred_num:
                                return True
                    except Exception:
                        continue
                return False
    except:
        pass
    
    if not prediction and prediction not in [0, False]:
        return False
    
    # Symbolic comparison
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    
    # Remove brackets for comparison
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True
    
    # Handle equations like "x = 5"
    if prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True
    
    # Symbolic equality
    if symbolic_equal(prediction, reference):
        return True
    
    return False


def extract_answer(pred_str: str, use_last_number: bool = True) -> str:
    """
    Extract the answer from model output.
    1. First try to extract from \\boxed{}
    2. If not found and use_last_number=True, extract the last number
    """
    # Try boxed first
    boxed_ans = find_box(pred_str)
    if boxed_ans:
        return strip_string(boxed_ans)
    
    # Fallback: extract last number
    if use_last_number:
        return extract_last_number(pred_str)
    
    return strip_string(pred_str)


def calculate_math_accuracy(predictions: list, ground_truths: list, use_boxed: bool = True) -> float:
    """
    Calculate accuracy for math problems using Qwen2.5-Math parser.
    
    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        use_boxed: If True, try to extract from \\boxed{} first
    
    Returns:
        Accuracy score (0 to 1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_str = str(pred).strip()
        gt_str = str(gt).strip()
        
        # Extract answer from prediction
        if use_boxed:
            pred_ans = extract_answer(pred_str, use_last_number=True)
        else:
            pred_ans = strip_string(pred_str)
        
        # Normalize ground truth
        gt_ans = strip_string(gt_str)
        
        # Compare
        if math_equal(pred_ans, gt_ans):
            correct += 1
    
    return correct / len(predictions)
