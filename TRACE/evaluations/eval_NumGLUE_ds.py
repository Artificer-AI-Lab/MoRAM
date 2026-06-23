import json
from metrics import caculate_accuracy

# Use Qwen2.5-Math parser for better numerical answer extraction
try:
    from evaluations.qwen_math_parser import calculate_math_accuracy, extract_answer, strip_string
    QWEN_PARSER_AVAILABLE = True
except ImportError:
    try:
        from qwen_math_parser import calculate_math_accuracy, extract_answer, strip_string
        QWEN_PARSER_AVAILABLE = True
    except ImportError:
        QWEN_PARSER_AVAILABLE = False


def eval(predicted_sequences, ground_truths, use_qwen_parser=True):
    """
    Evaluate NumGLUE-ds predictions.
    
    Args:
        predicted_sequences: List of model predictions
        ground_truths: List of ground truth answers
        use_qwen_parser: If True, use Qwen2.5-Math parser for better numerical extraction
    """
    if use_qwen_parser and QWEN_PARSER_AVAILABLE:
        # Use Qwen parser for robust numerical answer extraction
        accuracy = calculate_math_accuracy(predicted_sequences, ground_truths, use_boxed=True)
    else:
        # Fallback to simple exact match
        accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
