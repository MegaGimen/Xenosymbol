"""Pure-Python definitions shared by training, analysis and tests."""
import math


def content_tokens(tokens, stop_ids):
    """Remove the stopping token, retaining all actual content tokens."""
    for i, token in enumerate(tokens):
        if token in stop_ids:
            return tokens[:i], True
    return list(tokens), False


def reward(distance, message_length, length_weight):
    if not math.isfinite(distance) or message_length < 0 or length_weight < 0:
        raise ValueError('Invalid objective inputs')
    return -(distance + length_weight * message_length)


def leave_one_out_advantages(rewards):
    if len(rewards) < 2:
        raise ValueError('At least two independent samples are required')
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def selected_positions(prompt_length, output_length):
    if prompt_length < 1 or output_length < 1:
        raise ValueError('An actual output content token is required')
    return prompt_length - 1, prompt_length + output_length - 1
