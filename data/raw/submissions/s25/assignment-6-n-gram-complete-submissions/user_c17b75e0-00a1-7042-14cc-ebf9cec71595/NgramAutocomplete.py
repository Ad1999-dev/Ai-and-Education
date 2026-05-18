import collections
from typing import List, Dict


def create_frequency_tables(document: str, n: int) -> List[Dict[str, int]]:
    """
    Build n frequency tables for k-grams (k = 1..n) from the document.
    Each table[k-1] maps k-length string -> frequency.
    """
    # Initialize list of counters for k=1 to n
    tables: List[collections.Counter] = [collections.Counter() for _ in range(n)]
    length = len(document)
    # Slide over the document
    for i in range(length):
        # For each possible k-gram ending at i
        for k in range(1, n+1):
            if i - k + 1 < 0:
                break
            gram = document[i-k+1 : i+1]
            tables[k-1][gram] += 1
    return tables


def calculate_probability(sequence: str, char: str, tables: List[Dict[str, int]]) -> float:
    """
    Estimate P(next_char = char | sequence) using the provided n-gram tables.
    Uses the highest-order table available (i.e., len(tables)-1 context length).
    """
    n = len(tables)
    # Determine how many context chars we can use (0 to n-1)
    context_length = min(len(sequence), n - 1)
    context = sequence[-context_length:] if context_length > 0 else ''
    # Joint k-gram length = context_length + 1
    joint = context + char
    # Numerator: count of k-gram of length context_length+1
    numerator = tables[context_length].get(joint, 0)
    # Denominator: count of context (for context_length > 0) or total characters for unigram
    if context_length == 0:
        # Sum of all unigram counts
        denominator = sum(tables[0].values())
    else:
        denominator = tables[context_length-1].get(context, 0)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def predict_next_char(sequence: str, tables: List[Dict[str, int]], vocabulary: List[str]) -> str:
    """
    Predict the most likely next character given the sequence and n-gram tables.
    """
    best_char = None
    best_prob = -1.0
    for c in vocabulary:
        p = calculate_probability(sequence, c, tables)
        if p > best_prob:
            best_prob = p
            best_char = c
    # In case all probabilities are zero, fall back to first vocabulary item
    return best_char if best_char is not None else vocabulary[0]
