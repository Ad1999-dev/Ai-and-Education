# test_ngram_extended.py

import pytest
from NgramAutocomplete import create_frequency_tables, calculate_probability, predict_next_char

def almost_equal(a, b, eps=1e-9):
    return abs(a-b) < eps

@pytest.mark.parametrize("doc,n,expected_unigrams", [
    ("", 1, {}),                # empty document
    ("x", 1, {"x":1}),          # single character
    ("ababab", 1, {"a":3,"b":3}),
    ("aaaab", 1, {"a":4,"b":1}),
])
def test_unigram_counts(doc, n, expected_unigrams):
    tables = create_frequency_tables(doc, n)
    assert dict(tables[0]) == expected_unigrams

@pytest.mark.parametrize("doc,n,context,expected_counts", [
    # bigram counts for "ababab"
    ("ababab", 2, "a", {"ab":3}),
    ("ababab", 2, "b", {"ba":3}),
    # bigram counts for "aaaab"
    ("aaaab", 2, "a", {"aa":3}),
    ("aaaab", 2, "b", {"ab":1}),
    # trigram counts for "ababab"
    ("ababab", 3, "ab", {"aba":2, "abb":0}),
    ("ababab", 3, "ba", {"bab":2, "bac":0}),
])
def test_higher_order_counts(doc, n, context, expected_counts):
    tables = create_frequency_tables(doc, n)
    k = len(context)
    table = tables[k]  # counts of (k+1)-grams
    # verify each expected (context+char) count
    for suffix, exp in expected_counts.items():
        # suffix already includes context, e.g. "aba"
        assert table.get(suffix, 0) == exp

@pytest.mark.parametrize("doc,n,context,char,expected_prob", [
    # unigrams
    ("ababab", 1, "", "a", 3/6),
    ("ababab", 1, "", "b", 3/6),
    ("aaaab",   1, "", "a", 4/5),
    ("aaaab",   1, "", "b", 1/5),

    # bigrams
    ("ababab", 2, "a", "b", 1.0),   # f("ab")=3 / f("a")=3
    ("ababab", 2, "b", "a", 1.0),   # f("ba")=3 / f("b")=3
    ("aaaab",  2, "a", "a", 3/4),   # f("aa")=3 / f("a")=4
    ("aaaab",  2, "a", "b", 1/4),   # f("ab")=1 / f("a")=4

    # trigrams
    ("ababab", 3, "ab", "a", 2/3),  # f("aba")=2 / f("ab")=3
    ("ababab", 3, "ba", "b", 2/3),  # f("bab")=2 / f("ba")=3
    ("aaaab",  3, "aa", "a", 2/3),  # f("aaa")=2 / f("aa")=3
    ("aaaab",  3, "aa", "b", 1/3),  # f("aab")=1 / f("aa")=3

    # unseen context => 0
    ("abab", 2, "c", "a", 0.0),
    ("abab", 2, "a", "c", 0.0),
])
def test_calculate_probability(doc, n, context, char, expected_prob):
    tables = create_frequency_tables(doc, n)
    p = calculate_probability(context, char, tables)
    assert almost_equal(p, expected_prob)

def test_predict_next_char():
    # 1) Unigram fallback to most frequent
    tables = create_frequency_tables("aaabb", 1)
    vocab = ["a","b","c"]
    # P(a)=3/5, P(b)=2/5, P(c)=0 → predict "a"
    assert predict_next_char("", tables, vocab) == "a"

    # 2) Bigram obvious
    tables = create_frequency_tables("abab", 2)
    vocab = ["a","b"]
    # P(b|a)=1, P(a|a)=0 → predict "b"
    assert predict_next_char("a", tables, vocab) == "b"
    # P(a|b)=1 → predict "a"
    assert predict_next_char("b", tables, vocab) == "a"

    # 3) Trigram obvious
    tables = create_frequency_tables("abcabcabc", 3)
    vocab = ["a","b","c"]
    # P(c|ab)=1 → predict "c"
    assert predict_next_char("ab", tables, vocab) == "c"

    # 4) Unseen context → all zero → returns first vocab
    tables = create_frequency_tables("xyz", 2)
    vocab = ["m","n","o"]
    assert predict_next_char("qq", tables, vocab) == "m"

def test_empty_and_single_char_documents():
    # empty doc
    tables = create_frequency_tables("", 3)
    vocab = list("xyz")
    assert calculate_probability("", "x", tables) == 0.0
    assert calculate_probability("", "y", tables) == 0.0
    assert predict_next_char("", tables, vocab) == "x"

    # single-char doc, n=1
    tables = create_frequency_tables("z", 1)
    vocab = ["z","a"]
    assert almost_equal(calculate_probability("", "z", tables), 1.0)
    assert calculate_probability("", "a", tables) == 0.0
    assert predict_next_char("", tables, vocab) == "z"
