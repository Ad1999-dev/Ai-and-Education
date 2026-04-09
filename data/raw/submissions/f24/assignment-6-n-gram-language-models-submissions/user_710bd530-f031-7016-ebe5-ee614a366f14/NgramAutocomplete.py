def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    result = []
    for i in range(1, n + 1):
        currTable = {}          # key = i-gram, value = cnt
        total = 0
        j = 0
        while j < len(document) and j + i <= len(document):
            s = document[j : j + i]
            if s not in currTable:
                currTable[s] = 0
            currTable[s] += 1
            total += 1
            j += 1
        result.append(currTable)
    return result


def calculate_probability(sequence, char, tables):
    """
    Calculates the probability of observing a given sequence of characters using the frequency tables.

    - **Parameters**:
        - `sequence`: The sequence of characters whose probability we want to compute.
        - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
        - `char`: The character whose probability of occurrence after the sequence is to be calculated.

    - **Returns**:
        - Returns a probability value for the sequence.
    """
    # Formula 1:
    m = len(sequence)
    n = len(tables)
    if m >= n:
        return 0
    word = sequence + char
    sequence_table = tables[m - 1]
    word_table = tables[m]
    f_sequence = sequence_table[sequence] if sequence in sequence_table else 0
    f_word = word_table[word] if word in word_table else 0
    if f_sequence == 0:
        return 0
    return f_word / f_sequence

# Helper function
def calculate_target_probability(target, tables, n, c):
    # using n-grams formula
    m = len(target)
    result = 1
    if m > n:
        # apply n-grams formula
        for i in range(m - n + 1):
            if i == 0:
                num = target[i : i + n]
                table = tables[len(num) - 1]
                f_num = table[num] if num in table else 0
                result = result * (f_num / c)
            else:
                sequence = target[i : i + n - 1]
                char = target[i + n - 1]
                result = result * calculate_probability(sequence, char, tables)
    else:
        num = target
        table = tables[len(num) - 1]
        f_num = table[num] if num in table else 0
        result = result * (f_num / c)
    return result

def predict_next_char(sequence, tables, vocabulary):
    """
    Predicts the most likely next character based on the given sequence.

    - **Parameters**:
        - `sequence`: The sequence used as input to predict the next character.
        - `tables`: The list of frequency tables.
        - `vocabulary`: The set of possible characters.
    
    - **Functionality**:
        - Calculates the probability of each possible next character in the vocabulary, using `calculate_probability()`.

    - **Returns**:
        - Returns the character with the maximum probability as the predicted next character.
    """
    result = ""
    maxP = float("-inf")
    one_gram_table = tables[0]
    corpus_length = 0
    for value in one_gram_table.values():
        corpus_length += value
    # print("Corpus length -> ", corpus_length)
    for ch in vocabulary:
        target = sequence + ch
        p = calculate_target_probability(target, tables, len(tables), corpus_length)
        if p > maxP:
            result = ch
            maxP = p
    # print("Max Prob -> ", maxP)
    # print("Result -> ", result)
    return result


# Test code

# Part 1
# document = "aababcaccaaacbaabcaa"
# n = 3
# all_tables = create_frequency_tables(document, n)
# print(all_tables)
# for table in all_tables:
#     print(table)

# Part 2
# sequence = "aa"
# ch = "b"
# document = "aabaa"
# n = 3
# all_tables = create_frequency_tables(document, n)
# p_ch = calculate_probability(sequence, ch, all_tables)
# print("Probability of ch given sequence -> ", p_ch)

# Helper function
# target = "ab"
# helper_res = calculate_target_probability(target, all_tables, len(all_tables), 4)
# print(helper_res)

# Part 3
# sequence = "aa"
# voc = set(document)
# print("Voc -> ", voc)
# res = predict_next_char(sequence, all_tables, voc)
# print(res)
