def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    # Unigrams: char -> { "": freq }
    tables = [{} for _ in range(n+1)]
    for char in document:
        if char not in tables[0]:
            tables[0][char] = {}
        tables[0][char][""] = tables[0][char].get("", 0) + 1

    # higher orders as prefix -> {char -> count}
    for i in range(len(document)):
        for k in range(2, n+1):
            if i < k-1: continue
            prefix = document[i-(k-1):i]
            char = document[i]
            tables[k-1].setdefault(prefix, {})
            tables[k-1][prefix][char] = tables[k-1][prefix].get(char, 0) + 1

    return tables


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
    n = len(tables)
    full = sequence + char
    uni_table = tables[0]
    uni_total = sum(counts.get("", 0) for counts in uni_table.values())
    numerator, denominator = 1, 1

    for i, sym in enumerate(full):
        # Back off from highest k down to 1
        for k in range(n, 0, -1):
            prefix_len = k - 1
            if i < prefix_len:
                continue

            prefix = full[i-prefix_len : i] if prefix_len else ""
            table = tables[k-1]

            if prefix_len == 0:
                # Unigram case: table == uni_table, outer key is the symbol
                counts = table.get(sym)
                if not counts:
                    continue
                count = counts.get("", 0)
                total = uni_total
            else:
                # Higher‑order n-grams: outer key is the prefix
                counts = table.get(prefix)
                if not counts or sym not in counts:
                    continue
                count = counts[sym]
                total = sum(counts.values())

            # Multiply in the fraction count/total
            numerator   *= count
            denominator *= total
            joint = numerator /denominator
            break
        else:
            return 0.0

    return joint


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
    best_char = None
    best_prob = 0

    for char in vocabulary:
        prob = calculate_probability(sequence, char, tables)
        if prob > best_prob:
            #If new prob is the highest, replace the best prob and best char
            best_prob = prob
            best_char = char

    return best_char
