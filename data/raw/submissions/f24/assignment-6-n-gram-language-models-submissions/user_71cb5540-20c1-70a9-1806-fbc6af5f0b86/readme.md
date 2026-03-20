[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bx7CmlmG)

# **_Bayes Complete_**: Sentence Autocomplete using N-Gram Language Models

## Assignment Objectives

1. Understand the mathematical principles behind N-gram language models
2. Implement an n-gram language model from scratch
3. Apply the model to sentence autocomplete functionality.
4. Analyze the performance of the model in this context.

## Pre-Requisites

- **Python Basics:** Familiarity with Python syntax, data structures (lists, dictionaries), and file handling.
- **Probability:** Basic understanding of probability fundamentals (particularly joint distributions and random variables).
- **Bayes:** Theoretical knowledge of how n-gram language models work.

## Overview

In this assignment, you'll be stepping into the shoes of a language model developer. Your mission: to build a sentence autocomplete feature using the power of language models.

Imagine you're working on a messaging app where users want quick and accurate sentence completion suggestions. Your model will analyze the context of the sentence and predict the most likely next letter (repeatedly, thus completing the sentence), helping users express themselves faster and more efficiently.

You'll train your model on a large text corpus, teaching it the patterns and probabilities of letter sequences. Then, you'll put your model to the test, seeing how well it can predict the next letter and complete sentences.

This project will implement an autocomplete model using n-gram language models to predict the next character in a sequence. The model takes a training document, builds frequency tables for n-grams (with up to `n` conditionals), and calculates the probability of the next character given the previous `n` characters.

Get ready to dive into the world of language modeling and build an autocomplete feature that's both smart and helpful!

## Project Components

### 1. **Frequency Table Creation**

The model reads a document and constructs frequency tables based on character sequences. These tables store the frequency of occurrence of a given character, conditioned on the `n` previous characters (`n` grams).

For an `n` gram model, we will have to store `n` tables.

- **Table 1** contains the frequencies of each individual character.
- **Table 2** contains the frequencies of two character sequences.
- **Table 3** contains the frequencies of three character sequences.
- And so on, up to **Table N**.

Consider that our vocabulary just consists of 4 letters, $\{a, b, c, d\}$, for simplicity.

### Table 1: Unigram Frequencies

| Unigram | Frequency |
| ------- | --------- |
| f(a)    |           |
| f(b)    |           |
| f(c)    |           |
| f(d)    |           |

### Table 2: Bigram Frequencies

| Bigram  | Frequency |
| ------- | --------- |
| f(a, a) |           |
| f(a, b) |           |
| f(a, c) |           |
| f(a, d) |           |
| f(b, a) |           |
| f(b, b) |           |
| f(b, c) |           |
| f(b, d) |           |
| ...     |           |

### Table 3: Trigram Frequencies

| Trigram    | Frequency |
| ---------- | --------- |
| f(a, a, a) |           |
| f(a, a, b) |           |
| f(a, a, c) |           |
| f(a, a, d) |           |
| f(a, b, a) |           |
| f(a, b, b) |           |
| ...        |           |

And so on with increasing sizes of n.

### 2. **Computing Joint Probabilities for a Language Model**

In general, Bayesian Networks are used to visually represent the dependencies (edges) between distinct random varaibles (nodes) in a large joint distribution.

In the case of a language model, each node in the network corresponds to a character in the sequence, and edges represent the conditional dependencies between them.

For a character sequence of length 4 a bayesian network for our the full joint distribution of 4 letter sequences would look as follows.

![image](https://github.com/user-attachments/assets/7812c3c6-9ed2-40aa-bf16-ea4b15f1b394)

Where $X_1$ is a random variable that maps to the character found at position 1 in a character sequence, $X_2$ maps to the character at position 2, and so on.

This makes clear how the chain rule can be applied to expand the full joint form of a probability distribution.

$$P(X_1=x_1, X_2=x_2, X_3=x_3, X_4=x_4) = P(x_1) \cdot P(x_1 \mid x_2) \cdot P(x_3 \mid x_1, x_2) \cdot P(x_4 \mid x_1, x_2, x_3)$$

In our case we are interested in computing the next character (the character at position 4) given the characters at the previous positions (characters at position 1, 2, and 3). Applying the definition of conditional distributions we can see this is

$$P(X_4 = x_4 \mid X_1 = x_1, X_2 = x_2, X_3 = x_3) = \frac{P(X_1 = x_1, X_2 = x_2, X_3 = x_3, X_4 = x_4)}{P(X_1 = x_1, X_2 = x_2, X_3 = x_3)}$$

Which can be estimated using the frequencies of each sequence in a our corpus

$$P(X_4 = x_4 \mid X_1 = x_1, X_2 = x_2, X_3 = x_3) = \frac{f(x_1, x_2, x_3, x_4)}{f(x_1, x_2, x_3)}$$

To make this concrete, consider an input sequence `"thu"`, where we want to predict the probability the next character is "s".

$$P(X_4=s \mid X_1=t, X_2=h, X_3=u) = \frac{P(X_1 = t, X_2 = h, X_3 = u, X_4 = s)}{P(X_1 = t, X_2 = h, X_3 = u)} = \frac{f(t, h, u, s)}{f(t, h, u)}$$

If we wanted to predict the most likely next character, we could compute the probability of every possible completion given each character in our vocabularly. This will give us a probability distribution over the next character prediction $P(X_4=x_4 \mid X_1=t, X_2=h, X_3=u)$. Taking the character with the max probability value in this distribution gives us an autocomplete model.

#### General Case:

Given a sequence $x_1, x_2, \dots, x_t$, the probability of the next character $x_{t+1}$ is calculated as:

$$P(x_{t+1} \mid x_1, x_2, \dots, x_t) = \frac{P(x_1, x_2, \dots, x_t, x_{t+1})}{P(x_1, x_2, \dots, x_t)}$$

This can be generalized for different values of `t`, using the corresponding frequency tables.

### N-gram models:

For short sequences, we can compute our joint probabilities in their entirity. However, as the sequences grows longer, our tables become exponentially larger and this problem quickly grows intractable. Enter n-gram models. An n-gram model is the same model we described above except only `n-1` characters are considered as context for the prediction.

That is for a bigram model `n=2` we estimate the joint probability as

$$P(X_1=x_1, X_2=x_2, X_3=x_3, X_4=x_4) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_2) \cdot P(x_4 \mid x_3)$$

Which can be visually represented with the following Bayesian Network

![image](https://github.com/user-attachments/assets/e9590bfc-d1c6-4ecf-a9c2-bd54dbfa35bd)

Putting this network in terms of computations via our frequency tables is now slightly different as we now have to consider the ratio for each term

$$P(X_1=x_1, X_2=x_2, X_3=x_3, X_4=x_4) = P(x_1) \cdot P(x_1 \mid x_2) \cdot P(x_3 \mid x_2) \cdot P(x_4 \mid x_3) = \frac{f(x_1)}{size(C)} \cdot \frac{f(x_1,x_2)}{f(x_1)} \cdot \frac{f(x_2,x_3)}{f(x_2)} \cdot \frac{f(x_3,x_4)}{f(x_3)}$$

Where `size(C)` is the total number of characters in the corpus. Consider how this generalizes to an arbitrary n-gram model for any `n`, this will be the core of your implementation. Write this formula in your report.

## Starter Code Overview

The project starter code is structured across three main Python files:

1. **NgramAutocomplete.py**: This is where the main logic of the autocomplete model is implemented. You are expected to complete three functions in this file: `create_frequency_tables()`, `calculate_probability()`, and `predict_next_char()`.

2. **main.py**: This file provides the user interface and controls the flow of the program. It initializes the model, takes user inputs, and runs the character prediction process iteratively. You may modify this file to test their code, but no modifications are required to complete the project.

3. **utilities.py**: This file includes helper functions that facilitate the program, such as reading and preprocessing the training document. No modifications are needed in this file.

## TODOs

**_NgramAutocomplete.py_** is the core file where you will change in this project. Each function here builds upon each other to create a probabilistic model for predicting the next character in a sequence.

#### 1. `create_frequency_tables(document, n)`

This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

- **Parameters**:

  - `document`: The text document used to train the model.
  - `n`: The number of value of `n` for the n-gram model.

- **Returns**:
  - Returns a list of n frequency tables.

#### 2. `calculate_probability(sequence, char, tables)`

Calculates the probability of observing a given sequence of characters using the frequency tables.

- **Parameters**:

  - `sequence`: The sequence of characters whose probability we want to compute.
  - `tables`: The list of frequency tables created by `create_frequency_tables()`, this will be of size `n`.
  - `char`: The character whose probability of occurrence after the sequence is to be calculated.

- **Returns**:
  - Returns a probability value for the sequence.

#### 3. `predict_next_char(sequence, tables, vocabulary)`

Predicts the most likely next character based on the given sequence.

- **Parameters**:
  - `sequence`: The sequence used as input to predict the next character.
  - `tables`: The list of frequency tables.
  - `vocabulary`: The set of possible characters.
- **Functionality**:

  - Calculates the probability of each possible next character in the vocabulary, using `calculate_probability()`.

- **Returns**:
  - Returns the character with the maximum probability as the predicted next character.

# A Reports section

## 383GPT

Did you use 383GPT at all for this assignment (yes)

## `create_frequency_tables(document, n)`

### Code analysis

How It Works
Normalization: The document's newlines are replaced with spaces to ensure that the text is processed as a continuous stream of characters. This prevents newline characters from affecting the frequency counts.
Character List Creation: The code converts the document into a list of individual characters. This is important because n-grams are based on these characters.
Frequency Table Initialization: A list called `frequency_tables` is initialized. It contains `n` dictionaries, one for each n-gram length from 1 to n. Each dictionary will store character frequencies based on their contexts.
Counting Frequencies:
The outer loop iterates over `gram_length` from 1 to n. For each length `k`, the inner loop processes segments of the text that correspond to k-grams.
The inner loop's index `i` runs from 0 to `length - gram_length` to ensure that it stays within the bounds of the character list while creating k-grams.
For each position `i`, an n-gram is formed as a tuple of characters. `n_gram` is made up of characters from the current position `i` to `i + gram_length`.
The target character (the one we want to predict based on the preceding context) is obtained as the last character of the n-gram, and the preceding characters are extracted.
Updating Frequency Tables:
For each unique target character (from the n-gram), the function checks if it already exists in the respective frequency table.
If not, it initializes an entry for that character with an empty dictionary to track contexts.
It then checks if the specific context (prior characters) exists in the nested dictionary. If it doesn't, it initializes a counter for that context.
Finally, it increments the counter for the context associated with the target character.
Return Value: After processing all characters in the document for all lengths up to n, the function returns the list of frequency tables. Each table provides insights into how frequently each character follows various preceding character sequences (contexts).
Summary
The overall intuition of the code is to analyze the character frequency relationships in a document. By creating frequency tables for 1-grams, 2-grams, up to n-grams, the function establishes a foundation for understanding how likely certain characters are to follow others based on their preceding contexts. This information can be invaluable for applications like predictive text input, character-based language models, and more, allowing for better contextual understanding of character sequences in a given language.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables\*\*

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:

1. **_What is your vocabulary in this case_**
   - Write it here
     {'b', 'c', 'a'}
2. **_Write down your probabillity table 1_**:

   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

     | $P(\odot)$ | Probability value |
     | ---------- | ----------------- |
     | $P(a)$     | $\frac{11}{20}$   |
     | $P(b)$     | $\frac{4}{20}$    |
     | $P(c)$     | $\frac{5}{20}$    |

3. **_Write down your probability table 2_**:

   - as in your probability table should look like (wait a second, you should know what I'm talking about)
     | $P(\odot)$ | Probability value |
     | ----------------- | ----------------- |
     | $P(a \mid ('a'))$ | $\frac{5}{11}$ |
     | $P(a \mid ('b'))$ | $\frac{2}{11}$ |
     | $P(a \mid ('c'))$ | $\frac{3}{11}$ |
     | $P(b \mid ('a'))$ | $\frac{3}{4}$ |
     | $P(b \mid ('c'))$ | $\frac{1}{4}$ |
     | $P(c \mid ('b'))$ | $\frac{2}{5}$ |
     | $P(c \mid ('a'))$ | $\frac{2}{5}$ |
     | $P(c \mid ('c'))$ | $\frac{1}{5}$ |

4. **_Write down your probability table 3_**:
   - You got this!
     | $P(\odot)$ | Probability value |
     | ------------------------- | ----------------- |
     | $P(b \mid ('a', 'a'))$ | $\frac{2}{5}$ |
     | $P(b \mid ('b', 'a'))$ | $\frac{1}{2}$ |
     | $P(b \mid ('a', 'c'))$ | $\frac{1}{2}$ |
     | $P(a \mid ('a', 'b'))$ | $\frac{1}{3}$ |
     | $P(a \mid ('b', 'c'))$ | $\frac{2}{2}$ |
     | $P(a \mid ('c', 'c'))$ | $\frac{1}{1}$ |
     | $P(a \mid ('c', 'a'))$ | $\frac{2}{3}$ |
     | $P(a \mid ('a', 'a'))$ | $\frac{1}{5}$ |
     | $P(a \mid ('c', 'b'))$ | $\frac{1}{1}$ |
     | $P(a \mid ('b', 'a'))$ | $\frac{1}{2}$ |
     | $P(c \mid ('a', 'b'))$ | $\frac{2}{3}$ |
     | $P(c \mid ('c', 'a'))$ | $\frac{1}{3}$ |
     | $P(c \mid ('a', 'c'))$ | $\frac{1}{2}$ |
     | $P(c \mid ('a', 'a'))$ | $\frac{1}{5}$ |

## `calculate_probability(sequence, char, tables)`

### Formula

- **_Write the formula for sequence likelihood as described in section 2_**
  $$P(X_1=a, X_2=a, X_3=xchar) = P(a) \cdot P(a \mid a) \cdot P(x_3 \mid a) = \frac{f(a)}{20} \cdot \frac{f(a,a)}{f(a)} \cdot \frac{f(a,x_3)}{f(a)} = frac{11}{20} \cdot \frac{5}{11} \cdot \frac{f(a,x_3)}{5}$$

### Code analysis

- **_Put the intuition of your code here_**
  Sequence Preparation:
  The function first appends the target character (`char`) to the end of the sequence. This forms a complete sequence that includes the character whose probability we wish to calculate.
  Length Validation:
  The length of the extended sequence is calculated.
  The function checks if this length is valid in relation to the number of frequency tables. If the length is zero or longer than the number of available tables, it returns a probability of `0.0`, since no valid computation can be performed.
  Total Count Calculation:
  It calculates the total count of all occurrences of character sequences that can be represented by the 1-gram (the first table in `tables`). This total count is needed to normalize the probability in subsequent calculations.
  Index for n-gram Table:
  The index for the frequency tables is determined by the current sequence length minus one (`n_table = seq_length - 1`), because tables are zero-indexed.
  Calculate Initial Probability:
  The function retrieves the count of the first character (the first character of `sequence`) from the 1-gram table (the first frequency table).
  This is used to calculate the initial probability of this character appearing in the defined context provided by the sequence.
  Iterative Probability Calculation:
  For each character in the sequence starting from the second character:
  The preceding characters (up to the current position) are converted into a tuple for dictionary lookup.
  The function retrieves the count of how many times the current character occurs given the preceding characters from the corresponding frequency table.
  If there are no counts corresponding to the current character or context, the function returns `0.0`, indicating that the desired character cannot follow the specified context based on the data.
  Otherwise, the probability is updated by multiplying the current probability with the ratio of the counts of the current character given the previous characters to the previously computed probability value.
  Return the Computed Probability:
  Finally, once the loop processes all characters in the sequence, the function returns the final computed probability.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- **_Calculate the following and show all the steps involved_**

1. $P(X_1=a, X_2=a, X_3=a) = 0.05$

   - _Show your work_

   $\frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5} = 0.05$

2. $P(X_1=a, X_2=a, X_3=b) = 0.1$
   - _Show your work_
     $\frac{11}{20} \cdot \frac{5}{11} \cdot \frac{2}{5} = 0.1$
3. $P(X_1=a, X_2=a, X_3=c) = 0.05$
   - _Show your work_
     $\frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5} = 0.05$

## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- **_Put the intuition of your code here_**

How It Works
Initialization:
The function initializes two variables: `max_prob` to a very low value (e.g., -100) and `max_l` as an empty string. These will be used to track the maximum probability found and the corresponding predicted character.
Iterate Over Vocabulary:
The function loops through each character (`l`) in the provided `vocabulary`, which represents all possible characters that could follow the given sequence.
Probability Calculation:
For each character `l`, it calculates the probability of that character following the given sequence using the `calculate_probability` function. This function returns the likelihood of `l` occurring right after the `sequence`, considering the statistical information in the n-gram frequency tables.
Finding Max Probability:
The function prints the character along with its calculated probability for debugging or informational purposes.
It then checks if the computed probability is greater than the current maximum probability stored in `max_prob`. If so, it updates `max_prob` to the new probability and sets `max_l` to the character `l`.
Return Predicted Character:
Once all characters in the vocabulary have been evaluated, the function returns `max_l`, which is the character that has the highest computed probability of following the given sequence.

### So what should be the next character in the sequence?

- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**
  - _Your answer_
    b, because it has the highest probability

## Experiment

- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

When trying it out with the war and peace document, I found that it worked well for n=5 and below. After that it started to get a little bit slower but still usable. By n=10 there were too many permutations and combinations too be efficient in computing.

<hr>

Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
