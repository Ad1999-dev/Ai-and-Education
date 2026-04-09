[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/i8wht-pB)

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

# Submission Instructions

You are to include **2 files in a single Gradescope submission**: a **PDF of your Report Section** and your **NgramAutocomplete.py**.

How to generate a pdf of your Report Section:

- On your Github repository after finishing the assignment, click on readme.md to open the markdown preview.
- Use your browser 's "Print to PDF" feature to save your PDF.

Please submit to Assignment 6 N-Gram Complete on Gradecsope.

# A Reports section

## 383GPT

Did you use 383GPT at all for this assignment (yes/no)?

Yes

## Late Days

How many late days are you using for this assignment?

0

## `create_frequency_tables(document, n)`

### Code analysis

- **_Put the intuition of your code here_**

  The `create_frequency_tables` function initializes a list containing `n` empty dictionaries, where each dictionary corresponds to a frequency table for sequences of a specific length (from 1 to `n`). It then iterates through the input `document` character by character. For each position `i` in the document, it attempts to extract sequences of lengths 1 up to `n` that _end_ at position `i`. If a valid sequence is extracted (i.e., it doesn't go beyond the start of the document), the function finds the appropriate dictionary in the list based on the sequence length and increments the count for that specific sequence within that dictionary. If the sequence wasn't previously seen, it's added to the dictionary with a count of 1. This process populates all `n` tables with the frequencies of all occurring n-grams up to length `n`.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables\*\*

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, <redacted> the following:

1.  **_What is your vocabulary in this case_**

    - Write it here

      `{a, b, c}`

2.  **_Write down your probabillity table 1_**:

    - as in $P(a), P(b), \dots$
    - For table 1, as in your probability table should look like this:

      Based on frequencies: `f(a)=11`, `f(b)=4`, `f(c)=5`. Total = 20.
      $P(x) = f(x) / \text{total\_chars}$

      | $P(\odot)$ | Probability value |
      | ---------- | ----------------- |
      | $P(a)$     | $11/20 = 0.55$    |
      | $P(b)$     | $4/20 = 0.20$     |
      | $P(c)$     | $5/20 = 0.25$     |

3.  **_Write down your probability table 2_**:

    - as in your probability table should look like (wait a second, you should know what I'm talking about)

      Based on $P(y | x) = f(xy) / f(x)$, where $f(x)$ is the count of `x` appearing as context (i.e., followed by another character).

      Frequencies: `f(aa)=5`, `f(ab)=3`, `f(ac)=2`, `f(ba)=2`, `f(bb)=0`, `f(bc)=2`, `f(ca)=3`, `f(cb)=1`, `f(cc)=1`.

      Context counts: `f(a)=10`, `f(b)=4`, `f(c)=5`.

      | $P(\odot)$    | Probability value | Calculation  |
      | ------------- | ----------------- | ------------ |
      | $P(a \mid a)$ | $5/10 = 0.5$      | $f(aa)/f(a)$ |
      | $P(b \mid a)$ | $3/10 = 0.3$      | $f(ab)/f(a)$ |
      | $P(c \mid a)$ | $2/10 = 0.2$      | $f(ac)/f(a)$ |
      | $P(a \mid b)$ | $2/4 = 0.5$       | $f(ba)/f(b)$ |
      | $P(b \mid b)$ | $0/4 = 0.0$       | $f(bb)/f(b)$ |
      | $P(c \mid b)$ | $2/4 = 0.5$       | $f(bc)/f(b)$ |
      | $P(a \mid c)$ | $3/5 = 0.6$       | $f(ca)/f(c)$ |
      | $P(b \mid c)$ | $1/5 = 0.2$       | $f(cb)/f(c)$ |
      | $P(c \mid c)$ | $1/5 = 0.2$       | $f(cc)/f(c)$ |

4.  **_Write down your probability table 3_**:

    - You got this!

      Based on $P(z | xy) = f(xyz) / f(xy)$, where $f(xy)$ is the count of `xy` appearing as context.

      Trigram Frequencies: `f(aaa)=1`, `f(aab)=2`, `f(aac)=1`, `f(aba)=1`, `f(abc)=2`, `f(bab)=1`, `f(baa)=1`, `f(bca)=2`, `f(cac)=1`, `f(caa)=2`, `f(acc)=1`, `f(acb)=1`, `f(cca)=1`, `f(cba)=1`. Others are 0.

      Bigram Context Counts: `f(aa)=4`, `f(ab)=3`, `f(ba)=2`, `f(bc)=2`, `f(ca)=3`, `f(ac)=2`, `f(cc)=1`, `f(cb)=1`.

      | $P(\odot)$     | Probability value  | Calculation    |
      | -------------- | ------------------ | -------------- |
      | $P(a \mid aa)$ | $1/4 = 0.25$       | $f(aaa)/f(aa)$ |
      | $P(b \mid aa)$ | $2/4 = 0.5$        | $f(aab)/f(aa)$ |
      | $P(c \mid aa)$ | $1/4 = 0.25$       | $f(aac)/f(aa)$ |
      | $P(a \mid ab)$ | $1/3 \approx 0.33$ | $f(aba)/f(ab)$ |
      | $P(b \mid ab)$ | $0/3 = 0.0$        | $f(abb)/f(ab)$ |
      | $P(c \mid ab)$ | $2/3 \approx 0.67$ | $f(abc)/f(ab)$ |
      | $P(a \mid ba)$ | $1/2 = 0.5$        | $f(baa)/f(ba)$ |
      | $P(b \mid ba)$ | $1/2 = 0.5$        | $f(bab)/f(ba)$ |
      | $P(c \mid ba)$ | $0/2 = 0.0$        | $f(bac)/f(ba)$ |
      | $P(a \mid bc)$ | $2/2 = 1.0$        | $f(bca)/f(bc)$ |
      | $P(b \mid bc)$ | $0/2 = 0.0$        | $f(bcb)/f(bc)$ |
      | $P(c \mid bc)$ | $0/2 = 0.0$        | $f(bcc)/f(bc)$ |
      | $P(a \mid ca)$ | $2/3 \approx 0.67$ | $f(caa)/f(ca)$ |
      | $P(b \mid ca)$ | $0/3 = 0.0$        | $f(cab)/f(ca)$ |
      | $P(c \mid ca)$ | $1/3 \approx 0.33$ | $f(cac)/f(ca)$ |
      | $P(a \mid ac)$ | $0/2 = 0.0$        | $f(aca)/f(ac)$ |
      | $P(b \mid ac)$ | $1/2 = 0.5$        | $f(acb)/f(ac)$ |
      | $P(c \mid ac)$ | $1/2 = 0.5$        | $f(acc)/f(ac)$ |
      | $P(a \mid cc)$ | $1/1 = 1.0$        | $f(cca)/f(cc)$ |
      | $P(b \mid cc)$ | $0/1 = 0.0$        | $f(ccb)/f(cc)$ |
      | $P(c \mid cc)$ | $0/1 = 0.0$        | $f(ccc)/f(cc)$ |
      | $P(a \mid cb)$ | $1/1 = 1.0$        | $f(cba)/f(cb)$ |
      | $P(b \mid cb)$ | $0/1 = 0.0$        | $f(cbb)/f(cb)$ |
      | $P(c \mid cb)$ | $0/1 = 0.0$        | $f(cbc)/f(cb)$ |

## `calculate_probability(sequence, char, tables)`

### Formula

- **_Write the formula for sequence likelihood as described in section 2_**

  For an n-gram model, the probability of the next character $x_{t+1}$ given the preceding sequence $x_1, \dots, x_t$ is approximated by conditioning only on the last $n-1$ characters:

  $$P(x_{t+1} \mid x_1, \dots, x_t) \approx P(x_{t+1} \mid x_{t-n+2}, \dots, x_t)$$

  This conditional probability is estimated using frequencies from the corpus:

  $$P(x_{t+1} \mid x_{t-n+2}, \dots, x_t) = \frac{f(x_{t-n+2}, \dots, x_t, x_{t+1})}{f(x_{t-n+2}, \dots, x_t)}$$

  Where $f(\dots)$ denotes the frequency (count) of the sequence in the training document. If the context $x_{t-n+2}, \dots, x_t$ has never appeared (denominator is 0), the probability is 0. For the special case of predicting the first character (empty context) or when $n=1$, $P(x_1) = f(x_1) / \text{total\_chars}$.

### Code analysis

- **_Put the intuition of your code here_**

  The `calculate_probability` function implements the n-gram conditional probability formula. First, it determines the relevant context by taking the last `n-1` characters of the input `sequence`. If the sequence is shorter than `n-1` characters, or if `n=1`, the whole sequence (or an empty string for `n=1`) is used as context. It then forms the `target_sequence` by appending the `char` to the `context_sequence`. It retrieves the frequency count of the `target_sequence` (numerator) from the appropriate table (based on target sequence length) and the frequency count of the `context_sequence` (denominator) from its corresponding table. A special case handles the unigram probability (empty context) where the denominator is the total character count. Finally, it calculates the probability by dividing the numerator frequency by the denominator frequency, returning 0.0 if the denominator is zero to avoid division errors.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- **_Calculate the following and show all the steps involved_**

  We use the n-gram approximation for joint probability with $n=3$:
  $P(x_1, x_2, x_3) \approx P(x_1) \times P(x_2 \mid x_1) \times P(x_3 \mid x_1, x_2)$

1.  $P(X_1=a, X_2=a, X_3=a)$

    - _Show your work_

      $P(a, a, a) \approx P(a) \times P(a \mid a) \times P(a \mid aa)$
      $= (11/20) \times (5/10) \times (1/4)$
      $= (11/20) \times (1/2) \times (1/4)$
      $= 11 / 160 \approx 0.06875$

2.  $P(X_1=a, X_2=a, X_3=b)$

    - _Show your work_

      $P(a, a, b) \approx P(a) \times P(a \mid a) \times P(b \mid aa)$
      $= (11/20) \times (5/10) \times (2/4)$
      $= (11/20) \times (1/2) \times (1/2)$
      $= 11 / 80 = 22 / 160 \approx 0.1375$

3.  $P(X_1=a, X_2=a, X_3=c)$

    - _Show your work_

      $P(a, a, c) \approx P(a) \times P(a \mid a) \times P(c \mid aa)$
      $= (11/20) \times (5/10) \times (1/4)$
      $= (11/20) \times (1/2) \times (1/4)$
      $= 11 / 160 \approx 0.06875$

## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- **_Put the intuition of your code here_**

  The `predict_next_char` function aims to find the most likely character to follow the given input `sequence`. It iterates through every character (`candidate_char`) in the provided `vocabulary`. For each `candidate_char`, it calls the `calculate_probability` function to get the conditional probability $P(\text{candidate\_char} \mid \text{sequence})$ based on the n-gram model (`tables`). It keeps track of the character that yields the highest probability encountered so far. After checking all characters in the vocabulary, it returns the character associated with the maximum probability. If all probabilities are zero, it returns a default character (the first character in the sorted vocabulary) to ensure a prediction is always made.

### So what should be the next character in the sequence?

- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**

  - _Your answer_

    We need to find the character `x` that maximizes $P(x \mid \text{sequence})$ for the sequence `"aa"`. Using the trigram model (n=3), the relevant context is `"aa"`. We look at the probabilities calculated in Table 3 for the context `aa`:

  - $P(a \mid aa) = 1/4 = 0.25$
  - $P(b \mid aa) = 2/4 = 0.5$
  - $P(c \mid aa) = 1/4 = 0.25$

  The character with the highest probability is `b` (0.5). Therefore, the predicted next character is `b`.

## Experiment

- Experiment with the given corpus files and varying values of n. <redacted> any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

When experimenting with "War and Peace" and "Alice's Adventures in Wonderland", I found that the larger corpus of "War and Peace" generally produces more robust predictions due to its greater vocabulary and contextual diversity, particularly for higher values of n. However, this comes with significant computational trade-offs. Processing "Alice" remains efficient up to approximately n=6 or n=7, while "War and Peace" becomes noticeably slower around n=5, with performance degrading exponentially as n increases. Memory constraints become a limiting factor with large n values as the number of unique n-grams grows exponentially, potentially exceeding available RAM. In general, n=4 to n=6 is the optimal balance between prediction quality and computational efficiency, with the specific sweet spot depending on the corpus size.

<hr>

Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
