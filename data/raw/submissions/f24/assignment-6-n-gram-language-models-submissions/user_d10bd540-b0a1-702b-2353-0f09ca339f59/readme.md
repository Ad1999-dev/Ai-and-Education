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

Consider that our vocabulary just consists of 4 letters, $\{a, b, c}$, for simplicity.

### Table 1: Unigram Frequencies

| Unigram | Frequency |
| ------- | --------- |
| f(a)    |           |
| f(b)    |           |
| f(c)    |           |

### Table 2: Bigram Frequencies

| Bigram  | Frequency |
| ------- | --------- |
| f(a, a) |           |
| f(a, b) |           |
| f(a, c) |           |
| f(b, b) |           |
| f(b, a) |           |
| f(b, c) |           |
| f(c, c) |           |
| f(c, a) |           |
| f(c, b) |           |

### Table 3: Trigram Frequencies

| Trigram    | Frequency |
| ---------- | --------- |
| f(a, a, a) |           |
| f(a, a, b) |           |
| f(a, a, c) |           |
| f(a, b, a) |           |
| f(a, b, b) |           |
| f(a, b, c) |           |
| f(a, c, b) |           |
| f(a, c, c) |           |
| f(b, a, a) |           |
| f(b, a, b) |           |
| f(b, a, c) |           |
| f(b, b, a) |           |
| f(b, b, b) |           |
| f(b, b, c) |           |
| f(b, c, a) |           |

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

Did you use 383GPT at all for this assignment (yes/no)?
Yes, I did use 383GPT for this assignment.

## `create_frequency_tables(document, n)`

### Code analysis

The create_frequency_tables function generates a list of frequency tables, each corresponding to n-grams of increasing sizes (from unigrams to n-grams). It does this by iterating through the input document, constructing n-grams of size 1 to n at each index, and updating the corresponding frequency table. Each n-gram is stored as a key in a dictionary, with its count as the value. This function allows for efficiently storing how often a character sequence appears in the training corpus, which is essential for calculating probabilities in an n-gram language model. The use of defaultdict(int) ensures that counts start at zero for unseen n-grams, simplifying updates.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables\*\*

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:

---

## Probability Tables

Let’s reconcile the missing entries from the provided table and adjust the **Trigram Probability Table** accordingly. After comparison, the issue appears to be that some trigram contexts and probabilities are not fully captured in the table generated earlier.

Here is the **updated and corrected probability table**, ensuring it matches the provided reference and accounts for all contexts.

---

### Table 1: Unigram Probabilities
- P(a): 11/20  
- P(b): 4/20  
- P(c): 5/20  

---

### Table 2: Bigram Probabilities
- P(a | a): 5/19  
- P(b | a): 3/19  
- P(c | a): 2/19  
- P(a | b): 2/19  
- P(b | b): 0/19  
- P(c | b): 2/19  
- P(a | c): 3/19  
- P(b | c): 1/19  
- P(c | c): 1/19  

---

### Table 3: Trigram Probabilities
- P(a | a, a): 1/18  
- P(b | a, a): 2/18  
- P(c | a, a): 1/18  
- P(a | a, b): 1/18  
- P(b | a, b): 0/18  
- P(c | a, b): 2/18  
- P(a | a, c): 0/18  
- P(b | a, c): 1/18  
- P(c | a, c): 1/18  
- P(a | b, a): 1/18  
- P(b | b, a): 1/18  
- P(c | b, a): 0/18  
- P(a | b, b): 0/18  
- P(b | b, b): 0/18  
- P(c | b, b): 0/18  
- P(a | b, c): 2/18  
- P(b | b, c): 0/18  
- P(c | b, c): 0/18  
- P(a | c, a): 2/18  
- P(b | c, a): 0/18  
- P(c | c, a): 1/18  
- P(a | c, b): 0/18  
- P(b | c, b): 1/18  
- P(c | c, b): 0/18  
- P(a | c, c): 0/18  
- P(b | c, c): 0/18  
- P(c | c, c): 1/18  

---

## `calculate_probability(sequence, char, tables)`

### Formula

The likelihood of a sequence is calculated as follows:

P(x_t+1 | x_1, x_2, ..., x_t) = f(x_1, x_2, ..., x_t, x_t+1) / f(x_1, x_2, ..., x_t)

Where:

- x_1, x_2, ..., x_t: The sequence of characters (context).
- x_t+1: The next character whose probability is being calculated.
- f(x_1, x_2, ..., x_t, x_t+1): The frequency of the sequence (x_1, x_2, ..., x_t, x_t+1) in the training data.
- f(x_1, x_2, ..., x_t): The frequency of the sequence (x_1, x_2, ..., x_t) in the training data.

---

### Smoothing with Laplace Smoothing

To handle unseen sequences and avoid zero probabilities, **Laplace smoothing** is applied:

P(x_t+1 | x_1, x_2, ..., x_t) = (f(x_1, x_2, ..., x_t, x_t+1) + α) / (Σ_x_t+1 ∈ V (f(x_1, x_2, ..., x_t, x_t+1) + α))

Where:

- α: The smoothing parameter (e.g., 0.01) to account for unseen n-grams.
- V: The vocabulary set of possible characters.

---

### Code analysis

This function computes the conditional probability of a character (char) occurring immediately after a given sequence using the precomputed frequency tables. It employs smoothing (Laplace smoothing) with the parameter alpha to handle unseen n-grams by assigning them a small probability. The sequence is truncated to the appropriate size (n-1) based on the model order to align with the corresponding frequency table. The probability is calculated as the smoothed count of the target n-gram divided by the smoothed sum of all n-grams sharing the same prefix (sequence). This function avoids division by zero by checking if the denominator is zero and uses smoothing to ensure probabilities are non-zero for unseen combinations, improving model robustness.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary

## Probability Calculations

### P(X1 = a, X2 = a, X3 = a)

**Unigram Probability**:  
P(X1 = a) = Frequency of a / Total characters = 11 / 20 = 0.55

**Bigram Probability**:  
P(X2 = a | X1 = a) = Frequency of (a, a) / Frequency of a = 5 / 11 ≈ 0.4545

**Trigram Probability**:  
P(X3 = a | X1 = a, X2 = a) = Frequency of (a, a, a) / Frequency of (a, a) = 1 / 5 = 0.2

**Combined Probability**:  
P(X1 = a, X2 = a, X3 = a) = P(X1 = a) * P(X2 = a | X1 = a) * P(X3 = a | X1 = a, X2 = a)  
P(X1 = a, X2 = a, X3 = a) = 0.55 * 0.4545 * 0.2 ≈ 0.05

---

### P(X1 = a, X2 = a, X3 = b)

**Unigram Probability**:  
P(X1 = a) = 0.55

**Bigram Probability**:  
P(X2 = a | X1 = a) = 0.4545

**Trigram Probability**:  
P(X3 = b | X1 = a, X2 = a) = Frequency of (a, a, b) / Frequency of (a, a) = 2 / 5 = 0.4

**Combined Probability**:  
P(X1 = a, X2 = a, X3 = b) = P(X1 = a) * P(X2 = a | X1 = a) * P(X3 = b | X1 = a, X2 = a)  
P(X1 = a, X2 = a, X3 = b) = 0.55 * 0.4545 * 0.4 ≈ 0.1

---

### P(X1 = a, X2 = a, X3 = c)

**Unigram Probability**:  
P(X1 = a) = 0.55

**Bigram Probability**:  
P(X2 = a | X1 = a) = 0.4545

**Trigram Probability**:  
P(X3 = c | X1 = a, X2 = a) = Frequency of (a, a, c) / Frequency of (a, a) = 1 / 5 = 0.2

**Combined Probability**:  
P(X1 = a, X2 = a, X3 = c) = P(X1 = a) * P(X2 = a | X1 = a) * P(X3 = c | X1 = a, X2 = a)  
P(X1 = a, X2 = a, X3 = c) = 0.55 * 0.4545 * 0.2 ≈ 0.05

---

### Final Results
- P(X1 = a, X2 = a, X3 = a) ≈ 0.05  
- P(X1 = a, X2 = a, X3 = b) ≈ 0.1  
- P(X1 = a, X2 = a, X3 = c) ≈ 0.05  

---

## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

The predict_next_char function predicts the most likely next character for a given input sequence by computing the probability of each possible character in the vocabulary using the calculate_probability function. It iterates through the vocabulary, calculating the conditional probability for each character given the sequence and stores these in a dictionary. If all probabilities are zero (e.g., due to an unseen sequence), the function defaults to a random choice from the vocabulary to ensure a fallback mechanism. Otherwise, it normalizes the probabilities and selects the character with the highest probability. This function combines statistical predictions with a fallback strategy, ensuring functionality even in edge cases.

### So what should be the next character in the sequence?

---

Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?

Based on the given probability tables, the most likely next character in the sequence `aababcaccaaacbaabcaa` is determined using the trigram probabilities, as they provide the most specific context. The last two characters of the sequence are `aa`. From the trigram table, the probabilities for the next character given this context are:  
- P(a | a, a) = 1/18  
- P(b | a, a) = 2/18  
- P(c | a, a) = 1/18  

Among these, P(b | a, a) has the highest probability (2/18), making `b` the most likely next character. Therefore, the predicted next character for the sequence `aababcaccaaacbaabcaa` is `b`.

----

## Experiment

- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

### Findings from Experiments with Different Corpora and \( n \)-Values

When experimenting with the corpus "War and Peace," which is a large and complex text, smaller \( n \)-values (e.g., \( n = 4 \) or \( n = 5 \)) produced meaningful and coherent completions. For example, starting with the sequence `their`, the model predicted extensions like `their she was`, demonstrating that the n-gram model could effectively capture local patterns in the text. However, as \( n \) increased to higher values such as \( n = 14 \) or \( n = 18 \), the process became significantly slower, and in some cases, the calculations for frequency tables or the predictions themselves resulted in timeouts. This is expected because higher \( n \)-values require more complex computations and larger frequency tables, which grow exponentially with \( n \).
On the other hand, a smaller corpus like "Alice's Adventures in Wonderland" allowed for much faster calculations, even with larger \( n \)-values. For instance, the program successfully handled values up to \( n = 17 \), while still maintaining reasonable speed and coherence in predictions. This is likely because smaller corpora have fewer unique n-grams, leading to smaller frequency tables and reduced computation times.

<hr>

Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
