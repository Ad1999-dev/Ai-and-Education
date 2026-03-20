[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/i8wht-pB)
# ***Bayes Complete***: Sentence Autocomplete using N-Gram Language Models

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
|---------|-----------|
| f(a)    |           |
| f(b)    |           |
| f(c)    |           |
| f(d)    |           |

### Table 2: Bigram Frequencies

| Bigram   | Frequency |
|----------|-----------|
| f(a, a) |           |
| f(a, b) |           |
| f(a, c) |           |
| f(a, d) |           |
| f(b, a) |           |
| f(b, b) |           |
| f(b, c) |           |
| f(b, d) |           |
| ...      |           |

### Table 3: Trigram Frequencies

| Trigram    | Frequency |
|------------|-----------|
| f(a, a, a) |          |
| f(a, a, b) |          |
| f(a, a, c) |          |
| f(a, a, d) |          |
| f(a, b, a) |          |
| f(a, b, b) |          |
| ...        |          |
    
  
And so on with increasing sizes of n.

### 2. **Computing Joint Probabilities for a Language Model**

In general, Bayesian Networks are used to visually represent the dependencies (edges) between distinct random varaibles (nodes) in a large joint distribution. 

In the case of a language model, each node in the network corresponds to a character in the sequence, and edges represent the conditional dependencies between them.

For a character sequence of length 4 a bayesian network for our the full joint distribution of 4 letter sequences would look as follows.

![image1](https://github.com/user-attachments/assets/e1924619-a2ff-4ecb-8e78-eb84dcac0800)



Where $X_1$ is a random variable that maps to the character found at position 1 in a character sequence, $X_2$ maps to the character at position 2, and so on.

This makes clear how the chain rule can be applied to expand the full joint form of a probability distribution.

$$P(X_1=x_1, X_2=x_2, X_3=x_3, X_4=x_4) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdot P(x_4 \mid x_1, x_2, x_3)$$

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

![image2](https://github.com/user-attachments/assets/b7188a62-772f-44aa-b714-ba4b5b565760)


Putting this network in terms of computations via our frequency tables is now slightly different as we now have to consider the ratio for each term

$$P(X_1=x_1, X_2=x_2, X_3=x_3, X_4=x_4) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_2) \cdot P(x_4 \mid x_3) = \frac{f(x_1)}{size(C)} \cdot \frac{f(x_1,x_2)}{f(x_1)} \cdot \frac{f(x_2,x_3)}{f(x_2)} \cdot \frac{f(x_3,x_4)}{f(x_3)}$$

Where `size(C)` is the total number of characters in the corpus. Consider how this generalizes to an arbitrary n-gram model for any `n`, this will be the core of your implementation. Write this formula in your report.

## Starter Code Overview

The project starter code is structured across three main Python files:

1. **NgramAutocomplete.py**: This is where the main logic of the autocomplete model is implemented. You are expected to complete three functions in this file: `create_frequency_tables()`, `calculate_probability()`, and `predict_next_char()`.

2. **main.py**: This file provides the user interface and controls the flow of the program. It initializes the model, takes user inputs, and runs the character prediction process iteratively. You may modify this file to test their code, but no modifications are required to complete the project.

3. **utilities.py**: This file includes helper functions that facilitate the program, such as reading and preprocessing the training document. No modifications are needed in this file.

## TODOs

***NgramAutocomplete.py*** is the core file where you will change in this project. Each function here builds upon each other to create a probabilistic model for predicting the next character in a sequence.

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

#### **Purpose**
The `create_frequency_tables` function is responsible for building the core statistical foundation of an n-gram language model. It processes a text document to produce frequency tables for all substrings of lengths 1 through `n`, enabling probabilistic prediction of character sequences.

---

#### **How It Works**

1. **Initialization**
   - Creates an empty list called `tables` to store frequency dictionaries for each gram length (1 to `n`).

2. **Sliding Window**
   - For each gram length `k` (from 1 to `n`), a sliding window of size `k` moves through the document.
   - At each step, the substring `document[i : i + k]` is extracted.

3. **Counting Occurrences**
   - Each substring is added to a dictionary.
   - If it already exists, its count is incremented.
   - Otherwise, it is initialized with a count of `1`.

4. **Storing Results**
   - After processing all substrings of length `k`, the dictionary is appended to the `tables` list.
   - The result is a list of `n` frequency tables, where each table corresponds to a different n-gram size.

---

#### **Time Complexity**
- For each `k` from `1` to `n`, the function loops approximately `L - k` times, where `L` is the length of the document.
- Overall time complexity is approximately: O(n x L)
- This makes the function efficient for large text inputs and scalable with respect to both the document size and n-gram depth.

---

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
   - Vocabulary = { a, b, c }
2. ***Write down your probabillity table 1***:
   - Table 1:

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a)$ | $\frac{11}{20}$   |
        | $P(b)$     | $\frac{4}{20}$   |
        | $P(c)$     | $\frac{5}{20}$   |
 
1. ***Write down your probability table 2***:
   - Table 2:

        | $P(x \mid y)$     | Probability value   |
        | ----------------- | ------------------ |
        | $P(a \mid a)$     | $\frac{5}{10}$      |
        | $P(b \mid a)$     | $\frac{3}{10}$      |
        | $P(c \mid a)$     | $\frac{2}{10}$      |
        | $P(a \mid b)$     | $\frac{2}{4}$       |
        | $P(b \mid b)$     | $0$                 |
        | $P(c \mid b)$     | $\frac{2}{4}$       |
        | $P(a \mid c)$     | $\frac{3}{5}$       |
        | $P(b \mid c)$     | $\frac{1}{5}$       |
        | $P(c \mid c)$     | $\frac{1}{5}$       |

2. ***Write down your probability table 3***:
   - Table 3:

        | $P(x \mid yz)$     | Probability value   |
        | ------------------ | ------------------ |
        | $P(a \mid aa)$     | $\frac{1}{5}$       |
        | $P(b \mid aa)$     | $\frac{2}{5}$       |
        | $P(c \mid aa)$     | $\frac{1}{5}$       |
        | $P(a \mid ab)$     | $\frac{1}{3}$       |
        | $P(b \mid ab)$     | $0$                 |
        | $P(c \mid ab)$     | $\frac{2}{3}$       |
        | $P(a \mid ba)$     | $\frac{1}{2}$       |
        | $P(b \mid ba)$     | $\frac{1}{2}$       |
        | $P(c \mid ba)$     | $0$                 |
        | $P(a \mid bc)$     | $1$                 |
        | $P(b \mid bc)$     | $0$                 |
        | $P(c \mid bc)$     | $0$                 |
        | $P(a \mid ca)$     | $\frac{2}{3}$       |
        | $P(b \mid ca)$     | $0$                 |
        | $P(c \mid ca)$     | $\frac{1}{3}$       |
        | $P(a \mid cc)$     | $1$                 |
        | $P(b \mid cc)$     | $0$                 |
        | $P(c \mid cc)$     | $0$                 |
        | $P(a \mid ac)$     | $0$                 |
        | $P(b \mid ac)$     | $\frac{1}{2}$       |
        | $P(c \mid ac)$     | $\frac{1}{2}$       |
        | $P(a \mid cb)$     | $1$                 |
        | $P(b \mid cb)$     | $0$                 |
        | $P(c \mid cb)$     | $0$                 |




## `calculate_probability(sequence, char, tables)`

### Formula
The probability of a character `char` given a preceding sequence of characters `sequence` is estimated using the n-gram frequency tables as:

```math
P(char \mid sequence) = \frac{\text{count(sequence + char)}}{\text{count(sequence)}}
```

---

### Code analysis

The `calculate_probability` function is responsible for computing the likelihood that a specific character follows a given character sequence, using the frequency tables built during training. It works by:
- Trimming the input sequence to match the n-gram depth supported by the tables.
- Concatenating the sequence with the character and looking up its count in the corresponding table.
- Dividing that count by the count of the sequence alone (to get conditional probability).
- If the sequence is not found (i.e., zero denominator), the function can return 0 or back off to a shorter context.

---

### Your Calculations

We will use our probability tables above and the formula:

```math
P(X_1, X_2, X_3) = P(X_1) \cdot P(X_2 \mid X_1) \cdot P(X_3 \mid X_1, X_2)
```

---
- ***Calculate the following and show all the steps involved***

#### 1. $P(X_1 = a, X_2 = a, X_3 = a)$

**Step 1:**  
- $P(a) = \frac{11}{20}$

**Step 2:**  
- $P(a \mid a) = \frac{5}{10}$

**Step 3:**  
- $P(a \mid aa) = \frac{1}{5}$

**Final Calculation:**
```math
P(aaa) = \frac{11}{20} \cdot \frac{5}{10} \cdot \frac{1}{5} = \frac{55}{1000} = 0.055
```

---

#### 2. $P(X_1 = a, X_2 = a, X_3 = b)$

**Step 1:**  
- $P(a) = \frac{11}{20}$

**Step 2:**  
- $P(a \mid a) = \frac{5}{10}$

**Step 3:**  
- $P(b \mid aa) = \frac{2}{5}$

**Final Calculation:**
```math
P(aab) = \frac{11}{20} \cdot \frac{5}{10} \cdot \frac{2}{5} = \frac{110}{1000} = 0.11
```

---

#### 3. $P(X_1 = a, X_2 = a, X_3 = c)$

**Step 1:**  
- $P(a) = \frac{11}{20}$

**Step 2:**  
- $P(a \mid a) = \frac{5}{10}$

**Step 3:**  
- $P(c \mid aa) = \frac{1}{5}$

**Final Calculation:**
```math
P(aac) = \frac{11}{20} \cdot \frac{5}{10} \cdot \frac{1}{5} = \frac{55}{1000} = 0.055
```


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

The `predict_next_char` function determines the most likely next character to follow a given input sequence. It does this by:

1. Iterating over every character in the vocabulary.
2. For each character, it calls `calculate_probability(sequence, char, tables)` to compute the probability that the character follows the sequence.
3. It keeps track of the character with the highest probability and returns it as the prediction.

The function essentially performs a search over the probability distribution of possible next characters and selects the **argmax** (i.e., the character with the highest likelihood). This is the deterministic version of prediction — more advanced implementations may sample from the distribution for randomness, but this version always picks the most probable character.

---

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**

Based on the previously computed probability distribution for `P(X₁=a, X₂=a, X₃=?)`, we have:

- $P(aaa) = 0.055$
- $P(aab) = 0.11$
- $P(aac) = 0.055$

Since `aab` has the highest probability among all completions of the sequence `"aa"`, the most likely next character is:

```
**b**
```

---
 
## Experiment
While experimenting with different input sequences, corpora, and n-gram values, I noticed that recursive repetition was a recurring problem, especially at the extremes of `n`.

---

### Recursive Loops with Common Phrases

When I used `n = 6` with the input:

```
how are you doing on this fine
```

in Alice’s Adventures in Wonderland, the model output:

```
how are you doing on this fine day! and the project gutenberg™ electronic work is for the project gutenberg™ electronic work is fo
```

This is a classic case of an n-gram model getting stuck in a high-frequency phrase. Phrases like `"Project Gutenberg"` (73 occurrences) and `"electronic work"` (27 occurrences) dominate the sequence probabilities, especially at higher n-gram sizes where the model starts memorizing entire sequences rather than generalizing.

While increasing `n` (e.g., to 20 or 40) helped reduce this problem by including more context, it also made the model much slower, with noticeable delays at `n = 40` (about 30–40 seconds for a 100-character prediction).

---

### Looping from Small Context Windows

Another issue arose at smaller values of `n`, particularly with the `warandpeace.txt` corpus. When using `n = 3` or `n = 4`, the model often looped through short, common word sequences. For example:

```
Enter the number of grams (n): 3  
Enter an initial sequence: what happens with a small  
Enter the length of completion (k): 50
```

Output:

```
what happens with a small the and the and the and the and the and the and t
```

Here, common words like `"and"` and `"the"` feed into each other: `"and"` is frequently followed by `"the"`, and `"the"` (ending in `"e"`) is frequently followed by `"a"`, which begins `"and"` — creating an infinite loop. This happens because shorter n-grams limit the diversity of plausible next characters, making loops far more likely.

---

### Reflection & Potential Fixes

In summary, small values of `n` tend to make the model overgeneralize, which often leads to short, repetitive cycles involving common function words such as "and" or "the." Because the model is working with such limited context, it frequently falls into loops of predictable and high-frequency word pairings. On the other hand, large values of `n` cause the model to overfit, effectively memorizing long, specific sequences from the training corpus. This often results in the reproduction of boilerplate text, such as license metadata, or getting stuck in extended loops that are difficult to escape. Overall, there is a clear tradeoff between generalization and specificity, and neither extreme performs optimally across all inputs.

To address these limitations, several improvements could be made to the model. For example, using large n-gram values while applying decaying weights to the context — giving more influence to recent characters — could help balance specificity with flexibility. Introducing sampling with randomness instead of always selecting the most probable character would also reduce the likelihood of deterministic loops. Lastly, removing or filtering out boilerplate metadata such as the Project Gutenberg license text during preprocessing would reduce the dominance of non-literary patterns in the generated output. These changes could make the model more dynamic, reduce repetition, and better simulate natural language.

