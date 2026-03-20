[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bx7CmlmG)
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


# A Reports section

## 383GPT
Did you use 383GPT at all for this assignment (yes/no)?

Yes, I used it to understand what example usages of each method would look like and to understand the intuition behind the logic.

## `create_frequency_tables(document, n)`

### Code analysis

- ***Put the intuition of your code here***

    - Constructs frequency tables for n-grams from the input text, capturing how often each character follows a given sequence of previous characters.

    - Uses a nested dictionary to store counts:
        - Outer key: context (previous characters).
        - Inner key: next character, with its count.
    
    - Iterates through the document to populate these counts for n-grams of size 1 to `n`, creating a statistical model of character relationships.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
   - \{ a, b, c \}
2. ***Write down your probabillity table 1***:
   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a)$ | $\frac{11}{20}$ |
        | $P(b)$ | $\frac{4}{20} = \frac{1}{5}$ |
        | $P(c)$ | $\frac{5}{20} = \frac{1}{4}$ |
 
3. ***Write down your probability table 2***:
   - as in your probability table should look like (wait a second, you should know what I'm talking about) ##aababcaccaaacbaabcaa

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a)$ | $\frac{5}{11}$ |
        | $P(b \mid a)$ | $\frac{3}{11}$ |
        | $P(c \mid a)$ | $\frac{2}{11}$ |
        | $P(NULL \mid a)$ | $\frac{1}{11}$ |
        | $P(a \mid b)$ | $\frac{1}{2}$ |
        | $P(c \mid b)$ | $\frac{1}{2}$ |
        | $P(b \mid b)$ | $0$ |
        | $P(a \mid c)$ | $\frac{3}{5}$ |
        | $P(b \mid c)$ | $\frac{1}{5}$ |
        | $P(c \mid c)$ | $\frac{1}{5}$ |
    - Note: included $P(NULL\mid a)$ because the string ends with 'a' so the char after that is empty or null

4. ***Write down your probability table 3***:
   - You got this!

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a, a)$ | $\frac{1}{5}$ |
        | $P(b \mid a, a)$ | $\frac{2}{5}$ |
        | $P(c \mid a, a)$ | $\frac{1}{5}$ |
        | $P(NULL \mid a, a)$ | $\frac{1}{5}$ |
        | $P(a \mid a, b)$ | $\frac{1}{3}$ |
        | $P(b \mid a, b)$ | $0$ |
        | $P(c \mid a, b)$ | $\frac{2}{3}$ |
        | $P(a \mid a, c)$ | $0$ |
        | $P(b \mid a, c)$ | $\frac{1}{2}$ |
        | $P(c \mid a, c)$ | $\frac{1}{2}$ |
        | $P(a \mid b, a)$ | $\frac{1}{2}$ |
        | $P(b \mid b, a)$ | $\frac{1}{2}$ |
        | $P(c \mid b, a)$ | $0$ |
        | $P(a \mid b, b)$ | $0$ |
        | $P(b \mid b, b)$ | $0$ |
        | $P(c \mid b, b)$ | $0$ |
        | $P(a \mid b, c)$ | $1$ |
        | $P(b \mid b, c)$ | $0$ |
        | $P(c \mid b, c)$ | $0$ |
        | $P(a \mid c, a)$ | $\frac{2}{3}$ |
        | $P(b \mid c, a)$ | $0$ |
        | $P(c \mid c, a)$ | $\frac{1}{3}$ |
        | $P(a \mid c, b)$ | $1$ |
        | $P(b \mid c, b)$ | $0$ |
        | $P(c \mid c, b)$ | $0$ |
        | $P(a \mid c, c)$ | $1$ |
        | $P(b \mid c, c)$ | $0$ |
        | $P(c \mid c, c)$ | $0$ |

   - Note: included $P(NULL\mid a, a)$ because the string ends with 'aa' so the char after that is empty or null

## `calculate_probability(sequence, char, tables)`

### Formula
- ***Write the formula for sequence likelihood as described in section 2***
- Based on the sequence example given where n=3, the formula looks like:
    - $P(X_1 = x_1, X_2 = x_2, X_3 = x_3) = P(X_1 = x_1) \cdot P(X_2 = x_2 | X_1 = x_1) \cdot P(X_3 = x_3 | X_1 = x_1, X_2 = x_1)$
- The general formula for a n-gram model (where n=n) is:
    - $P(X_1 = x_1, X_2 = x_2, ... , X_n = x_n) = P(X_1 = x_1) \cdot P(X_2 = x_2 | X_1 = x_1) \cdot ... \cdot P(X_n = x_n | X_1 = x_1, X_2 = x_2, ... , X_{n-1} = x_{n-1})$

### Code analysis

- ***Put the intuition of your code here***
- Identifying the frequency table: It checks the appropriate frequency table based on the length of the `sequence`. If the `sequence` is of length \( m \), it uses the frequency table for \( m+1 \) grams.
- Calculating Total Context Count: It sums up the counts of characters that follow the given `sequence` to find the total occurrences.
- Finding Character Count: It retrieves the count of the specific character that follows the sequence.
Calculating and Returning Probability: Finally, it computes the probability as the ratio of the character count to the total count for the sequence, returning the result.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***
1. $P(X_1=a, X_2=a, X_3=a)$

   - $P(a, a, a) = P(X_1 = a) \cdot P(X_2 = a | X_1 = a) \cdot P(X_3 = a | X_1 = a, X_2 = a)$
        - From probability table 1, $P(X_1 = a) = P(a) = \frac{11}{20}$
        - From probability table 2, $P(X_2 = a | X_1 = a) = P(a | a) = \frac{5}{11}$
        - From probability table 3, $P(X_3 = a | X_1 = a, X_2 = a) = P(a | a, a) = \frac{1}{5}$
   - $P(a, a, a) = P(a) \cdot P(a | a) \cdot P(a | a, a)$
        - $= \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5}$
        - $= \frac{11 \cdot 5 \cdot 1}{20 \cdot 11 \cdot 5}$
   - $= \frac{1}{20}$


2. $P(X_1=a, X_2=a, X_3=b)$
   - $P(a, a, b) = P(X_1 = a) \cdot P(X_2 = a | X_1 = a) \cdot P(X_3 = b | X_1 = a, X_2 = a)$
        - From table 1, $P(X_1 = a) = P(a) = \frac{11}{20}$
        - From table 2, $P(X_2 = a | X_1 = a) = P(a | a) = \frac{5}{11}$
        - From table 3, $P(X_3 = b | X_1 = a, X_2 = a) = P(b | a, a) = \frac{2}{5}$
   - $P(a, a, b) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{2}{5}$
        - $= \frac{11 \cdot 5 \cdot 2}{20 \cdot 11 \cdot 5}$
   - $= \frac{1}{10}$


3. $P(X_1=a, X_2=a, X_3=c)$
   - $P(a, a, c) = P(X_1 = a) \cdot P(X_2 = a | X_1 = a) \cdot P(X_3 = c | X_1 = a, X_2 = a)$
        - From table 1, $P(X_1 = a) = P(a) = \frac{11}{20}$
        - From table 2, $P(X_2 = a | X_1 = a) = P(a | a) = \frac{5}{11}$
        - From table 3, $P(X_3 = c | X_1 = a, X_2 = a) = P(c | a, a) = \frac{1}{5}$
   - $P(a, a, c) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5}$
        - $= \frac{11 \cdot 5 \cdot 1}{20 \cdot 11 \cdot 5}$
   - $= \frac{1}{20}$


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- ***Put the intuition of your code here***
- Purpose: The function predicts the next character after a given sequence by utilizing n-gram probabilities.
- Process:
    1. Input: Receives a sequence, a list of frequency tables, and a vocabulary of potential characters.
    2. Probability Calculation: Iterates over each character in the vocabulary to compute the probability of each character following the sequence using the `calculate_probability` function.
    3. Max Probability Selection: Tracks and identifies the character with the highest calculated probability as the most likely next character.
        Output: Returns the character that has the highest probability of following the input sequence.

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**
  - If the sequence given to us is "aa", the next character is predicted to be: 'b'
  - This prediction is because the probability $P(b | aa)$ is the highest among the possible next characters ($\frac{2}{5}$), which indicates that `b` is the most likely character to follow the sequence `"aa"` based on the training data and the established n-gram model.
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

When experimenting with the two provided texts, "Alice's Adventures in Wonderland" and "War and Peace," different n-gram model behaviors emerged based on the characteristics of each corpus.

Findings:
- **"Alice's Adventures in Wonderland"**: This text contains whimsical and repetitive language, making it well-suited for higher n values (such as 4 or 5). The structured narrative and frequent use of dialogue allow the model to recognize patterns and predict the next character or word accurately. Predictions for sequences tend to be reliable, and the model performs efficiently even with larger n values.

- **"War and Peace"**: In contrast, this text presents a more complex and varied structure, with longer sentences and diverse vocabulary. As a result, while an n value of 2 or 3 suffices for capturing basic character relationships, the performance begins to decline with n values above 3. The model struggles with the variability and nuance of the language, resulting in less accurate predictions. Additionally, building frequency tables for larger n values became time-consuming due to the text's length and complexity.

In conclusion, while both texts provide valuable data for n-gram modeling, "Alice's Adventures in Wonderland" demonstrates better performance and maintains computational efficiency at higher n values compared to "War and Peace." As a practical guideline, values of n greater than 4 tend to become increasingly computationally intensive and less effective in predicting character sequences, especially for texts like "War and Peace" that introduce greater variability. I found that n = 3 to 4 strikes an effective balance, allowing for meaningful predictions without excessive computation.


<hr>


Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
