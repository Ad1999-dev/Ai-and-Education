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
Did you use 383GPT at all for this assignment (yes/no)? yes

## Late Days
How many late days are you using for this assignment? 2

## `create_frequency_tables(document, n)`

### Code analysis

- create_frequency_tables is a function that takes in a corpus (document) and n-gram size (n) and returns a list of tables from unigram to n-gram + 1. These tables map chunks of chars to their frequency in the corpus. It does this by iterating through the document by n-gram length chunks and using each chunk as a key in the relevant n-gram table in order to increment its frequency. The n+1 table is for the prediction character.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
   - {a,b,c}
2. ***Write down your probabillity table 1***:
   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a)$ | $\frac{11}{20}$ |
        | $P(b)$ | $\frac{1}{5}$ |
        | $P(c)$ | $\frac{1}{4}$ |
 
1. ***Write down your probability table 2***:
   - as in your probability table should look like (wait a second, you should know what I'm talking about)

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a)$ | $\frac{5}{11}$ |
        | $P(b \mid a)$ | $\frac{3}{11}$ |
        | $P(c \mid a)$ | $\frac{2}{11}$ |
        | $P(a \mid b)$ | $\frac{2}{4}$ |
        | $P(b \mid b)$ | $\frac{0}{4}$ |
        | $P(c \mid b)$ | $\frac{2}{4}$ |
        | $P(a \mid c)$ | $\frac{3}{5}$ |
        | $P(b \mid c)$ | $\frac{1}{5}$ |
        | $P(c \mid c)$ | $\frac{1}{5}$ |

2. ***Write down your probability table 3***:
   - You got this!

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a,a)$ | $\frac{1}{5}$ |
        | $P(b \mid a,a)$ | $\frac{2}{5}$ |
        | $P(c \mid a,a)$ | $\frac{1}{5}$ |
        | $P(a \mid a,b)$ | $\frac{1}{3}$ |
        | $P(b \mid a,b)$ | $\frac{0}{3}$ |
        | $P(c \mid a,b)$ | $\frac{2}{3}$ |
        | $P(a \mid a,c)$ | $\frac{0}{2}$ |
        | $P(b \mid a,c)$ | $\frac{1}{2}$ |
        | $P(c \mid a,c)$ | $\frac{1}{2}$ |
        | $P(a \mid b,a)$ | $\frac{1}{2}$ |
        | $P(b \mid b,a)$ | $\frac{1}{2}$ |
        | $P(c \mid b,a)$ | $\frac{0}{2}$ |
        | $P(a \mid b,c)$ | $\frac{2}{2}$ |
        | $P(b \mid b,c)$ | $\frac{0}{2}$ |
        | $P(c \mid b,c)$ | $\frac{0}{2}$ |
        | $P(a \mid c,a)$ | $\frac{2}{3}$ |
        | $P(b \mid c,a)$ | $\frac{0}{3}$ |
        | $P(c \mid c,a)$ | $\frac{1}{3}$ |
        | $P(a \mid c,b)$ | $\frac{1}{1}$ |
        | $P(b \mid c,b)$ | $\frac{0}{1}$ |
        | $P(c \mid c,b)$ | $\frac{0}{1}$ |
        | $P(a \mid c,c)$ | $\frac{1}{1}$ |
        | $P(b \mid c,c)$ | $\frac{0}{1}$ |
        | $P(c \mid c,c)$ | $\frac{0}{1}$ |



## `calculate_probability(sequence, char, tables)`

### Formula
- ***Write the formula for sequence likelihood as described in section 2***

For an n-gram model, the probability of a sequence $x_1, x_2, \dots, x_t$ followed by character $x_{t+1}$ is calculated as:

$$P(x_{t+1} \mid x_1, x_2, \dots, x_t) = \frac{f(x_1, x_2, \dots, x_t, x_{t+1})}{f(x_1, x_2, \dots, x_t)}$$

Where $f(\cdot)$ represents the frequency of the sequence in the training corpus.

### Code analysis

- The calculate_probability function calculates the probability of a character occurring after a given sequence. It does this by dividing the number of times the sequence with its predicted character occurs in the corpus by the number of times the sequence alone occurs in the corpus. tables[n] contains the n+1 table for the sequence with its predicted char, and tables[n-1] contains the n table for the sequence alone.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***
1. $P(X_1=a, X_2=a, X_3=a)$
   - $= P(X_1=a) \cdot P(X_2=a \mid X_1=a) \cdot P(X_3=a \mid X_1=a, X_2=a) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5
   } = \frac{1}{20}$

2. $P(X_1=a, X_2=a, X_3=b)$
   - $= P(X_1=a) \cdot P(X_2=a \mid X_1=a) \cdot P(X_3=b \mid X_1=a, X_2=a) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{2}{5} = \frac{1}{10}$

3. $P(X_1=a, X_2=a, X_3=c)$
   - $= P(X_1=a) \cdot P(X_2=a \mid X_1=a) \cdot P(X_3=c \mid X_1=a, X_2=a) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5} = \frac{1}{20}$

## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- The predict_next_char function finds the most likely next character of a sequence. It iterates through each character in the vocabulary, calculating the probability of each character occurring after the given sequence using the calculate_probability function. It maps each character to its probability and returns the character with the highest probability as the prediction.

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**

    - $P(X_1=a, X_2=a, X_3=a) = \frac{1}{20}$
    - $P(X_1=a, X_2=a, X_3=b) = \frac{1}{10}$
    - $P(X_1=a, X_2=a, X_3=c) = \frac{1}{20}$
  
  The highest probability is $P(X_1=a, X_2=a, X_3=b) = \frac{1}{10}$, so "b" is most likely to follow "aa".
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

- I did not find one corpus to be better than the other. War and Peace sometimes created odd characters, like the trademark symbol or accented characters, but the actual autocomplete performance was similar to Alice's Adventures in Wonderland. At n=300, it takes tens of seconds to complete, several gigbytes of RAM, and a prompt from MacOS to force quit the application. For realistic autocomplete speeds, n=30 has about 1 second delay from submission to the first character autocompleted, which is the most I would consider to be responsive. I find n=4 to be nice because this application does not account for non-existent sequences, so longer grams tend not to autocomplete correctly compared to more common shorter sequences. If non-existent sequences were handled differently than proability=0, longer grams might perform better. Smaller grams like n=2 tend to repeat common short words like 'the' due to their small context window.

<hr>


Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
