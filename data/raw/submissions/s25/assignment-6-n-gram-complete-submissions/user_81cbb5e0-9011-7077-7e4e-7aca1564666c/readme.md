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
How many late days are you using for this assignment? 0

## `create_frequency_tables(document, n)`

### Code analysis

The `create_frequency_tables` function constructs a list of frequency tables for an n-gram model. The intuition behind the code is to capture the occurrences of n-grams of varying lengths (from 1 to \( n \)) within the given text document. This allows us to analyze the relationships and patterns of character sequences in the text, which is essential for tasks like character prediction.
Essentially, we:
Iterate through the document to extract substrings (n-grams).
Count the occurrences of each n-gram and store these counts in frequency tables.
Return a list of these tables, where each table corresponds to a different length of n-gram.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***

   Given the training document `"aababcaccaaacbaabcaa"`, we can identify the unique characters present in the text as follows:
Vocabulary:
\[
\{ 'a', 'b', 'c' \}
\]
3. ***Write down your probabillity table 1***:
   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a)$ | $\frac{11}{20}$ |
        | $P(b)$ | $\frac{4}{20}$ |
        | $P(c)$ | $\frac{5}{20}$ |
 
4. ***Write down your probability table 2***:
   - as in your probability table should look like (wait a second, you should know what I'm talking about)

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a)$ | $\frac{5}{10}$ |
        | $P(a \mid b)$ | $\frac{2}{4}$ |
        | $P(a \mid c)$ | $\frac{3}{5}$ |
        | $P(b \mid a)$ | $\frac{3}{10}$ |
        | $P(b \mid b)$ | $0$ |
        | $P(b \mid c)$ | $\frac{1}{5}$ |
        | $P(c \mid a)$ | $\frac{2}{10}$ |
        | $P(c \mid b)$ | $\frac{2}{4}$ |
        | $P(c \mid c)$ | $\frac{1}{5}$ |

5. ***Write down your probability table 3***:
   - You got this!
  
        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a, a)$ | $\frac{1}{4}$ |
        | $P(a \mid a, b)$ | $\frac{1}{3}$ |
        | $P(a \mid b, a)$ | $\frac{1}{2}$ |
        | $P(a \mid b, c)$ | $1$ |
        | $P(a \mid c, a)$ | $\frac{2}{3}$ |
        | $P(a \mid c, b)$ | $1$ |
        | $P(a \mid c, c)$ | $1$ |
        | $P(b \mid a, a)$ | $\frac{2}{4}$ |
        | $P(b \mid a, c)$ | $\frac{1}{2}$ |
        | $P(b \mid b, a)$ | $\frac{1}{2}$ |
        | $P(c \mid a, a)$ | $\frac{1}{4}$ |
        | $P(c \mid a, b)$ | $\frac{2}{3}$ |
        | $P(c \mid a, c)$ | $\frac{1}{2}$ |
        | $P(c \mid c, a)$ | $\frac{1}{3}$ |


## `calculate_probability(sequence, char, tables)`

### Formula

$$P(X_1=a, X_2=b, X_3=c) = P(a) \cdot P(b \mid a) \cdot P(c \mid a, b)$$

### Code analysis

The function `calculate_probability(sequence, char, tables)` is designed to calculate the conditional probability of a character occurring after a specified sequence using the n-gram frequency tables. The intuition behind this function is to leverage previously computed frequencies to make informed predictions about what character might come next based on language patterns observed in the training data.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***
1. $P(X_1=a, X_2=a, X_3=a)$

$$P(X_1=a, X_2=a, X_3=a) = P(a) \cdot P(a \mid a) \cdot P(a \mid a, a)$$

$$                       = \frac{5}{10} \cdot \frac{5}{10} \cdot \frac{1}{4}$$

$$                       = \frac{1}{16}$$

2. $P(X_1=a, X_2=a, X_3=b)$

$$P(X_1=a, X_2=a, X_3=b) = P(a) \cdot P(a \mid a) \cdot P(b \mid a, a)$$

$$                       = \frac{5}{10} \cdot \frac{5}{10} \cdot \frac{2}{4}$$

$$                       = \frac{1}{8}$$
   
3. $P(X_1=a, X_2=a, X_3=c)$

$$P(X_1=a, X_2=a, X_3=c) = P(a) \cdot P(a \mid a) \cdot P(c \mid a, a)$$

$$                       = \frac{5}{10} \cdot \frac{5}{10} \cdot \frac{1}{4}$$

$$                       = \frac{1}{16}$$


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

The function `predict_next_char(sequence, tables, vocabulary)` aims to predict the next character after a given sequence based on previously computed frequency tables. Here’s the intuition behind the code:

Conditional Probability: The function leverages the conditional probabilities calculated from n-grams, allowing it to utilize the context of the preceding characters in the sequence to make an informed prediction about the next character.

Iterate Over Vocabulary: The function iterates through all possible next characters (from the specified vocabulary) and computes the probability of each character occurring after the provided sequence. It calculates this probability using the frequency tables that detail how often each character follows a specific sequence of characters.

Select Maximum Probability: After calculating the probabilities for each potential next character, the function identifies the character with the highest probability, which is returned as the predicted next character. This allows for a prediction that is informed by both the immediate context and the patterns observed in the training data.

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**

Since the highest probability calculated above for all the next possible characters is $P(X_1=a, X_2=a, X_3=b) = \frac{1}{8}$, $b$ would be the next character in the sequence.
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

In experimenting with the corpus files "War and Peace" and "Alice's Adventures in Wonderland," I observed that both texts performed similarly in the context of n-gram modeling. The computational time and memory usage increased noticeably as I raised the value of `n`. While values of `n` up to 10 were handled reasonably well, I found that setting `n` to 20 began to significantly slow down the calculations, likely due to the exponential growth in the number of possible n-grams. At `n=50`, the performance became extremely sluggish, to the point where processing took a prohibitive amount of time. This suggests that while higher n values can capture more contextual information in the text, they also dramatically increase the complexity of the model, leading to longer computation times. Thus, a balance between capturing sufficient context and maintaining computational efficiency is crucial when selecting an optimal value for `n`. Overall, moderate values of `n` (between 2 and 10) seem to provide a good compromise for practical use in n-gram models while still yielding meaningful predictions.


Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
