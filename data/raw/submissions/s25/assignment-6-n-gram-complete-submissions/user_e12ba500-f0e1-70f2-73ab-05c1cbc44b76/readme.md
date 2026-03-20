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
Did you use 383GPT at all for this assignment (yes/no)? Yes

## Late Days
How many late days are you using for this assignment? 0

## `create_frequency_tables(document, n)`

### Code analysis

- ***Put the intuition of your code here***
``` python
   def create_frequency_tables(document, n):
    """
    This function constructs a list of `n` frequency tables for an n-gram model, each table capturing character frequencies with increasing conditional dependencies.

    - **Parameters**:
        - `document`: The text document used to train the model.
        - `n`: The number of value of `n` for the n-gram model.

    - **Returns**:
        - Returns a list of n frequency tables.
    """
    frequency_tables_of_n_grams = []
    for i_th_table in range(1, n+1):
        i_th_gram_table = defaultdict(int)
        for position in range(len(document) - i_th_table + 1):
            sequence = document[position:position+i_th_table]
            i_th_gram_table[sequence] += 1
        frequency_tables_of_n_grams.append(i_th_gram_table)


    return frequency_tables_of_n_grams
   ```
- The logic behind this function is straightforward. I use a dictionary to store the occurrences of character sequences. Each table keeps track of sequences of different lengths up to n. For example, table[0] is the base table that stores the occurrences of individual characters. Similarly, table[1] tracks sequences of length 2, and so on. Finally, I return the list of tables. The for loop means the ith table correspoding to the table of sequences which have length i. The second loop keeps traversing the document to calculate sequences which have length i. 
### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
Since Vocabulary is a set of characters make of the document. Then Vocabulary includes a,b,c in this case
2. ***Write down your probabillity table 1***:
   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a)$ | $\frac{11}{20}$ |
        | $P(b)$ | $\frac{4}{20}$ |
        | $P(c)$ | $\frac{5}{20}$ |
 
1. ***Write down your probability table 2***:
   - as in your probability table should look like (wait a second, you should know what I'm talking about)

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a \mid a)$ | $\frac{5}{11}$ |
        | $P(b \mid a)$ | $\frac{3}{11}$ |
        | $P(c \mid a)$ | $\frac{2}{11}$ |
        | $P(c \mid b)$ | $\frac{2}{4}$ |
        | $P(a \mid b)$ | $\frac{2}{4}$ |
        | $P(c \mid c)$ | $\frac{1}{5}$  |
        | $P(a \mid c)$ | $\frac{3}{5}$ |
        | $P(b \mid c)$ | $\frac{1}{5}$ |
        

2. ***Write down your probability table 3***:
   - You got this!

    | $P(\odot)$ | Probability Value |
    |-----------------------|-------------------|
    | $P(a \mid (a, a))$ | $\frac{1}{5}$ |
    | $P(b \mid (a, a))$ | $\frac{2}{5}$ |
    | $P(c \mid (a, a))$ | $\frac{1}{5}$ |
    | $P(a \mid (a, b))$ | $\frac{1}{3}$ |
    | $P(c \mid (a, b))$ | $\frac{2}{3}$ |
    | $P(c \mid (a, c))$ | $\frac{1}{2}$ |
    | $P(b \mid (a, c))$ | $\frac{1}{2}$ |
    | $P(a \mid (b, a))$ | $\frac{1}{2}$ |
    | $P(b \mid (b, a))$ | $\frac{1}{2}$ |
    | $P(a \mid (b, c))$ | $\frac{2}{2}$ |
    | $P(a \mid (c, a))$ | $\frac{2}{3}$ |
    | $P(c \mid (c, a))$ | $\frac{1}{3}$ |
    | $P(a \mid (c, b))$ | $\frac{1}{1}$ |
    | $P(a \mid (c, c))$ | $\frac{1}{1}$ |



## `calculate_probability(sequence, char, tables)`

### Formula
- ***Write the formula for sequence likelihood as described in section 2***

$$\begin{align}
P(X_1 = x_1, X_2 = x_2, \dots , X_k = x_k) &= P(x_1)P(x_2 \mid x_1)\dots P(x_k \mid x_{k-1},\dots,x_{k-n+1}) \\
&= \frac{f(x_1)}{size(C)}\frac{f(x_1,x_2)}{f(x_1)}\dots\frac{f(x_{k-n+1},\dots,x_k)}{f(x_{k-n+1},\dots,x_{k-1})}
\end{align}$$
Since n =3 in this case, then
$$\frac{f(x_1)}{size(C)}\frac{f(x_1,x_2)}{f(x_1)}\dots\frac{f(x_{k-3+1},\dots,x_k)}{f(x_{k-3+1},\dots,x_{k-1})}$$

### Code analysis

- ***Put the intuition of your code here***
- As the sequence length increases, the size of the tables grows exponentially. To address this issue, I implemented an N-gram model to calculate probabilities instead of using the full joint distribution. In the code, I first check the length of the sequence. If it is 0, the function simply returns the probability of individual characters. Otherwise, it calculates the probability of the sequence using the N-gram formula.

- More specifically, the “previous letters” refer to the sequence of characters preceding a specific character in the sequence up to n (in this case n == 3). For example, if the sequence is `aab`, then at position 2, the previous letters are `aa`. The algorithm looks up the table storing 2-character sequences to find the occurrence of `aa`, and then checks the 3-character table to find the occurrence of `aab`. Finally, it applies the formula $\frac{f(x1,x2,\dots,x_{k+1})}{f(x1,x2,\dots,x_{k})}$ to find the probability of `aab`, to calculate the probability of `aab`, and multiplies that result into the final probability of the full sequence.
  
``` python
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
    re_probability = 1
    corpus_size = sum(tables[0].values())
    if(len(sequence) == 0 or len(sequence) == 1):
        re_probability *= (tables[0].get(char,0)/corpus_size)
        return re_probability
    re_probability *= tables[0].get(sequence[0],0) / corpus_size
    sequence = sequence + char
    for pos in range(1,len(sequence)):
        current_letter = sequence[pos]
        previous_letters = sequence[max(0,pos - (len(tables)-1)):pos]
        table_correspoding_to_current_letter = tables[len(previous_letters)-1]
        freq_letters_so_far = tables[len(previous_letters)].get(previous_letters+current_letter,0)
        previous_letters_freq = table_correspoding_to_current_letter.get(previous_letters,0)
        if(previous_letters_freq == 0 or freq_letters_so_far == 0):
            return 0
        re_probability *= (freq_letters_so_far/previous_letters_freq)
        

    return re_probability
```

### Your Calculations
- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary

1. $P(X_1=a, X_2=a, X_3=a)$

$$\begin{align}
        P(X_1=a, X_2=a, X_3=a) &= P(a)P(a \mid a)P(a \mid (aa)) \\
        &= \frac{f(a)}{size(C)}\frac{f(aa)}{f(a)}\frac{f(aaa)}{f(aa)}\\
        &= \frac{11}{20}\frac{5}{11}\frac{1}{5}\\
        &= 1/20 \\ 
        &= 0.05 
\end{align}$$

2. $P(X_1=a, X_2=a, X_3=b)$

$$\begin{align}
P(X_1=a, X_2=a, X_3=b) &= P(a)P(a \mid a)P(b \mid (aa)) \\
&= \frac{f(a)}{size(C)}\frac{f(aa)}{f(a)}\frac{f(aab)}{f(aa)}\\
&= \frac{11}{20}\frac{5}{11}\frac{2}{5}\\
&= 2/20 \\
&= 0.1 
\end{align}$$

3. $P(X_1=a, X_2=a, X_3=c)$
   
$$\begin{align}
P(X_1=a, X_2=a, X_3=c) &= P(a)P(a \mid a)P(c \mid (aa)) \\
&= \frac{f(a)}{size(C)}\frac{f(aa)}{f(a)}\frac{f(aac)}{f(aa)}\\
&= \frac{11}{20}\frac{5}{11}\frac{1}{5}\\
&= 1/20 \\ 
&= 0.05
\end{align}$$


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- ***Put the intuition of your code here***
``` python
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
    init_max = 0
    init_char = ''
    for char in vocabulary:
        prob_char = calculate_probability(sequence,char,tables)
        if(prob_char> init_max):
            init_max = prob_char
            init_char = char
    return init_char
```
This function calculates the probabilities of possible next characters in the sequence and selects the one with the maximum probability using the helper function `calculate_probability()`. Specifically, two variables `init_max` and `init_char` keep track maximum probability and next character. Then, the code traverses the list of vocabulary calcuting the prob of each next letter.   

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**
  - b
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

- After conducting several experiments, I observed the main different key point: The corpus from 'Alice's Adventures in Wonderland' performed better than that of 'War and Peace.' This difference is attributed to the size of the corpus; a larger corpus leads to a more extensive set of frequency tables. As a result, 'Alice's Adventures in Wonderland' works exceptionally well with higher values of n. For example, with n = 10, both texts performed adequately, but when n exceeded 10, especially when n > 100, only 'Alice's Adventures in Wonderland' was able to produce results." The second reason I can think up is to the number of unique terms in 'War and Peace' is pretty high. For example, the size of vocabulary of 'War and Peace' is 82; meanwhile that of 'Alice's Adventures in Wonderland' is just 66.     
<hr>


Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
