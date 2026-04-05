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
Did you use 383GPT at all for this assignment (yes/no)? YES

## `create_frequency_tables(document, n)`

### Code analysis

- When passing in a document and an integer n, the function will essentially go through the string n times. Each time it will count the occurences of a substring starting with length 1, all the way up to n+1. This means that the first occurences it counts every single individual letter. Then it counts every susbtring of length 2. This is all stored in a list of dictionaries which have a key of the start letter, and then have a value of another dictionary of the second letter, or the third letter. For example, the key of "" has 1 value, which is a dictionary of the count of the vocabulary. Then there is another dicitionary with keys a, b and c, which all have values of dictionaries with counts of the vocabulary that come after these letters. This function will generate dictionaries up to a key of size n.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
   - The vocabulary would be every unique letter in our training document `"aababcaccaaacbaabcaa"`, which makes the vocabulary {a,b,c}

2. ***Write down your probabillity table 1***:
   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

        #### Final Table
        | Character | Probability |
        |-----------|-------------|
        | \( P(a) \) | 0.55        |
        | \( P(b) \) | 0.2         |
        | \( P(c) \) | 0.25        |
 
1. ***Write down your probability table 2***:
   - as in your probability table should look like (wait a second, you should know what I'm talking about)


### Probability Table 2 (Bigram Probabilities)

| Context + Next Character | Probability |
|---------------------------|-------------|
| \( P(a \| a) \)         | 0.5         |
| \( P(b \| a) \)         | 0.3         |
| \( P(c \| a) \)         | 0.2         |
| \( P(a \| b) \)         | 0.5         |
| \( P(c \| b) \)         | 0.5         |
| \( P(a \| c) \)         | 0.6         |
| \( P(c \| c) \)         | 0.2         |
| \( P(b \| c) \)         | 0.2         |


2. ***Write down your probability table 3***:
   - You got this!
### Probability Table 3 (Trigram Probabilities)

| Context + Next Character | Probability |
|---------------------------|-------------|
| \( P(b \| aa) \)        | 0.5         |
| \( P(a \| aa) \)        | 0.25        |
| \( P(c \| aa) \)        | 0.25        |
| \( P(a \| ab) \)        | 0.3333      |
| \( P(c \| ab) \)        | 0.6667      |
| \( P(b \| ba) \)        | 0.5         |
| \( P(a \| ba) \)        | 0.5         |
| \( P(a \| bc) \)        | 1.0         |
| \( P(c \| ca) \)        | 0.3333      |
| \( P(a \| ca) \)        | 0.6667      |
| \( P(c \| ac) \)        | 0.5         |
| \( P(b \| ac) \)        | 0.5         |
| \( P(a \| cc) \)        | 1.0         |
| \( P(a \| cb) \)        | 1.0         |



## `calculate_probability(sequence, char, tables)`



# Formula for Sequence Likelihood (as described in Section 2)

The likelihood of a sequence can be computed using the **chain rule of probability**, where each character's probability depends on its context (previous characters). For a sequence \( X = x_1, x_2, x_3, x_4 \), the full joint probability is given by:

$$
P(X_1 = x_1, X_2 = x_2, X_3 = x_3, X_4 = x_4) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdot P(x_4 \mid x_1, x_2, x_3)
$$

---

## Computing the Probability of the Next Character

When predicting the next character, the probability is computed using conditional distributions. For a sequence \( x_1, x_2, x_3 \), the probability of \( x_4 \) is:

$$
P(X_4 = x_4 \mid X_1 = x_1, X_2 = x_2, X_3 = x_3) = \frac{P(X_1 = x_1, X_2 = x_2, X_3 = x_3, X_4 = x_4)}{P(X_1 = x_1, X_2 = x_2, X_3 = x_3)}
$$

This can be estimated using frequency tables:

$$
P(X_4 = x_4 \mid X_1 = x_1, X_2 = x_2, X_3 = x_3) = \frac{f(x_1, x_2, x_3, x_4)}{f(x_1, x_2, x_3)}
$$

---

## Example: Predicting the Next Character

For the sequence `"thu"`, where we want to predict the probability of the next character being `"s"`:

$$
P(X_4 = s \mid X_1 = t, X_2 = h, X_3 = u) = \frac{f(t, h, u, s)}{f(t, h, u)}
$$

---

## General Case for N-Gram Models

For an n-gram model, the joint probability of a sequence \( x_1, x_2, \dots, x_t \) can be approximated as:

$$
P(x_1, x_2, \dots, x_t) = \prod_{i=1}^{t} P(x_i \mid x_{i-n+1}, \dots, x_{i-1})
$$

Where \( P(x_i \mid x_{i-n+1}, \dots, x_{i-1}) \) is computed as:

$$
P(x_i \mid x_{i-n+1}, \dots, x_{i-1}) = \frac{f(x_{i-n+1}, \dots, x_{i-1}, x_i)}{f(x_{i-n+1}, \dots, x_{i-1})}
$$

---

## Simplified Formula for a Bigram Model (\( n = 2 \))

For a bigram model, the sequence probability is computed as:

$$
P(X_1 = x_1, X_2 = x_2, X_3 = x_3, X_4 = x_4) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_2) \cdot P(x_4 \mid x_3)
$$

Where the probabilities are calculated using the ratio of frequencies:

$$
P(x_i \mid x_{i-1}) = \frac{f(x_{i-1}, x_i)}{f(x_{i-1})}
$$

---

## Full Formula for Likelihood

The general likelihood formula for any n-gram model is:

$$
P(x_1, x_2, \dots, x_t) = \prod_{i=1}^{t} \frac{f(x_{i-n+1}, \dots, x_{i-1}, x_i)}{f(x_{i-n+1}, \dots, x_{i-1})}
$$

This represents the sequence likelihood under the assumption that each character depends only on the previous \( n-1 \) characters.




### Code analysis

- This function computes the conditional probability of a character that occurs after a sequence. The function determines the context length by taking up to n-1 characters from th sequence. n is the size of the largest n-gram table. It selects the appropriate frequence table based on the context length and checks if the context exists in the table. If the context is missing, it returns a probability of 0 to handle unsees data. Otherwise, it retrieves the total frequency of the character following that context. The conditional probability is then calculated as the ration of these two values. 


### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***


### 1. $P(X_1=a, X_2=a, X_3=a)$


### Multiply the Probabilities:

Using the formula:

$
P(X_1 = a, X_2 = a, X_3 = a) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = a \mid X_1 = a, X_2 = a)
$

Substitute the values from the probability tables:

$
P(X_1 = a, X_2 = a, X_3 = a) = 0.55 \cdot 0.5 \cdot 0.25
$

Perform the multiplication:

$
P(X_1 = a, X_2 = a, X_3 = a) = 0.06875
$

---

### Final Answer:

$
P(X_1 = a, X_2 = a, X_3 = a) = 0.06875
$




### 2. $P(X_1=a, X_2=a, X_3=b)$

### Calculating \( P(X_1 = a, X_2 = a, X_3 = b) \)

Using the formula:

$
P(X_1 = a, X_2 = a, X_3 = b) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = b \mid X_1 = a, X_2 = a)
$


---

### Multiply the Probabilities:

Now substitute the values into the formula:

$
P(X_1 = a, X_2 = a, X_3 = b) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = b \mid X_1 = a, X_2 = a)
$


$
P(X_1 = a, X_2 = a, X_3 = b) = 0.55 \cdot 0.5 \cdot 0.5
$

Perform the multiplication:

$
P(X_1 = a, X_2 = a, X_3 = b) = 0.1375
$

---

### Final Answer:

$
P(X_1 = a, X_2 = a, X_3 = b) = 0.1375
$



### 3. $P(X_1=a, X_2=a, X_3=c)$

### Calculating \( P(X_1 = a, X_2 = a, X_3 = c) \)

Using the formula:

$
P(X_1 = a, X_2 = a, X_3 = c) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = c \mid X_1 = a, X_2 = a)
$

---


### Multiply the Probabilities:

Now substitute the values into the formula:

$
P(X_1 = a, X_2 = a, X_3 = c) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = c \mid X_1 = a, X_2 = a)
$


$
P(X_1 = a, X_2 = a, X_3 = c) = 0.55 \cdot 0.5 \cdot 0.25
$

Perform the multiplication:

$
P(X_1 = a, X_2 = a, X_3 = c) = 0.06875
$

---

### Final Answer:

$
P(X_1 = a, X_2 = a, X_3 = c) = 0.06875
$



## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- For each string in the vocabulary, we compute the probability of getting that string given the sequence. After finding all of those probabilities, we find the sequence that has the maximum probability and return the character that is most likely to show up.

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**

Based on our probability distribution, b would be the most likely character as the probability of b given aa is .1375 while the probability of a or c is .06875
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

When using the war and piece with n = 15, sequence = dark, and k = 10, it took a substantial amount of time, approximately 30 seconds. Using these same parameters but with the Alice in Wonderland doc, it took less than 2 seconds. This is mainly due to the war and piece corpus being around 70,000 lines on VSCode while Alice in Wonderland is around a measly 1700. I will make the assumption that almost any query run on the Training document of Alice In Wonderland will be much faster than any that is run on war and piece as there will be less content to parse. war and piece will get exponentially longer as we continue increasing n due to the iterative nature of the functions. There could possibly be ways to optimiize this but in any case, a shorter document will usually take lesser amounts of time.


Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
