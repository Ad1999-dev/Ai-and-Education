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

In general, Bayesian Networks are used to visually represent the dependencies (edges) between distinct random variables (nodes) in a large joint distribution. 

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

If we wanted to predict the most likely next character, we could compute the probability of every possible completion given each character in our vocabulary. This will give us a probability distribution over the next character prediction $P(X_4=x_4 \mid X_1=t, X_2=h, X_3=u)$. Taking the character with the max probability value in this distribution gives us an autocomplete model.

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
No

## Late Days
How many late days are you using for this assignment?
0

## `create_frequency_tables(document, n)`

### Code analysis

- ***Put the intuition of your code here***
To start, I separated the input document into a list of words. This was to eliminate all of the empty characters in the document while also allowing for an easy way to parse the documents contents. Then, I iterated through the words of the document to find all the unique characters in the document, which is done so we know which characters to include in our tables of frequencies, as characters that are never in the document will have frequencies of 0 for any string including them.

Then, I did an important step of initializing each entry string of the n frequency dictionaries to 0. The entries to table i consist of all strings of exactly i characters(with replacement) from the possible character set. This will be greatly beneficial when calculating joint probabilities, since every possible character sequence up to length n will have its own frequency now.

Next, I devised a function that given a particular word and subsequence size, will update the appropriate frequency table(based on the size) by considering all possible length gram subsequences of the input word. We do this by cleverly noticing that the number of possible subsequences in a given word is equal to the number of characters in that word, minus our desired size, plus 1. For example, if we wanted all length 2 subsequences in the sequence "thus", we are looking for 4 - 2 + 1 = 3 subsequences, which is correct(only "th", "hu", and "us"). We start with index=0 and find all of the possible subsequences, updating our table appropriately.

Lastly, we run this function for all words and all possible subsequences(according to our input). This gives us our array of frequency tables.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
   - ["a", "b", "c"]
2. ***Write down your probability table 1***:
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
        | $P(a \mid aa)$ | $\frac{1}{5}$ |
        | $P(b \mid aa)$ | $\frac{2}{5}$ |
        | $P(c \mid aa)$ | $\frac{1}{5}$ |
        | $P(a \mid ab)$ | $\frac{1}{3}$ |
        | $P(b \mid ab)$ | $\frac{0}{3}$ |
        | $P(c \mid ab)$ | $\frac{2}{3}$ |
        | $P(a \mid ac)$ | $\frac{0}{2}$ |
        | $P(b \mid ac)$ | $\frac{1}{2}$ |
        | $P(c \mid ac)$ | $\frac{1}{2}$ |
        | $P(a \mid ba)$ | $\frac{1}{2}$ |
        | $P(b \mid ba)$ | $\frac{1}{2}$ |
        | $P(c \mid ba)$ | $\frac{0}{2}$ |
        | $P(a \mid bb)$ | $\frac{0}{0}$ |
        | $P(b \mid bb)$ | $\frac{0}{0}$ |
        | $P(c \mid bb)$ | $\frac{0}{0}$ |
        | $P(a \mid bc)$ | $\frac{2}{2}$ |
        | $P(b \mid bc)$ | $\frac{0}{2}$ |
        | $P(c \mid bc)$ | $\frac{0}{2}$ |
        | $P(a \mid ca)$ | $\frac{2}{3}$ |
        | $P(b \mid ca)$ | $\frac{0}{3}$ |
        | $P(c \mid ca)$ | $\frac{1}{3}$ |
        | $P(a \mid cb)$ | $\frac{1}{1}$ |
        | $P(b \mid cb)$ | $\frac{0}{1}$ |
        | $P(c \mid cb)$ | $\frac{0}{1}$ |
        | $P(a \mid cc)$ | $\frac{1}{1}$ |
        | $P(b \mid cc)$ | $\frac{0}{1}$ |
        | $P(c \mid cc)$ | $\frac{0}{1}$ |

## `calculate_probability(sequence, char, tables)`

### Formula
- ***Write the formula for sequence likelihood as described in section 2***
Given integers n and m, where n is the length of the frequency array and m is the length of the sequence of characters that we are trying to find a joint probability of:

P(XM = xm | X1 = x1, X2 = x2, ... , X(M-1) = x(m-1)) = P(xm | x1, x2, ... , x(m-1)) = 
{
if (m <= n):
    P(x1, x2, ... , xm) / P(x1, x2, ... , x(m-1))
else (m > n):
    P(x(m-n+1), x(m-n+2), ... , xm) / P(x(m-n+1), ... , x(m-1))
}

### Code analysis

- ***Put the intuition of your code here***
Most of the algorithm resolves around the definition of a single function termJP(i), which calculates what the ith probability term in the joint probability calculation. The function distinguishes 3 cases:

- Case 1: i == 1
In this case, when we are trying to find the first term of the joint probability calculation(jpc), we need to consider only the first entry of the frequency array(which must always have at least one entry). This involves summing over all characters in the input document.

- Case 2: i <= n
In this case, we are trying to find the ith term of a jpc, and we are referencing 'tables' which has the frequencies of all substrings of at least length i or less. Thus, we don't need to approximate, and can simply express the jpc using the conditional probabilities in our table.

- Case 3: i > n
In this case, we are trying to find the ith term of a jpc, and we referencing 'tables' which has the frequencies of all substrings of at most length n, which is less than the length we want to find the probability of. Thus, we shorten our scope to the last n terms of the sequence ending at the ith term of the sequence, and approximate the ith jpc term this way.

After properly implementing these cases, we multiply an initial p=1 by each of the i terms in the input sequence formed by "sequence + char", which has length i.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***
1. $P(X_1=a, X_2=a, X_3=a)$
$P(X_1=a, X_2=a, X_3=a) = P(X_1=a) * P(X_2=a | X_1=a) * P(X_3=a | X_1=a, X_2=a) = (11/20) * (5/11) * (1/5) = (1/20)$

2. $P(X_1=a, X_2=a, X_3=b)$
$P(X_1=a, X_2=a, X_3=b) = P(X_1=a) * P(X_2=a | X_1=a) * P(X_3=b | X_1=a, X_2=a) = (11/20) * (5/11) * (2/5) = (2/20) = (1/10)$

3. $P(X_1=a, X_2=a, X_3=c)$
$P(X_1=a, X_2=a, X_3=c) = P(X_1=a) * P(X_2=a | X_1=a) * P(X_3=c | X_1=a, X_2=a) = (11/20) * (5/11) * (1/5) = (1/20)$


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- ***Put the intuition of your code here***
To predict the next character, we find the probability of each sequence that start with the input 'sequence', and end in any of the characters in 'vocabulary'. Each time we find the probability of a sequence with the current ending character, we check to see if that probability is greater than our current highest out of any of the characters we've tried so far. If so, we replace the highest probability/character with the current probability/character. When we finish, we return the character corresponding to the highest probability.

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**

Based on the probability distribution obtained above, we should predict that the next character in the sequence "aa" is "b".
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

<hr>
I started experimenting by testing with different values of n on the standard corpus warandpeace.txt. The main problem with using large values of n is that the computation time for computing the n frequency tables can get very large. For these experiments, I modified the main.py file slightly by commenting out the line print(tables), because this added unnecessary runtime when only trying to find the letter that made the most sense to predict.

For any values of n>4, the runtime was unreasonably large, regardless of how large the initial sequence was. However, despite these large runtimes, as long as the length of the input string was at most one less than n, we could be certain that the solution was exactly correct based on our corpus.

For values of n<=4, even with input strings that are sizeably larger than n, the process will terminate within 30 seconds. For n=4, the solutions generated should be correct or close to correct, while for any lower n, the solutions do not weigh previous characters enough, and thus the solutions tend to be simply the most common character sequences(like "the").

I also tried repeating these experiments after changing the size of the input text. I used a text with about the same number of distinct characters(39), but with overall length of only 455. In this scenario, I was able to evaluate a several next_char predictions at freq_array of n=5 in only around 30 seconds. This shows that the huge corpus definitely contributed to why n=5 was not feasible on warandpeace.txt.

I also tried repeating these experiments after changing the number of unique characters to 4, but with a similarly large corpus. For my input text, the vocabulary was only 4 characters: ["a", "s", "d", " "]. I tried using a text file of these characters with length 5.1 million. The result was that for frequency arrays of length 5, the prediction with an input string a length 4 was around 5 seconds. This performance is significantly faster than the performance of the original corpus which had an even smaller size(< 4 million characters total). It even worked with n=7 in under 10 seconds. Clearly, reducing the number of different distinct input characters is the main key in reducing the runtime of these types of operations. The time cost is exponential with the number of distinct characters in the vocabulary, so reducing this size is clearly vital for reducing runtime and allowing for the evaluation of joint probabilities for longer character string inputs.





Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
