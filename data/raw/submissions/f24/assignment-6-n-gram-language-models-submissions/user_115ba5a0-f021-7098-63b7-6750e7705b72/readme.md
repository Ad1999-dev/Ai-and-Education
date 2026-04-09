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
YES
## `create_frequency_tables(document, n)`

### Code analysis

- ***Put the intuition of your code here***
  For the creation of tables I used a helper function to make each of the tables, then I ran a for loop inserting those tables into a list for the return of the function

  My helper function takes 2 inputs, a document value, and a value of x (represents the curr n of the gram table)
  It parses the document as a string via a while loop, ending when it goes through the entire document (this is done inside calling the helper method later)
  and records every combination of strings that are x length.
  Ex "this is cool" would be "this" "his " "is i" etc
  inside the loop, the parsing records all of the instances of x character combinations and insets them into a dictionary. If the combination is not in the dictionary it creates a instance of the x letter combination with 1 as frequency. If it is already in, then it adds 1 to the frequency.

  Lastly A for loop is. This loop runs the helper function to create a table, with all n values up until and including n.
It adds these tables to a list. And outside of the loop the list is returned.

  

  

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
    vocab is {a,b,c}
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
        | $P(a \mid b)$ | $\frac{2}{4}$ |
        | $P(b \mid b)$ | $\frac{0}{4}$ |
        | $P(c \mid b)$ | $\frac{2}{4}$ |
        | $P(a \mid c)$ | $\frac{3}{5}$ |
        | $P(b \mid c)$ | $\frac{1}{5}$ |
        | $P(c \mid c)$ | $\frac{1}{5}$ |

2. ***Write down your probability table 3***:


| $P(\odot)$ | Probability value |  
| ------ | ----------------- |
| $P(a \mid aa)$         | $\frac{1}{5}$     |
| $P(b \mid aa)$         | $\frac{2}{5}$     |
| $P(c \mid aa)$         | $\frac{1}{5}$     |
| $P(a \mid ab)$         | $\frac{1}{3}$     |
| $P(b \mid ab)$         | $\frac{0}{3}$     |
| $P(c \mid ab)$         | $\frac{2}{3}$     |
| $P(a \mid ac)$         | $\frac{0}{2}$     |
| $P(b \mid ac)$         | $\frac{1}{2}$     |
| $P(c \mid ac)$         | $\frac{1}{2}$     |
| $P(a \mid ba)$         | $\frac{0}{2}$     |
| $P(b \mid ba)$         | $\frac{1}{2}$     |
| $P(c \mid ba)$         | $\frac{1}{2}$     |
| $P(a \mid bb)$         | N/A               |
| $P(b \mid bb)$         | N/A               |
| $P(c \mid bb)$         | N/A               |
| $P(a \mid bc)$         | $\frac{2}{2}$     |
| $P(b \mid bc)$         | $\frac{0}{2}$     |
| $P(c \mid bc)$         | $\frac{0}{2}$     |
| $P(a \mid ca)$         | $\frac{2}{3}$     |
| $P(b \mid ca)$         | $\frac{0}{3}$     |
| $P(c \mid ca)$         | $\frac{1}{3}$     |
| $P(a \mid cb)$         | $\frac{1}{1}$     |
| $P(b \mid cb)$         | $\frac{0}{1}$     |
| $P(c \mid cb)$         | $\frac{0}{1}$     |
| $P(a \mid cc)$         | $\frac{1}{1}$     |
| $P(b \mid cc)$         | $\frac{0}{1}$     |
| $P(c \mid cc)$         | $\frac{0}{1}$     |





## `calculate_probability(sequence, char, tables)`

### Formula
- ***Write the formula for sequence likelihood as described in section 2***
For
sequence = $X_1, X_2, \ldots, X_n$  
char = $X_c$

$$ P(char \mid sequence) = \frac{P(X_1, X_2, \ldots, X_n, X_c)}{P(X_1, X_2, \ldots, X_n)} $$


### Code analysis

- ***Put the intuition of your code here***
  First part of my code checks to make sure the inputs correctly match what is desired.
  - character is only 1 length
  - Sequence at least 1 letter long
  
  Next it checks to see if the sequence and sequence + char are in the tables.
  If ethier are not in their respected table the function returns 0 for the probablity since it doesnt occur in the document.
  (This is checked inside the loop for cases in which n<len(sequence+char)
  
  next I run a for loop that creates the joint probility of the sequence.
  The table assocaited with the n of the sequence so far is taken from the input ngram list, and assigned to a variable.
  Then a frequency value is taken from this table. (number of this part of the sequence in the document tables are made of)
  
  For intial letter (i=0) it represents $\frac{X_1}{Size(C)}$, It assigns this fraction to a probabiliy variable.
  
  
  After the intial (i=1-> len(sequence)-1, the loop does the same process of assigning a frequency uses the current and previous
  frequencies until it finds the probabilty of the next part of the sequence.
  In other words, $P(next char in sequence \mid sequence so far)$
  If n<len(sequence+char) Then it looks like $P(next char in sequence \mid last n characters of sequence so far)$
  
  Ex. $P(X_2 \mid X_1), P(X_3 \mid X_2, X_1), P(X_4 \mid X_3, X_2)$ etc.
  
  Each prob is multiplied by the probability variable
  It does this until it gets the joint probabilty for the entire sequence.
  
  Next it finds the joint probablity the sequence with the character added on
  Using the same process. (Finding the frequency of the table and finding $P(char | sequence)$
  
  Lastly it divides the using the fomula:
  $P(char \mid sequence) = \frac{P(X_1, X_2, \ldots, X_n, X_c)}{P(X_1, X_2, \ldots, X_n)}$
  
  


### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***
1. $P(X_1=a, X_2=a, X_3=a)$ 
   - *Show your work* <br>
  $P(X_1=a, X_2=a, X_3=a) = P(X_1=a) * P(X_2=a \mid X_1=a) * P(X_3=a \mid X_2=a, X_1=a)$<br><br>
                          = $\frac{11}{20} * \frac{5}{11} * \frac{1}{5}$<br><br>
                          = $\frac{55}{1100} $<br><br>
                          = $\frac{1}{20} $


     
2. $P(X_1=a, X_2=a, X_3=b)$
      - *Show your work* <br>
  $P(X_1=a, X_2=a, X_3=b) = P(X_1=a) * P(X_2=a \mid X_1=a) * P(X_3=b \mid X_2=a, X_1=a)$<br><br>
                          = $\frac{11}{20} * \frac{5}{11} * \frac{2}{5}$<br><br>
                          = $\frac{110}{1100} $<br><br>
                          = $\frac{1}{10} $
         
3. $P(X_1=a, X_2=a, X_3=c)$
   - *Show your work* <br>
  $P(X_1=a, X_2=a, X_3=c) = P(X_1=a) * P(X_2=a \mid X_1=a) * P(X_3=c \mid X_2=a, X_1=a)$<br><br>
                          = $\frac{11}{20} * \frac{5}{11} * \frac{1}{5}$<br><br>
                          = $\frac{55}{1100} $<br><br>
                          = $\frac{1}{20} $


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- ***Put the intuition of your code here***
For the predict_next_char I made a for loop that tests all of the possible characters inside the vocabulary function and calculates there probablity of occuring. Then the character with the highest probabilty is returned.

2 variables start, char are intialized as 0
The for loop is ran with i as one of the letters in vocabulary.

Inside the loop the probabilty is calculated for $P(i \mid sequence)$

If this probabilty is greater than the current prob assigned to a variable start. Then it assigns itself to be the new start value, At the same time the current char (i) is saved as the variable char.

At the end of the loop the variables char and start will have the highest probablity character inside the vocabulary set, and the specific char respectfully

Finally the character inside the char variable is returned.

NOTE: If none of the characters inside the vocabulary set follow the sequence within the document the table is made after. The variable Char returns 0. This is the nature of the function, and stated in campuswire that this is 


  

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**
  - *Your answer*
    The next character in the sequence should be "b" so the whole sequence will be "aab" since it has a higher probabilty of occuring.

NOTE: The desired output of the function calculate_probabilty() is $P(char | sequence)$ not $P(X_1, X_2, \ldots X_n,X_c)$ <br>
But this will still output the same for predict next character, since the conditional prob of "aab" is still higher at <br> $\frac{2}{5}$ vs $\frac{1}{5}$ for "aaa", "aac"

 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

Off simple testing, it seems like the creation of a tables time consumption is based off two factors, the n value and the length of the text. The longer the text is, as one would assume, the more combinations of N-length strings there are and such more separate inputs into the dictionary. As the N value increases so does the time to run the function since you are calculating up until the N value. So naturally an N value of 6 vs 7 is vastly different. An N value of 7 includes all of the tables in the N=6 function return value, as well as the one table from N=7. Overall on the considerably long document (war and peace) the N=15 took around 20 seconds, and N=20 took around 30 Using what we just learned an N=21 would take at least 30 seconds or more. With my computer in mind, this was the highest I was willing to go for calculations. This is a huge amount of time to be assessed for autocorrect, additionally, I think that it is pretty unnecessary to be calculating with N values larger than this for autocorrect. Overall If you were running autocorrect on the entire English dictionary with this tool, it would not really be realistic, but it works as intended.
<hr>


Please don't hesitate to reach out to us in case of any questions (no question is dumb), and come meet us during office hours XD!
Happy coding!
