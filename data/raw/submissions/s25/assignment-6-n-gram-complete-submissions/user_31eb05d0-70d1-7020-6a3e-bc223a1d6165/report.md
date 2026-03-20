# Report

## 383GPT
Did you use 383GPT at all for this assignment (yes/no)?
- Yes

## Late Days
How many late days are you using for this assignment?
- Zero

## `create_frequency_tables(document, n)`

### Code analysis

- The function builds a set of frequency tables to capture how likely each character is to appear based on its preceding context in the document. For an n-gram model, this means collecting statistics for sequences of 1 up to n characters. For each character in the document, the function looks back up to n−1 characters to form sequences like unigrams, bigrams, trigrams, and so on. Each table tracks how often a given character follows a specific context (eg. a certain number of previous characters), which will later help the model predict the most likely next character based on what's been typed so far.

### Compute Probability Tables

**Note:** _Probability tables_ are different from _frequency_ tables**

- Assume that your training document is (for simplicity) `"aababcaccaaacbaabcaa"`, and the sequence given to you is `"aa"`. Given n = 3, do the following:
1. ***What is your vocabulary in this case***
   - {'a', 'b', 'c'}
2. ***Write down your probabillity table 1***:
   - as in $P(a), P(b), \dots$
   - For table 1, as in your probability table should look like this:

        | $P(\odot)$ | Probability value |  
        | ------ | ----------------- |
        | $P(a)$ | $\frac{11}{20}$ |
        | $P(b)$ | $\frac{4}{20}$ |
        | $P(c)$ | $\frac{5}{20}$ |
 
3. ***Write down your probability table 2***:
   - as in your probability table should look like (wait a second, you should know what I'm talking about)

      | $P(\odot)$ | Probability value |
      | $P(a \mid a)$ | $\frac{4}{9}$ |
      | $P(b \mid a)$ | $\frac{3}{9}$ |
      | $P(c \mid a)$ | $\frac{2}{9}$ |
      | $P(a \mid b)$ | $\frac{2}{4}$ |
      | $P(c \mid b)$ | $\frac{2}{4}$ |
      | $P(a \mid c)$ | $\frac{2}{3}$ |
      | $P(c \mid c)$ | $\frac{1}{3}$ |


4. ***Write down your probability table 3***:
   
         | $P(\odot)$ | Probability value |
         | $P(b \mid aa)$ | $\frac{2}{4}$ |
         | $P(a \mid aa)$ | $\frac{1}{4}$ |
         | $P(c \mid aa)$ | $\frac{1}{4}$ |
         | $P(a \mid ab)$ | $\frac{1}{3}$ |
         | $P(c \mid ab)$ | $\frac{2}{3}$ |
         | $P(b \mid ba)$ | $\frac{2}{4}$ |
         | $P(a \mid ba)$ | $\frac{2}{4}$ |
         | $P(a \mid bc)$ | $1$ |
         | $P(c \mid ca)$ | $\frac{1}{3}$ |
         | $P(a \mid ca)$ | $\frac{2}{3}$ |
         | $P(c \mid ac)$ | $\frac{2}{4}$ |
         | $P(b \mid ac)$ | $\frac{2}{4}$ |
         | $P(a \mid cb)$ | $1$ |
         | $P(a \mid cc)$ | $1$ |



## `calculate_probability(sequence, char, tables)`

### Formula
- ***Write the formula for sequence likelihood as described in section 2***

$$P(char \mid sequence) = \frac{P(count(sequence+char))}{P(count(sequence))}$$

$$P(x_{t+1} \mid x_1, x_2, \dots, x_t) = \frac{P(x_1, x_2, \dots, x_t, x_{t+1})}{P(x_1, x_2, \dots, x_t)}$$

### Code analysis

- The calculate_probability function estimates how likely a given character is to follow a specific sequence of characters based on patterns learned from a training document. It does this by looking up how often the full sequence (including the predicted character) appears compared to how often just the prefix (the sequence without the final character) appears.

### Your Calculations

- Now using your probability tables above, it is time to calculate the probability distribution of all the next possible characters from the vocabulary
- ***Calculate the following and show all the steps involved***

Starting with: "aababcaccaaacbaabcaa" and $n=3$

1. $P(X_1=a, X_2=a, X_3=a)$
   - $P(a) = 11/20$
   - $P(a | a) = 5/10$
   - $P(a | aa) = 2/3$
   - $$11/20 * 5/10 * 2/3 = 11/60$

2. $P(X_1=a, X_2=a, X_3=b)$
   - $P(a) = 11/20$
   - $P(a | a) = 5/10$
   - $P(b | aa) = 2/3$
   - $$11/20 * 5/10 * 1/3 = 11/120$


3. $P(X_1=a, X_2=a, X_3=c)$
   - $P(a) = 11/20$
   - $P(a | a) = 5/10$
   - $P(c | aa) = 2/3$
   - $$11/20 * 5/10 * 0 = 0$


## `predict_next_char(sequence, tables, vocabulary)`

### Code analysis

- Given a short sequence of characters, the function looks through every possible character in the vocabulary and calculates the conditional probability for each candidate character, then picks the one with the highest score (likelihood of being the next character).

### So what should be the next character in the sequence?
- **Based on the probability distribution obtained above for all the next possible characters, which character would be next in the sequence?**
  - Probably a since it has a highest probability calculated that b and c. 
 
## Experiment
- Experiment with the given corpus files and varying values of n. Do any corpus work better than others? How high of a value of n can you run before the table calculation becomes too time consuming? Write a short paragraph describing your findings.

Longer texts like `warandpeace.txt` had better and more coherent predictions, especially for n = 3 or 4. Shorter ones like `Alice's Adventures in Wonderland.txt` had slightly less accurate predictions. The predictions improved when moving from n=1 to n=3, but beyond n=5, it took way to long to execute and the predictions did not improve (they may even have detiroirated). This might be because some sequences simply don't appear often enough.

<hr>