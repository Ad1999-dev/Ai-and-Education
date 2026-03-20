[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xUFhSpv5)
# Assignment 7: Neural Complete

## Overview

In Assignment 6 you computed a language model from scratch. Now it's time to apply your deep learning knowledge to the autocomplete problem and use what you've learned about deep learning to train a neural language model for next character prediction.

## Assignment Objectives

1. Understand how a character-level RNN works and how it can model sequences.
2. Implement a recurrent neural network in PyTorch.
3. Learn about sequence modeling, hidden state propagation, and embedding layers.
4. Train a model to predict the next character in a sequence using a sliding window dataset.
5. Generate novel sequences of text based on a trained model.
6. Experiment with model hyperparameters and observe their effect on performance.

## Pre-Requisites

- **Python & PyTorch:** You should be familiar with Python syntax and have basic experience with PyTorch tensors and modules (from Assignment 5).
- **Neural Networks:** You should understand how neural networks work, including layers, forward passes, and training with loss functions.
- **Recurrent Neural Networks:** You should have seen the basic RNN recurrence equations in lecture.

---

## Student Tasks

### Milestone 0. Understand the code

Start by opening `rnn_complete.py` reading through whats provided and familiarizing yourself with the structure.

The key components are
- A `CharDataset` class to slice training data into overlapping character sequences.
- A `CharRNN` class with an incomplete `forward()` method and missing parameters.
- A training loop that handles batching and the forward pass.
- A sampling loop to generate new text using your trained model 2 functions are incomplete.

In the `CharDataset` class you will notice a concept of `stride` is used. When creating the training data for a character-level language model, we break long text into shorter overlapping sequences so the model can learn from many parts of the text.

This is most easily understood with an example. Lets say your training data is the sequence "abcedfgh" and you are learning a model for `sequence_length=3`. 

#### With `stride = 1`:

| Input  | Target |
|--------|--------|
| "abc"  | "bcd"  |
| "bcd"  | "cde"  |
| "cde"  | "def"  |
| "def"  | "efg"  |
| "efg"  | "fgh"  |

#### With `stride = 2`:

| Input  | Target |
|--------|--------|
| "abc"  | "bcd"  |
| "cde"  | "def"  |
| "efg"  | "fgh"  |

So as you can see, a higher stride results in less examples. This is a training hyperparameter which you can experiment with — smaller values increase data size and overlap, while larger values reduce redundancy and speed up training.

### Milestone 1. Teach an RNN the alphabet

Now that you've gone through the code it's time to implement the RNN and get the model to train on the alphabet sequence. Note once you've completed this your model should get a very high accuracy (close to 100%) as this is a very simple repeated sequence.

First, we'd recommend you complete the training section up until the training loop. Then, complete the model implementation. Then complete the training loop and try to train your model.

#### Training setup components

The code has a number of TODOs prior to the training loop, these should be pretty straightforward and are designed to help you understand the flow of the code by tieing in concepts from previous assignments.

#### RNN implementation

Inside `CharRNN.__init__()`, you’ll need to define the learned parameters of the RNN

**Your task**: Randomly initialize each parameter using `nn.Parameter(...)`, and follow the structure discussed in lecture. Keep standard deviations small (e.g., * 0.01).

Inside the `forward()` method:

```python
for t in range(l):
    #  TODO: Implement forward pass for a single RNN timestamp
    pass
```

Here you’ll implement the recurrence equation for the RNN. Each timestep receives:
- the current input embedding x_t
- the previous hidden state h_{t-1}

and outputs:
- the new hidden state h_t

**Your task**:
- Implement the RNN recurrence step
- Append the computed hidden to the `output` list
- Update `h_t_minus_1` to be the computed hidden for subsequent timesteps
- After the loop, compute:
  - `final_hidden` = create a `clone()` (deep copy) of your final hidden state to return
  - `logits` = result of projecting the full hidden sequence to the output space

---

#### Finish the training loop, test loop, and set the hyperparameters
Now that you've finished the model you have the forward pass established, finish the backward pass of the model using the PyTorch formula from Assignment 5 and create a test loop following a similar structure (don't forget to stop computing gradients in the test loop!).

Once that's done the code should start training when you run the file. However, it will not train successfully. In order to train the model properly you will need to update the training hyperparameters. If everything is set up properly at this point you should see a model that learns to predict the alphabet with very high accuracy 98+% and very low loss (near 0).

#### Hyperparmeter Tuning Tips

1. **Start with reasonable model parameters**

The first thing you should do is set reasonable starting hyperparams for the model itself. This will come to understanding what each hyperparams does by understanding the architecture and the objective you're training your model to complete. Set these and keep them fixed while you tune the training hyperparameters. As long as these are close enough the model will learn. They can be further refined once you have your training is starting to learn something.

2. **Refine learning rate**

When it comes to learning hyperparameters, the most important is learning rate. Others often are just optimizations to learn faster or maximize the output of your hardware. It's useful to imagine your loss space as a large flat desert. The loss space for neural networks is often very 'flat' with small 'divots' that are optimal regions. You want a learning rate that is small enough to be able to find these divots without jumping over them. Further you also want them to be small enough to reach the bottom of the divot (although optimizers these days often change your learning rate dynamically to accomplish this). I'd recommend starting with as small a learning rate as possible, if it's too small you're not traversing the space fast enough (never finding a divot, or only moving slightly into it). If this is the case, make it progressively larger, say by a factor of 10. Eventually you'll find a "sweet spot" and your model will learn.

3. **Refine other parameters**

Now that your model is learning something you can try to optimize it further. At this point try refining the model and other learning parameters. I wouldn't recommend changing the learning rate by much maybe only a factor of 5 or less.

### Milestone 2. Generating Text

Now that we've learned a model, let's use it to generate text. In this part of the assignment, your task is to implement the `generate_text` function, which uses a trained RNN model to generate text character-by-character, continuing from a given input. The function will produce an extended sequence by repeatedly predicting and appending the next character to the input.

#### `generate_text(model, start_text, n, k, temperature=1.0)`
- Take an initial input text of length n from the user, convert it into indices using a - predefined vocabulary (char_to_idx).
- Use a trained model to predict the next character in the sequence.
- Append the predicted character to the input, extend the input sequence, and repeat the process until k additional characters are generated.
- Return the generated text, including the original input and the newly predicted characters.

**Your task**: Generate text and test that you can generate an alphabet sequence from your trained model.

```
Enter the initial text: cde
Enter the number of characters to generate: 5
Generated text: fghijk
```

### Milestone 3. Predicting English Words

Now that you have trained the model on a simple sequence it's time to see how well it performs on an English corpus: `warandpeace.txt`. To do this, uncomment the read_file line at the beginning of the training section and re-run your code.

Now that we're using real data you will notice a few things, first the training will take much longer per epoch as the dataset is much larger. Second, training may not proceed as smoothly as it did before. This is because the relationships between characters in english is much more complex than in the simple sequence, so we will need to revisit our hyperparameters. 

**Your task**: Get your RNN working on the real data by adjusting your training hyperparameters.

#### Tips
In addition to the tips provided in Milestone 1, here's some specific tips.

1. If you use the full `warandpeace.txt` dataset you can get a well-trained model in **1 epoch**. And with a reasonable selection of hyperparameters, this epoch will take 5-10 minutes.

2.  If you don't see a significant jump after the first epoch, you shouldn't wait, change the parameters and try again. 

3. If you're losing patience, try taking a fraction of the dataset so you don't have to wait as long, and then run it on the full set after that's working. 

4. Don't expect a perfect model. What would it mean to have 90% accuracy on this model, is that realistic? You'd have created a novel writing masterpiece of a model! Realistically your performance will be much lower, around 50-60% with a loss around 1.5. But even with this "low performance" you should see words (or pseudo-words) in your output but not meaningful sentences.

### Milestone 4. Final Report

In your report, describe your experiments and observations when training the model with two datasets: (1) the sequence "abcdefghijklmnopqrstuvwxyz" * 100 and (2) the text from warandpeace.txt.

Include the final train and test loss values for both datasets and discuss how the generated text differed between the two. Explain the impact of changing the temperature parameter on the text generation, and provide examples. Reflect on the challenges you faced, your thought process during implementation, and the key insights you gained about RNNs and sequence modeling.

This section should be about 1-2 paragraphs in length and can include a table or figure if it helps your explanation. You can put this report at the end of this readme or in a separate markdown file.


## What to Submit

1. Your completed `rnn_complete.py` file with all TODOs filled in.
2. A PDF of your Final Report.

How to generate a pdf of your Final Report Section:
    
- On your Github repository after finishing the assignment, click on README.md to open the markdown preview.
- Use your browser 's "Print to PDF" feature to save your PDF.

Please submit to Assignment 7 Neural Complete on Gradecsope.

## TODO: Complete 383GPT Exit Survey
Link : https://forms.gle/YWBbJc3wDMu3U1CYA

Please also complete this 383GPT Exit Survey about your experience using 383GPT in this course. 

This survey is an important opportunity for us to gather feedback about how the tool supported your learning and how we can improve it for future classes.

Completion of the survey is worth 10 points as a part of Assignment 7 so please take the time to provide thoughtful and honest responses. It should take approximately 10 minutes to complete.

Your answers to this survey will in no way impact your grade in this course, please answer honestly.

## TODO: Fill out your Final Report here

How many late days are you using for this assignment? 3

1. Describe your experiments and observations

For the alphabet:
For the alphabet, since the sequence was very short, predictable and repetative, the hyperparameters didn't seem to matter that much. I found that a low sequence lenght and stride worked because of how small the sequence was. The hidden size, embedding dimensions, number of epochs and batch size could all be relativly low without effecting the loss and accuracy because of how predictable the data was, meaning not many features were needed for the model to learn the dataset. The learning rate could also be relativly high, allowing the model to converge quickly and for the loss to become very low without it fluctuating.

For war and peace:
I expiremented with many different hyperparameters but found the learning rate to be the most sensitive. If the learning rate was too high, such as at 0.01 then the model would often seem to overcorrect and the loss would fluctuate between epochs. With a low learning rate such as 0.0001 I noticed the loss barely changed between epochs. Higher sequence lengths seemed to allow the model to string more words together. If the sequence length was too low, such as 5, then the model couldn't really form any real words. Having a higher stride allowed the model to train faster by eliminating some redundant parts of sequences, but gave the model less data and if it was too high led to innaccutate models. A higher number of embedding dimensions increased accuracy but decreased performance. The hidden size also helped with accuracy but hurt performance and there were diminishing returns as the hidden layer size increased. Having more epochs also allowed for lower loss and higher accuracy but increasing the number of epochs greatly increased training time and quickly experienced diminishing returns in the improvments. Lower batch sizes caused training to take longer, and led to more loss.

2. Analysis on final train and test loss for both datasets

For the alphabet:
I chose sequence_length = 20, stride = 5, embedding_dim = 32, hidden_size = 32, learning_rate = 0.01, num_epochs = 20, batch_size = 64. I chose these hyperparameters because it allowed the model to get a very low loss of 0.0037 without taking more than a couple of seconds to train. Overall the model was very accurate and could always continue the alphabet from whatever the last character of the initial text sequence was, as long as the temperature wasn't too high.

For war and peace:
I chose sequence_length = 200, stride = 75, embedding_dim = 128, hidden_size = 128, learning_rate = 0.003, num_epochs = 5, batch_size = 64. I chose sequence length of 200 because I it gave enough context for the model to form words and start sometimes trying to make sentances without causing the training to take too long. I chose stride 75 because it gave a lot of context, but also didn't give the model too much redundant data to train off of. I chose an 128 embedding dimensions because it gave the model enough options to be able to identify a lot of different words and gramatical situations. I chose a hidden size of 128 to allow the model to be complex enough without overfitting or takeing too long to train. I chose a learning rate of 0.003 because it was the highest I could get the learning rate without causing the model to overcorrect and cause the loss to fluctuate between epochs. I did 5 epochs because more epochs signifigantly increased training time, but after 5 epochs, more epochs had little effect on the loss. I chose a batch size of 64 because it allowed for the training to happen relativly quick without using too much memory or overfitting. In the end the model was able to output some actual words, as well as some gibberish. That said, a lot of the gibberish was close to actual words, such as things like "helin" which isn't really a word, but has vowels in the correct places and could be a word. This showed me that the model might have learned some of the basic rules of english and how words are formed. Overall the training took less than 3 minuites and allowed for a test loss of 1.5767 and a training loss of 1.5373. The loss was relativly high but it seems like a larger dataset and more complex parameters would be needed for the model to be able to give accurate responses to prompts and have low loss.

3. Explain impact of changing temperature

For the alphabet:
Having a low temperatrue, below 1, allowed for the model to be perfect at writing the alphabet starting at the last character of the initial text. When the temperature got to around 1.3 it started to make some errors, occasionally missing a letter. Once the temperature was around 3, the model started giving somewhat random output, but still had traces of the alphabet. When the temperature was around 10, the output became very random.

For war and peace:
Lower temperatures, such as close to 0 allowed the model to generate actual english words. Often the words were common such as "the" or "had" but they were pretty much always words that appear in "war and peace". Increasing the temperature, such as having it at around 1-2 caused the model to start making up its own words, but they were often close to real words and followed the rules of english such as having appropriate spacing for vowels and consonants. If the temperature got too high, the output would become very random, such as generating words with no vowels, or with punctuation in the middle of words.

4. Reflection

The process of training the RNN model using both the repetitive sequence "abcdefghijklmnopqrstuvwxyz" * 100 and the text from *War and Peace* provided me with valuable insights into the nuances of sequence modeling and the behavior of recurrent neural networks. When training on the alphabet sequence, the model quickly learned to predict the next character due to the simplicity and uniformity of the data. The final train and test loss values were significantly lower, often nearing zero, demonstrating that the model could memorize the sequence and generate output that closely mirrored the input. The generated text was mostly accurate repetitions of the input sequence, illustrating how RNNs can effectively capture short-term dependencies in data.
In contrast, when training on the *War and Peace* text, the loss values were higher, reflecting the complexity and variability of natural language. The model struggled to generate coherent text at times, as it had to contend with the diversity of sentence structures, vocabulary, and thematic elements inherent in the literary piece. The generated text often displayed a better flow and context when the temperature was set lower (e.g., 0.5), producing more predictable and coherent outputs. However, as the temperature increased (e.g., 1.5), the results became increasingly random and less grammatically correct, showcasing how temperature can influence creativity in text generation. This highlighted an essential aspect of working with RNNs: tuning hyperparameters like temperature can significantly impact the model's output quality.
Throughout my implementation, I faced challenges such as overfitting and the difficulty of optimizing hyperparameters for both datasets. Balancing the trade-off between training loss and generalization to the test set required careful monitoring and adjustments. These challenges reinforced the importance of validation in machine learning and the need for a systematic approach to model training. Overall, this project deepened my understanding of RNNs, particularly how they learn from sequences and the impact of different training data characteristics and hyperparameter choices on model performance.