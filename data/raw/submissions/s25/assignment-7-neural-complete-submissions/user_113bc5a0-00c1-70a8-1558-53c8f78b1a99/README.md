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

Start by opening `char_rnn_starter.py` reading through whats provided and familiarizing yourself with the structure.

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

1. Your completed `char_rnn_starter.py` file with all TODOs filled in.
2. A PDF of your Final Report.

How to generate a pdf of your Final Report Section:
    
- On your Github repository after finishing the assignment, click on README.md to open the markdown preview.
- Use your browser 's "Print to PDF" feature to save your PDF.

Please submit to Assignment 7 Neural Complete on Gradecsope.


## TODO: Fill out your Final Report here

How many late days are you using for this assignment? ZERO

1. Describe your experiments and observations 
In this assignment, I experimented with a character-level RNN model for text prediction and generation. The experiments were conducted on two datasets:

Alphabet Dataset: A simple repeating sequence of the alphabet ("abcdefghijklmnopqrstuvwxyz" repeated 100 times)
War and Peace: A complex English text corpus with rich linguistic patterns

For the alphabet dataset, I observed that the model reached perfect accuracy very quickly with appropriate hyperparameters. The sequence was highly predictable, allowing the model to learn the pattern completely.
For War and Peace, the training was more challenging. The model needed to capture complex relationships between characters in natural language. I observed that:

Training took significantly longer per epoch due to the larger dataset
The model required higher capacity (larger hidden size and embedding dimensions)
Performance metrics were more modest, as expected given the complexity of the data. I got 55.66% accuracy.

2. Analysis on final train and test loss for both datasets
Alphabet Dataset:

Training Loss: ~0.0000 (converged to essentially zero)
Test Loss: ~0.0000
Test Accuracy: 100%

The perfect performance was expected given the simplicity and deterministic nature of the repeating alphabet pattern. The model completely memorized the sequence and could predict with perfect accuracy.
War and Peace Dataset:

Training Loss: 1.5959
Test Loss: 1.5017
Test Accuracy: 55.75%

The higher loss and lower accuracy for War and Peace reflect the inherent complexity and unpredictability of natural language. A character-level model can't perfectly predict the next character in many contexts because multiple valid continuations exist in English text and thus any prediction is not 100% certain to be correct. For example, after "th" could come "e", "a", "i", "o", etc., depending on the word. 
The similar training and test loss indicates the model didn't overfit, suggesting the hyperparameters were appropriate for the task which I am happy with.

3. Explain impact of changing temperature
Temperature is a hyperparameter that controls the randomness in text generation. My experiments with different temperature values showed:

Low temperature (0.1-0.5): Generated text was more deterministic and repetitive. The model consistently chose the most probable next character, resulting in common words and patterns but less creativity. The text often got stuck in loops or repeated common phrases.
Medium temperature (0.7-1.0): Provided a good balance between predictability and creativity. The generated text contained recognizable words and occasionally meaningful phrases while avoiding excessive repetition.
High temperature (1.5+): Produced more random and diverse text. The model explored less probable character sequences, resulting in more creative but often nonsensical combinations. This occasionally produced interesting novel word-like structures but reduced the coherence of the generated text.

Temperature adjustment allows for controlling the trade-off between accuracy/ most probable and creativity in text generation, which makes it a powerful tool for applications where different levels of randomness might be desirable.

4. Reflection
This assignment provided valuable insights into sequence modeling using RNNs. I learned how the hidden state in RNNs preserves information across time steps, enabling the model to capture sequential patterns.Having to implement the forward pass with the recurrence equation deepened my understanding of how information flows through RNNs. The experiments highlighted the importance of appropriate hyperparameter selection and were fun to do. Parameters like hidden size, learning rate, and sequence length dramatically affected training dynamics and model performance. I found a  greater hidden size is better for a larger corpus. The stark difference in performance between the simple alphabet sequence and War and Peace demonstrated how data complexity affects model performance. It showed that while neural networks excel at memorizing patterns, capturing the nuances of natural language remains extremely to challenging to do. Even we humans have trouble with it. While the model could generate text with believable character transitions, it lacked understanding of higher-level linguistic structures like grammar or semantics. This illustrates the limitations of character-level models compared to word-level or more sophisticated approaches. The temperature parameter experiments showed the inherent trade-off between predictability and creativity in generative models, a concept that extends beyond text generation to other creative AI applications and was exciting to play with as I tested my LLM.

Overall, this assignment provided hands-on experience with fundamental concepts in sequence modeling and natural language processing, while also illustrating both the capabilities and limitations of simple RNN architectures for text generation tasks. 

## SECTION 5: Screenshot of the output for generate_text() function

![Screenshot terminal outputs generate_text()](383HW7screenshot.png)
