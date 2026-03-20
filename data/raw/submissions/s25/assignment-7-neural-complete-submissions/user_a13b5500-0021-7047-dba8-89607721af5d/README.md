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
   
| Hyperparameter       | Alphabet Sequence | War and Peace |
|----------------------|-------------------|---------------|
| `sequence_length`    | 26                | 100           |
| `stride`             | 1                 | 10            |
| `embedding_dim`      | 32                | 64            |
| `hidden_size`        | 64                | 256           |
| `learning_rate`      | 0.01              | 0.001         |
| `num_epochs`         | 10                | 1             |
| `batch_size`         | 16                | 64            |


   
- I conducted experiments with the CharRNN model on two datasets: (1) a repeating alphabet sequence ("abcdefghijklmnopqrstuvwxyz" * 100, 2600 characters) and (2) the English corpus from warandpeace.txt (~500k–600k characters). For the alphabet sequence, I experimented with various hyperparameter values until the model was able to predict perfectly due to the small size and nature of the dataset. The model trained quickly (~0.5 seconds per epoch) and mastered the predictable pattern. The model converged rapidly, with loss dropping to ~0.0000 by epoch 9, indicating near-perfect prediction. The fast training time reflects the small dataset (2600 characters) and simple, deterministic pattern.

-  For War and Peace, I switched to sequence_length=100, stride=10, embedding_dim=64, hidden_size=256, learning_rate=0.001, batch_size=64, and num_epochs=1. Training took around 5 minutes, and the model produced word-like sequences, though not coherent sentences, as expected. The effect of changing num_epochs = 2 doubled the training time to around 10 minutes but yielded no significant reduction in loss (~1.63 train, ~1.55 test), suggesting the model had already captured most learnable patterns in the first epoch or was limited by the current hyperparameter configuration. I found that a larger stride reduces the number of samples significantly because the sequences will be less overlapping. This can speed up training because there are fewer examples to process, but it may decrease the amount of information the model can learn. I found that the sweet spot was 10 for me.

2. Analysis on final train and test loss for both datasets

- For the alphabet sequence, the training loss dropped from 0.1779 to 0.0000 over 10 epochs, with a test loss of 0.0000, suggesting 100% accuracy. Generated text (e.g., input: cde, k=10, output: cdefghijklmno) was accurate, reflecting the dataset’s simplicity.The model performed exceptionally well on the simple sequence due to the repetitive and predictable nature of the task, which was reflected in the minimal loss values. This high performance suggests that the model learned to accurately output characters following the observed patterns.
- For War and Peace, one epoch yielded a training loss of 1.6327 and a test loss of 1.5461. Extending to two epochs provided no loss reduction, doubling training time to ~10 minutes without improving text quality. This suggests the model reached its capacity limit or required hyperparameter adjustments (e.g., higher hidden_size or learning_rate) for further gains. The higher loss reflects the complexity of English text, where predicting the next character is less deterministic. Generated text  contained pseudo-words and partial phrases, indicating the model learned character patterns but not semantic meaning. I found that Changing learning_rate significantly affects training stability and convergence speed. I found that a learning rate of 0.001 was optimal for me.

4. Explain impact of changing temperature
- The temperature parameter influences the randomness of predictions made by the model during text generation:
Lower Temperatures (e.g., 0.5): Produce outputs that are more conservative and closer to the training examples. 
Higher Temperatures (e.g., 1.5): Create more diverse outputs that can be imaginative but often lack coherence. The same input with higher value of temperature might yield odd responses, blurring meaning and creativity.
Throughout my tests, the outputs varied significantly with temperature adjustments. While lower temperatures tended to produce more conventional text, higher temperatures encouraged creative but often less meaningful language, showcasing how the temperature setting serves as a control dial for creativity versus coherence.

- I experimented with different temperature values on both datasets and found that there was essentially no change with the alphabet dataset, which makes me think that the model had already learned the entire pattern.
- With the war and peace dataset, here are my experiemnts:

| Initial Text | k (Characters to Generate) | Temperature | Generated Text |
|--------------|---------------------------|-------------|---------------|
| It was the feeling that induces a volunteer recruit to spend | 30 | 0.5 | it was the feeling that induces a volunteer recruit to spend the doors and said to the sam |
| It was the feeling that induces a volunteer recruit to spend | 30 | 1.2 | it was the feeling that induces a volunteer recruit to spend to teater, at five letter hum |
| It was the feeling that induces a volunteer recruit to spend | 30 | 1.5 | it was the feeling that induces a volunteer recruit to spend!hiesglanc, sol lisspea blurgu |

5. Screenshot
   
![image](https://github.com/user-attachments/assets/66e9828e-71e5-445a-933c-c6b48f6de60f)

   

6. Reflection
   - This project highlighted the contrast between modeling simple and complex sequences with RNNs. The alphabet sequence was straightforward—its simple pattern let the model hit near-perfect accuracy, and getting outputs like `cdefghijklmno` felt like an early win, but War and Peace posed challenges in hyperparameter tuning and training stability. Tuning learning_rate and hidden_size for War and Peace document required a lot of iterative experimentation. I ran into a divide by 0 error in the test loop which stumped me for a while. I ultimately fixed this by changing and tuning the hyperparameters which made me realize just how important a change in these values are to the overall result of the model.
   - I learned that RNNs excel at capturing short-term dependencies, but their performance on complex texts is limited by probabilistic patterns. The temperature parameter’s role in balancing determinism and creativity was a key insight, emphasizing the probabilistic nature of language modeling. This experience deepened my understanding of sequence modeling and the practical challenges of training neural networks. It was challenging but rewarding to see how small changes could make a big difference in modeling sequences.
