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

1. Your completed `rnn_complete.py` file with all TODOs filled in.
2. A PDF of your Final Report.

How to generate a pdf of your Final Report Section:
    
- On your Github repository after finishing the assignment, click on README.md to open the markdown preview.
- Use your browser 's "Print to PDF" feature to save your PDF.

Please submit to Assignment 7 Neural Complete on Gradecsope.

## TODO: Fill out your Final Report here

**How many late days are you using for this assignment?**  
5

---

### 1. Describe your experiments and observations

I trained a character-level RNN on two datasets: a repeating alphabet string 
and the full text of *War and Peace*. For each, I ran three experiments: a 
baseline with minimal capacity, a faster configuration with more efficient 
settings, and a larger model with higher capacity.

The alphabet model learned very quickly due to its simplicity. Even small 
models achieved near-zero loss. In contrast, *War and Peace* required a larger 
hidden size and more tuning to make progress. Increasing model capacity led to 
clear improvements, especially in more complex data.

Due to the alphabet dataset being much shorter than War and Peace, I reduced 
sequence_length, stride, and batch_size for those experiments to ensure enough 
samples were generated for training and testing. These changes didn’t affect model 
architecture, only the way data was prepared and batched.

---

### 2. Final train and test loss for both datasets


| Dataset       | Exp | Description        | Train Loss | Test Loss |
|---------------|-----|--------------------|------------|-----------|
| Alphabet      | 1   | Baseline           | 604.3468   | 272.1562  |
| Alphabet      | 2   | Faster Config      | 0.8615     | 0.0494    |
| Alphabet      | 3   | Higher Capacity    | 0.0050     | 0.0019    |
| War and Peace | 1   | Baseline           | 414.2027   | 411.7207  |
| War and Peace | 2   | Faster Config      | 1.8912     | 1.8048    |
| War and Peace | 3   | Higher Capacity    | 1.4116     | 1.4979    |

---

### Parameters for Each Experiment

#### Alphabet – Experiment 1 (Baseline)
- `sequence_length = 50`
- `stride = 5`
- `embedding_dim = 2`
- `hidden_size = 1`
- `learning_rate = 200`
- `num_epochs = 1`
- `batch_size = 8`

#### Alphabet – Experiment 2 (Faster Config)
- `sequence_length = 50`
- `stride = 5`
- `embedding_dim = 16`
- `hidden_size = 64`
- `learning_rate = 0.003`
- `num_epochs = 1`
- `batch_size = 8`

#### Alphabet – Experiment 3 (Higher Capacity)
- `sequence_length = 50`
- `stride = 5`
- `embedding_dim = 64`
- `hidden_size = 256`
- `learning_rate = 0.003`
- `num_epochs = 3`
- `batch_size = 8`

#### War and Peace – Experiment 1 (Baseline)
- `sequence_length = 1000`
- `stride = 10`
- `embedding_dim = 2`
- `hidden_size = 1`
- `learning_rate = 200`
- `num_epochs = 1`
- `batch_size = 64`

#### War and Peace – Experiment 2 (Faster Config)
- `sequence_length = 100`
- `stride = 20`
- `embedding_dim = 16`
- `hidden_size = 64`
- `learning_rate = 0.003`
- `num_epochs = 1`
- `batch_size = 64`

#### War and Peace – Experiment 3 (Higher Capacity)
- `sequence_length = 100`
- `stride = 10`
- `embedding_dim = 64`
- `hidden_size = 256`
- `learning_rate = 0.003`
- `num_epochs = 3`
- `batch_size = 64`

---

Across experiments, I varied the model's capacity and the granularity of input slicing.  
Experiment 1 used minimal resources and an overly aggressive learning rate to illustrate poor convergence.  
Experiment 2 focused on fast, stable training with a moderate-sized model and shorter sequences.  
Experiment 3 emphasized learning capacity with deeper embeddings and longer training over multiple epochs.  
The alphabet dataset required lower `sequence_length` and `batch_size` due to its limited size, while *War and Peace* allowed for larger settings.

### 3. Explain the impact of changing temperature

To evaluate how temperature affects output diversity and coherence, I used the 
model from **War and Peace, Experiment 2**, and generated 40 characters using 
the prompt `"and"` at different temperature values:

**Prompt**: "and", **Temperature = 0.1**  
> andread and he was the seemed the seemed th

**Prompt**: "and", **Temperature = 0.5**  
> andressed catter and been of the from her f

**Prompt**: "and", **Temperature = 0.8**  
> andenstances to so the brownentensally of h

**Prompt**: "and", **Temperature = 1.0**  
> andron solptale!orvered princessbeck? ather

**Prompt**: "and", **Temperature = 1.5**  
> andrown meselutth, thatifi brofusgofn watp,

At low temperatures (e.g., 0.1–0.5), the output is repetitive and more likely 
to echo known patterns in the training data. As temperature increases, 
diversity grows, but grammatical and semantic coherence decline. By 1.5, the 
model explores highly unlikely character sequences, leading to novel but 
nonsensical outputs.

This behavior shows how temperature acts as a knob to balance predictability 
and creativity in character-level text generation.

---

### 4. Reflection

This assignment helped clarify how character-level language models operate, 
especially in contrast to word-based models. Implementing the RNN manually 
forced me to understand the recurrence step in detail, including how the hidden 
state is propagated and updated at each timestep.

The clearest insight came from directly observing how model capacity and 
training configuration impacted both loss and output quality. In particular, 
the difference in performance between Experiment 1 and Experiment 2 on War and 
Peace was striking: simply changing the learning rate and model size led to a 
300x improvement in loss. It became clear that some hyperparameters (like 
learning rate and sequence length) are far more sensitive than others.

Working with both datasets also gave me a sense of how to adapt architectures to 
fit data scale. The alphabet model reached near-zero loss easily, but its size 
forced me to rethink how I created sequences — adjusting stride, sequence 
length, and batch size just to ensure I had a test set at all. By contrast, 
War and Peace provided ample data, but required a more expressive model to make 
progress.

Finally, the temperature experiments at the end really demonstrated the effect 
of randomness in generation. I could see how small shifts in temperature 
produced drastically different styles of text — from deterministic repetition to 
chaotic nonsense. It reminded me that sampling strategy is just as important as 
the model itself when it comes to generating usable or creative language.

Overall, this project deepened my understanding of sequential modeling, RNN 
dynamics, and how design choices ripple through training, evaluation, and 
generation. It also made me appreciate how even simple models can feel 
“intelligent” when tuned carefully.


### 5. Screenshot Proof

This was done with the hyperparameters in Experiment 3 of warandpeace which can be viewed in the above sections. 
<img width="952" alt="Screenshot 2025-05-05 at 11 50 26 PM" src="https://github.com/user-attachments/assets/6d8b8110-8c15-4c6f-a0b5-cbef66f39eeb" />


