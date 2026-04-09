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

How many late days are you using for this assignment?: 4 days.

1. Describe your experiments and observations

Prior to any test, my first adjustment to the hyperparameters was to bring the `sequence_length` down to 100, as I believed that 1000 would take an excessively long time to run. I also noticed that the `learning_rate` was quite high (200), and a high learning rate can easily lead to the model not converging, so I reduced it to 0.01.

Since the hidden state for RNNs acts as a sort of memory that stores information over time, I felt that a `hidden_size` of 1 would not be sufficient to retain much information, so I increased it to 128. I raised the `embedding_dim` to 30 as well but this should make no different as it is a default setting in the function. For the `stride`, which controls how much of the text we skip during training, I lowered it to 5 for the time being, despite it potentially extending the training duration. The rest of the parameters remain as given.

With these settings applied to warandpeace.txt, I obtained a training loss of 1.66 and a testing loss of 1.80, with accuracies of 0.51 for training and 0.48 for testing. I tried raising the `embedding_dim` up to 128 but it doesn't reduce the loss value. However, it increase runtime significantly so I reverted it back to the default value of 30. This configuration didn't work well with the alphabet dataset, as we encountered the issue of the dataset not being long enough. This causes a zero division error as there is no batch for the test. Therefore, I reduced the `batch_size` back to 10, and it managed to predict results almost perfectly (0.97 for training and 1.00 for testing).

Because this setting perform really well for the alphabet dataset, I shifted my focus back to tuning the parameters for warandpeace.txt. Through more trial and error, I noticed that increasing the `sequence_length` generally decreased both training and testing loss, but if raised too much, it could lead to overfitting (a much higher loss in testing compared to training). I found that the sweet spot was around 150.

After several rounds of experimentation and adjusting the parameters to minimize overfitting while achieving the lowest possible loss, I arrived at a set of parameters that yielded a training loss of approximately 1.62 and a testing loss of 1.64, both with an accuracy of 0.52. I believe that increasing the `batch_size` was the most effective way to reduce the loss without overfitting. The final set of parameters is: `sequence_length = 150`, `stride = 5`, `embedding_dim = 128`, `hidden_size = 128`, `learning_rate = 0.01`, `num_epochs = 1`, and `batch_size = 200`.

2. Analysis on final train and test loss for both datasets

Due to the simple pattern of the alphabet data, just a slight change from the originally given hyperparameters to more reasonable ones (as mentioned in the first question) was able to achieve nearly perfect predictions in training and perfect predictions in testing. However, the text from "War and Peace" is much more complicated and significantly longer, thus require more sophisticated hyperparameter tuning. The testing results after tweaking some parameters yielded a loss of roughly 1.64 and an accuracy of 0.52, which I would say is quite decent considering that we have only been training for one epoch. This indicates that the model has already learned some patterns in the English text, and we can see that it is true, as it outputs somewhat recognizable English words. Had we trained for longer, I believe we would have achieved better loss and accuracy.

3. Explain impact of changing temperature

Changing temperature directly alters the randomness of the output. For the set of parameter that I have achieved, I tested how temperature affect the output by running the program with the same text "they do not" and set the program to fill up 50 character with different temperature.

Results
| Temperature | Initial sentences | Output |
|----------|----------|----------|
| -1 (0.00000001) | they do not | something the same the same the same the same the |
| 0.5 | they do not | all them so have the ground to the same of showin |
| 0.7 | they do not | approod live and the merranged the counted nunion |
| 1 | they do not | he glance. wiled it that ofthe opened. but have a |
| 2 | they do not | fish pus! evpnnisepari wnvibpeoa.ampeats!kyouharat |

We can see that when the temperature is set to -1 (or 0.00000001, as interpreted by the program), the results are very deterministic, switching between only two words. As we increase the temperature, the sentences begin to include more varied word choices, forming somewhat English-like sentences. However, after the temperature exceeds 0.7, the coherence of the sentences starts to fall apart. It becomes noticeably harder to make out words within the sentences. By the end of my experiment, with the temperature set to 2, the sentences are now filled with random letters, with barely any actual words present.

4. Reflection

At the beginning of this assignment, I tried to look into the starter code but ended up having no idea where to start. Therefore, my approach was to ensure that I truly understood how RNNs work first, then attempt to write the code. This helped a lot, as I now understand why the code is structured the way it is and how the different components are supposed to work together. I believe my understanding of RNNs has greatly improved after this project. I found the topic of how the loss function works for RNNs particularly challenging.

Some challenges that I faced while working on this project were related to hyperparameter tuning. It is fairly easy to establish a somewhat acceptable baseline since I can adjust the parameters to be similar to those I have seen before in both my personal projects and online (e.g., learning rate, batch size, etc.). However, tuning each hyperparameter by hand further is extremely difficult because it may depend on other parameters. For example, a certain value of the learning rate might not work well now, but it could be effective if I change another parameter. It also takes quite a long time to run, and reducing the size of the data ended up affecting the loss (so when I achieve a good loss for the half dataset, it sometimes turns out to be bad for the full dataset).

5. `generate_text()` output

<div style="display: flex;">
  <img src="results.png" style="width: 100%; margin-right: 10px; padding-bottom: 20px;">
</div>
