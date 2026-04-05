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

How many late days are you using for this assignment? 2 DAYS

0. Hyperparameter and Metric Tables

Alphabet:

| Sequence Length | Stride | Embedding Dim | Hidden Size | Learning Rate | Num Epochs | Batch Size | Test Accuracy | Avg Training Loss | Avg Test Loss |
| --------------- | ------ | ------------- | ----------- | ------------- | ---------- | ---------- | ------------- | ------------- | --------- |
| 10              | 1      | 10            | 10          | 0.01          | 10         | 64         | 100.00%       | 0.3481        | 0.0142    |

War and Peace:

| Hyperparameter Configuration Number (HCN) | Sequence Length | Stride | Embedding Dim | Hidden Size | Learning Rate | Num Epochs | Batch Size | Test Accuracy | Avg Training Loss | Avg Test Loss | Notes                                                                                              |
| ----------------------------------------- | --------------- | ------ | ------------- | ----------- | ------------- | ---------- | ---------- | ------------- | ------------- | --------- | -------------------------------------------------------------------------------------------------- |
| 1                                         | 10              | 1      | 10            | 10          | 0.01          | 1          | 64         | 33%           |               |           |                                                                                                    |
| 2                                         | 100             | 5      | 10            | 10          | 0.01          | 1          | 64         | 35%           |               |           |                                                                                                    |
| 3                                         | 100             | 5      | 10            | 10          | 0.001         | 1          | 64         | 33.50%        | 2.3838        |           |                                                                                                    |
| 4                                         | 100             | 5      | 10            | 10          | 0.1           | 1          | 64         | 31.32%        | 2.3163        |           |                                                                                                    |
| 5                                         | 100             | 10     | 100           | 100         | 0.01          | 1          | 64         | 49.31%        | 1.7128        | 1.7161    |                                                                                                    |
| 6                                         | 100             | 10     | 10            | 100         | 0.01          | 1          | 64         | 37.50%        | 1.7353        | 3.7742    | Checking if the improvement was from embedding dim or hidden size (or both)                        |
| 7                                         | 100             | 10     | 100           | 10          | 0.01          | 1          | 64         | 34.26%        | 2.2948        | 2.2614    | Checking if the improvement was from embedding dim or hidden size (or both)                        |
| 8                                         | 250             | 25     | 100           | 100         | 0.01          | 1          | 64         | 49.98%        | 1.7197        | 1.6881    | Increasing sequence length (and stride, so we don't increase the amount of time to train too much) |
| 9                                         | 100             | 10     | 200           | 200         | 0.01          | 1          | 64         | 47.63%        |  1.6661       | 1.7697    |  |

1. Describe your experiments and observations

Alphabet:

Using the hyperparameters as above for the alphabet, the model was able to get perfect accuracy on the test set after less than a second of total training time, which makes sense given such a simple input. When using the model, you have to crank up the temperature to something like 10 before it starts outputting stuff that isn't just the perfect alphabet.

War and Peace:

In HCN1 (hyperparameter configuration number 1, aka the first row of the table), the model outputted fairly reasonable outputs that felt almost like a baby speaking. For more details see Section 3 where I go into more detail about how the temperature affected the output.

In HCN5, with temperature of 0.6-0.75, the model actually outputted fairly reasonable text that seemed english-like. The sentences were much longer than expected or had no periods at all, and the grammar was basically nonsense, but the words by themselves and the "shape" of the sentences were reasonable. A higher temperature leads to too much randomness and nonsense words, while a lower temperature leads to too much repetition.

For HCN6 and HCN7 I tried to isolate if the improvement from 4 to 5 came mostly from the embedding dim increase or the hidden size increase, but those experiments implied that both of those increases to 100 were probably important. 

In HCN8 I further increased the sequence length (and also increased stride in compensation to make sure training time wasn't too long). It did similarly well to HCN5 and had similar-feeling outputs.

In HCN9 I further increased hidden size and embedding size but it didn't do as much as I had hoped. Looking at the graph of the training losses, it seemed to get a little confused about 3/4 into the book.... maybe something changed, or just some randomness caused the model to increase loss. At the end, the test accuracy was slightly worse than HCN8.

Also one other small observation was that moving torch's tensors to MPS (metal performance shaders) still was substantially slower than just using the CPU on my M1 Mac. This might be because the M1 Mac CPU is already pretty fast, or maybe because the matrices we are dealing with (100x100 etc) are still not big enough for graphics card speedup to outpace the overhead of moving stuff back and forth from CPU to MPS, or maybe just that the torch MPS integration isn't perfect yet (it is still in open beta testing).

2. Analysis on final train and test loss for both datasets

Alphabet:

Final train loss was 0.0158 (avg train loss 0.3481), and the average test loss was 0.0142; and the test accuracy was 100%, as expected. 

War and peace: My best test accuracy was 49.98% with average training loss of 1.7197 and test loss of 1.6881 (remember this is avg training loss, which is why train loss looks worse than test loss, since train loss starts high). This resulted in a model that can create words pretty well (with relatively low temperature), but doesn't really understand grammar or sentence structure. 

3. Explain impact of changing temperature

In HCN1, a low temperature (close to 0) leads to the model having a couple of reasonable characters, reaching a space, and proceeding to repeat "the the the the" forever, which is very similar to assignment 6's language model when it had a decent context size. A temperature of 1 gives a bunch of almost-gibberish words, although if you squinted and looked at it from far away, the sentence structure almost looks reasonable. At temp=0.5 it goes back and forth between repeating "the the the" and having some interspersed words randomly.

This phenomenon happens similarly in better models; in HCN8, my best model, low temperature (near 0) just repeats the same thing over and over again, while high temperature (1 or more) results in much more gibberish. I found the sweet spot between 0.5 and 0.75, where it has enough randomness to not spit out the same few words over and over, but enough stability to have real English words instead of gibberish.

4. Reflection

The models we created are able to generate small packets of meaning, like words, fairly well. The models of both assignment 6 and assignment 7 apply to this. The RNN allows for more randomness than the previous language model, though, and I think it makes it a better model. The assignment 6 model often just recited stitched-together passages of war and peace. They both had a tendency to repeat common words when certain conditions were met (for assignment 6, when the context window was just right, and for assignment 7 when the temperature was low).

Implementing the assignment 6 model felt like more of an algorithms assignment, where you had to make sure the code worked correctly. But once you "get it", it's done. In contrast, the RNN training process has a lot of trial and error, where you have to check your code to make sure the code is working, then also tweak hyperparameters to allow the model to learn effectively. Once you "get it", you can still maybe do better by optimizing hyperparameters.

The main general thing I learned, though, is that sequence modeling in general is very hard. Even the best outputs from either model are still complete nonsense, and it's no wonder that to get true turing-test-passing writing you would need hundreds of billions to trillions of parameters like in the biggest LLMs today (sidenote: yes I think most LLMs pass the original Turing Test nowadays). 

5. Screenshot Examples

HCN5
<img width="1006" alt="Screenshot 2025-05-04 at 10 52 40 PM" src="https://github.com/user-attachments/assets/947d92e9-89c1-4b50-b1df-2e161e6ec8e3" />
<img width="1015" alt="Screenshot 2025-05-04 at 10 52 05 PM" src="https://github.com/user-attachments/assets/c8ab1b5f-d3e6-447f-aab6-1dc180fd132a" />

HCN8
<img width="909" alt="Screenshot 2025-05-04 at 11 09 14 PM" src="https://github.com/user-attachments/assets/cd77c026-292c-4085-863a-4015a98fd7a2" />
<img width="912" alt="Screenshot 2025-05-04 at 11 08 46 PM" src="https://github.com/user-attachments/assets/3c77f595-d4a2-43c1-a175-7df0816fa9b5" />
<img width="1006" alt="Screenshot 2025-05-04 at 11 08 03 PM" src="https://github.com/user-attachments/assets/f0375a6e-0c17-4d85-be6e-162dd5dd848e" />
<img width="1007" alt="Screenshot 2025-05-04 at 11 07 36 PM" src="https://github.com/user-attachments/assets/ab1e126a-6ab4-4a0c-95d2-bdc42c5c001e" />
