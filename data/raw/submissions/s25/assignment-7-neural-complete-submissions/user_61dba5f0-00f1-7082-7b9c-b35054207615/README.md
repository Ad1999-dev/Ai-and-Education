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

How many late days are you using for this assignment? 
2

1. Describe your experiments and observations

      When using the "abc" sequence I tested different number epochs with stride 1 and a learning rate of 0.005. With only 3 epochs the neural network was able to achieve a loss of 0.0015 and with 10 epochs the loss was decreased to only 0.0002. However, there was no difference in character completion since the loss was already so small, they both were perfect in predicting the next character in the alphabet. For example, when I entered "abc" as the initial text, both model generated "defghij" as the next 7 characters when the temperature was below 1. 

      Training on the warandpeace.txt took roughly 17 minutes for one epoch when using a stride of 8 and a learning rate of 0.005. The train loss was 1.6608 and the test loss was 1.6623. The letters and words that it generates is still no very coherent but this is still true even when I tried it with 3 epochs. The loss was down to around 1.5 but it still was not able to string together two words that makes sense. I think I might have to use a smaller stride or a larger embedding dimension, but that would take too long, since I tried to lower the stride to 5 and keep everything the same but it was not even able to complete 1 epoch within 30 minutes. Screenshots of the output is in section 5.

      I also tried a sequence length of 20 instead of 100 and the loss for one epoch became roughly 1.8, however the quality of the generated words seemed to be no different. Maybe I would need to try to generate a lot more characters in order to see a difference in quality between a loss of 1.6 versus a 1.8 

2. Analysis on final train and test loss for both datasets

      THe final train and test loss for the first dataset of abc's was near perfect and it did not take long. I used the following hyperparameters for it:

      sequence_length = 100
      stride = 1
      embedding_dim = 30
      hidden_size = 128
      learning_rate = 0.005
      num_epochs = 10
      batch_size = 64 

      and it resulted in a training loss of 0.0005 and a test loss of 0.0002.

      When using the same parameters for the war and peace dataset, but with the stride changed to 8, and the epochs to 3, the final train loss was 1.5620 and the test loss was 1.6236 (shown below). The amount of loss barely decreased with each epoch but the training time was faster with each epoch, which is expected.

      Using device: cpu
      Epoch 1/3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5484/5484 [20:22<00:00,  4.49it/s]
      Epoch 1, Train Loss: 1.6465
      Epoch 1, Test Loss:  1.6394

      Epoch 2/3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5484/5484 [10:17<00:00,  8.88it/s]
      Epoch 2, Train Loss: 1.5673
      Epoch 2, Test Loss:  1.6233

      Epoch 3/3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5484/5484 [06:50<00:00, 13.36it/s]
      Epoch 3, Train Loss: 1.5620
      Epoch 3, Test Loss:  1.6236

3. Explain impact of changing temperature

    Changing the temperature causes the generated text to be more random and incoherent. With a temperature of 0 it would sometimes generate a few complete words strung together, whereas a higher temperature such as 1 or even greater leads to less words and just letters appended onto my initial provided text. 

    For example, this is the difference between a temperature of 0, 1, and 5 respectively:

    then the same the same

    thethe set the gom he u

    thezvndaz,yiapvpfghmzw

4. Reflection

    I found this assignment to be really helpful in making me understand the many different parts needed to form a complete recurrent neural network. When testing my code I found that the following line gave me errors sometimes depending on the length of my stride:

    h_t = torch.tanh(x_t @ self.W_xh + h_t_minus_1 @ self.W_hh + self.b_h)

    it took me a while to realize that it was because the last batch in the loader was sometimes not the same size as the full batches and so it would throw the error "RuntimeError: The size of tensor a (12) must match the size of tensor b (64) at non-singleton dimension 0" I found that the easiest way to solve this was to just reset the hidden state each batch to None. This way I dont mix a 64 sized batch to a size 12 batch. 

    I also found that it was very difficult to decide on the hyperparameters since it all seemed extremely arbitrary to me. I think that it required a certain combinaition  of knowledge and experience before I knew where to even begin. Another barrier was how long it took to train when using the war and peace dataset. This made me less trigger happy to just randomly try different parameters and see what happens.

5. IMPORTANT: Include screenshot of output for generate_text() function: 

    ![Drag Racing](1.jpg)
    ![Drag Racing](2.jpg)
    ![Drag Racing](3.jpg)
    ![Drag Racing](4.png)
    ![Drag Racing](5.jpg)