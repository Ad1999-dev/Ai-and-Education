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

How many late days are you using for this assignment? 2 days 

1. Describe your experiments and observations

Starting off the first alphabetical sequence I tried tweaking it with a few parameters which are as follows and yielded their respective MSE. 

sequence_length = 50 # Length of each input sequence
stride = 20            # Stride for creating sequences
embedding_dim = 30      # Dimension of character embeddings
hidden_size = 32        # Number of features in the hidden state of the RNN
learning_rate = 0.1    # Learning rate for the optimizer
num_epochs = 5         # Number of epochs to train
batch_size = 128        # Batch size for training

Test Loss: 1.0243

Epoch 1/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.74it/s]
Epoch 1, Loss: 3.2571
Epoch 2/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.44it/s] 
Epoch 2, Loss: 3.1348
Epoch 3/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 11.10it/s] 
Epoch 3, Loss: 2.8598
Epoch 4/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.91it/s] 
Epoch 4, Loss: 2.3578
Epoch 5/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.46it/s] 
Epoch 5, Loss: 1.6816
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 56.32it/s] 
Test Loss: 1.0243
Training complete. Now you can generate text.
Enter the initial text (n characters, or 'exit' to quit): apologetic
Enter the number of characters to generate: 25
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: apologeticatggeegqhsugqigsyoppplnlt
Enter the initial text (n characters, or 'exit' to quit): apolo   
Enter the number of characters to generate: 5
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: apololstuv
Enter the initial text (n characters, or 'exit' to quit): apolo
Enter the number of characters to generate: 5         
Enter the temperature value (1.0 is default, >1 is more random): 2
Generated text: apoloiykgn


sequence_length = 100 # Length of each input sequence
stride = 10            # Stride for creating sequences
embedding_dim = 16      # Dimension of character embeddings
hidden_size = 64        # Number of features in the hidden state of the RNN
learning_rate = 0.01    # Learning rate for the optimizer
num_epochs = 10         # Number of epochs to train
batch_size = 64        # Batch size for training

Test Loss: 0.0005

Epoch 1/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.78it/s]
Epoch 1, Loss: 3.0686
Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 11.56it/s] 
Epoch 2, Loss: 1.6910
Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.65it/s] 
Epoch 3, Loss: 0.3847
Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 12.24it/s] 
Epoch 4, Loss: 0.0690
Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 12.27it/s] 
Epoch 5, Loss: 0.0366
Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.97it/s] 
Epoch 6, Loss: 0.0390
Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.03it/s] 
Epoch 7, Loss: 0.0332
Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10.76it/s] 
Epoch 8, Loss: 0.0291
Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 11.58it/s] 
Epoch 9, Loss: 0.0239
Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 11.86it/s] 
Epoch 10, Loss: 0.0218
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 47.99it/s] 
Test Loss: 0.0005
Training complete. Now you can generate text.
Enter the initial text (n characters, or 'exit' to quit): vsdv
Enter the number of characters to generate: 3
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: vsdvvwx
Enter the initial text (n characters, or 'exit' to quit): wswcdw
Enter the number of characters to generate: 12
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: wswcdwabcdefghijk

sequence_length = 200 # Length of each input sequence
stride = 5            # Stride for creating sequences
embedding_dim = 30      # Dimension of character embeddings
hidden_size = 128        # Number of features in the hidden state of the RNN
learning_rate = 0.001    # Learning rate for the optimizer
num_epochs = 5         # Number of epochs to train
batch_size = 64        # Batch size for training

Test Loss: 0.0002
 
Epoch 1/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  4.36it/s]
Epoch 1, Loss: 2.3435
Epoch 2/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  4.95it/s] 
Epoch 2, Loss: 0.1346
Epoch 3/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  5.72it/s] 
Epoch 3, Loss: 0.0359
Epoch 4/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  4.71it/s] 
Epoch 4, Loss: 0.0371
Epoch 5/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  5.63it/s] 
Epoch 5, Loss: 0.0286
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 27.48it/s] 
Test Loss: 0.0002
Training complete. Now you can generate text.
Enter the initial text (n characters, or 'exit' to quit): pea   
Enter the number of characters to generate: 6
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: peabcdefg


For the given text, I also tested a few hyperparamter values as follows:

sequence_length = 100 # Length of each input sequence
stride = 5            # Stride for creating sequences
embedding_dim = 20      # Dimension of character embeddings
hidden_size = 64        # Number of features in the hidden state of the RNN
learning_rate = 0.01    # Learning rate for the optimizer
num_epochs = 2         # Number of epochs to train
batch_size = 64        # Batch size for training

Epoch 1/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8774/8774 [10:00<00:00, 14.61it/s]
Epoch 1, Loss: 1.7949
Epoch 2/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8774/8774 [18:21<00:00,  7.96it/s] 
Epoch 2, Loss: 1.7692
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 975/975 [00:18<00:00, 51.45it/s]
Test Loss: 1.8098
Training complete. Now you can generate text.
Enter the initial text (n characters, or 'exit' to quit): pea
Enter the number of characters to generate: 7
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: peady it t
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 11
Enter the temperature value (1.0 is default, >1 is more random): 4
Generated text: peacet.yiaf glm.
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 11   
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: peaces that her 
Enter the initial text (n characters, or 'exit' to quit): love 
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: love alas his withfather
Enter the initial text (n characters, or 'exit' to quit): war
Enter the number of characters to generate: 30
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: ward kearty.ya, they backsation t
Enter the initial text (n characters, or 'exit' to quit): war
Enter the number of characters to generate: 10
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: wark the fu! 
Enter the initial text (n characters, or 'exit' to quit): war
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: warkered and that one t
Enter the initial text (n characters, or 'exit' to quit): war
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 2
Generated text: wart oryimp imselwawwy,
Enter the initial text (n characters, or 'exit' to quit): war
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: ward buk mipibet led th
Enter the initial text (n characters, or 'exit' to quit): love 
Enter the number of characters to generate: 10
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: love weiel f c
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 10  
Enter the temperature value (1.0 is default, >1 is more random): 2
Generated text: lovebssioabfns
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 10  
Enter the temperature value (1.0 is default, >1 is more random): 1 
Generated text: love were why.

Second instance 

sequence_length = 50 # Length of each input sequence
stride = 20            # Stride for creating sequences
embedding_dim = 16      # Dimension of character embeddings
hidden_size = 64        # Number of features in the hidden state of the RNN
learning_rate = 0.001    # Learning rate for the optimizer
num_epochs = 2         # Number of epochs to train
batch_size = 128        # Batch size for training

Epoch 1/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1096/1096 [01:14<00:00, 14.76it/s]
Epoch 1, Loss: 2.2603
Epoch 2/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1096/1096 [00:55<00:00, 19.85it/s] 
Epoch 2, Loss: 1.9133
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 122/122 [00:01<00:00, 66.13it/s]
Test Loss: 1.8929
Training complete. Now you can generate text.
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 10
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: loves thatiens
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 10
Enter the temperature value (1.0 is default, >1 is more random): 2
Generated text: lovexnn;tthiar
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 10
Enter the temperature value (1.0 is default, >1 is more random): 3
Generated text: love whojluwns
Enter the initial text (n characters, or 'exit' to quit): love 
Enter the number of characters to generate: 10
Enter the temperature value (1.0 is default, >1 is more random): 5
Generated text: love wutndjux(s
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: loves, but llaid die plo
Enter the initial text (n characters, or 'exit' to quit): love 
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 5
Generated text: love yisknoo, lexpy[laico
Enter the initial text (n characters, or 'exit' to quit): love 
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 100000
Generated text: love tgfbrxse;b,a?tc?z[q;
Enter the initial text (n characters, or 'exit' to quit): love  
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 0.5
Generated text: love to was the our, and 
Enter the initial text (n characters, or 'exit' to quit): love 
Enter the number of characters to generate: 0
Enter the temperature value (1.0 is default, >1 is more random): 0
Generated text: love 
Enter the initial text (n characters, or 'exit' to quit): love
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 0
Generated text: love and the count the c
Enter the initial text (n characters, or 'exit' to quit): peace 
Enter the number of characters to generate: 0
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: peace 
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 50
Enter the temperature value (1.0 is default, >1 is more random): 0
Generated text: peace and the count the count the count the count the c
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 40
Enter the temperature value (1.0 is default, >1 is more random): 0.3
Generated text: peace and the face and had and and and and be
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 0.6
Generated text: peaced and he for and a s
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: peace. ubing be had evene
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 0.4
Generated text: peace and the was the cou
Enter the initial text (n characters, or 'exit' to quit): peace
Enter the number of characters to generate: 20
Enter the temperature value (1.0 is default, >1 is more random): 0.3
Generated text: peace to the count the be


2. Analysis on final train and test loss for both datasets

Alphabet dataset

I trained on "abcdefghijklmnopqrstuvwxyz" × 100 with fairly modest settings (seq_len=200, hidden_size=128, embedding_dim=30, lr=0.001, 5 epochs, batch size = 64). By the end, the training loss goes down to 0.02 and test loss reached 0.002. In other words, the model learned that repeating pattern with a good prediction and on an unseen data results even better. Meaning that the model is neither too overfitted nor underfitted. 

War & Peace dataset

Switching to the real text, in the best run of hyperparameters (seq_len=100, stride=5, embed=20, hidden=64, lr=0.01, batch size=64, epochs=2) ended with a train loss of 1.7692 and a test loss of 1.8098. Losses in the 1.0–2.0 range translate to roughly fair accuracy of identifying words, which matches the observation. The model starts to get “the”, “and”, “peace” right about half the time, but still produces lots of pseudo‑words / fake words / wrong words. The relatively small difference between train and test loss tells that I’m not massively overfitting, but the absolute loss is still high that it isent perfect.

3. Explain impact of changing temperature

A temperature between 0.2 to 0.5 gives me proper words that are legible and seem real and make somewhat sense. As I increase it to 1, it on random gives good words and meaningless words and any temperature over 1 gives absolute random words whoch dont make sense. A temperature of 0 gives me continuously repeated phrases. Hence I feel that overall temperature gives a randomness of words and letters. 

4. Reflection

Overall I believe that the project was quite time consuming given the learning of the entire text with the differing hyperparameters which was quite decent figuring out, however took quite a while to compute on my system since it took longer with differing hyperparameters. Apart from the code implementation was not too bad. I feel like overall it was a nice implementation since Im learning a similar class that deals with learning models and analysing them but this class helps me better to implement them which gives me a hands on experience which I felt was the best. 

5. IMPORTANT: Include screenshot of output for generate_text() function: 

All the required data has been given in section 1.  
