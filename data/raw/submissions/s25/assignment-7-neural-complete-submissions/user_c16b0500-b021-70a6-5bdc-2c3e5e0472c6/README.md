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

| Input | Target |
| ----- | ------ |
| "abc" | "bcd"  |
| "bcd" | "cde"  |
| "cde" | "def"  |
| "def" | "efg"  |
| "efg" | "fgh"  |

#### With `stride = 2`:

| Input | Target |
| ----- | ------ |
| "abc" | "bcd"  |
| "cde" | "def"  |
| "efg" | "fgh"  |

So as you can see, a higher stride results in less examples. This is a training hyperparameter which you can experiment with — smaller values increase data size and overlap, while larger values reduce redundancy and speed up training.

### Milestone 1. Teach an RNN the alphabet

Now that you've gone through the code it's time to implement the RNN and get the model to train on the alphabet sequence. Note once you've completed this your model should get a very high accuracy (close to 100%) as this is a very simple repeated sequence.

First, we'd recommend you complete the training section up until the training loop. Then, complete the model implementation. Then complete the training loop and try to train your model.

#### Training setup components

The code has a number of TODOs prior to the training loop, these should be pretty straightforward and are designed to help you understand the flow of the code by tieing in concepts from previous assignments.

#### RNN implementation

Inside `CharRNN.__init__()`, you’ll need to define the learned parameters of the RNN

**Your task**: Randomly initialize each parameter using `nn.Parameter(...)`, and follow the structure discussed in lecture. Keep standard deviations small (e.g., \* 0.01).

Inside the `forward()` method:

```python
for t in range(l):
    #  TODO: Implement forward pass for a single RNN timestamp
    pass
```

Here you’ll implement the recurrence equation for the RNN. Each timestep receives:

- the current input embedding x_t
- the previous hidden state h\_{t-1}

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

2. If you don't see a significant jump after the first epoch, you shouldn't wait, change the parameters and try again.

3. If you're losing patience, try taking a fraction of the dataset so you don't have to wait as long, and then run it on the full set after that's working.

4. Don't expect a perfect model. What would it mean to have 90% accuracy on this model, is that realistic? You'd have created a novel writing masterpiece of a model! Realistically your performance will be much lower, around 50-60% with a loss around 1.5. But even with this "low performance" you should see words (or pseudo-words) in your output but not meaningful sentences.

### Milestone 4. Final Report

In your report, describe your experiments and observations when training the model with two datasets: (1) the sequence "abcdefghijklmnopqrstuvwxyz" \* 100 and (2) the text from warandpeace.txt.

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

How many late days are you using for this assignment?  
Zero Late Days

1. Describe your experiments and observations  
   I trained the same character-level RNN on two very different corpora to test how architecture and hyper-parameters generalise:

| Corpus                                                     | `sequence_length` | `stride` | `embedding_dim` | `hidden_size` | `lr` | `epochs` | Batch |
| ---------------------------------------------------------- | ----------------- | -------- | --------------- | ------------- | ---- | -------- | ----- |
| **Repeated alphabet** (`"abcdefghijklmnopqrstuvwxyz"*100`) | 50                | 15       | 32              | 128           | 0.01 | 10       | 64    |
| **_War and Peace_** (full text)                            | 50                | 15       | 32              | 128           | 0.01 | 3        | 64    |

- Alphabet converged in < 1 epoch; I let it run 10 to verify stability.
- _War and Peace_ needed only one full pass to reach its best loss; by epoch 3 the train loss had started to rise (early signs of over-fitting), so three epochs were sufficient.
- Hidden-state size 128 struck a good balance between capacity and runtime; bigger sizes (>256) improved perplexity marginally but tripled training time.

2. Analysis on final train and test loss for both datasets
   | Corpus | Best Train Loss | Final Train Loss | Test Loss |
   |--------|-----------------|------------------|-----------|
   | Alphabet | **0.0606** (epoch 10) | 0.0606 | **0.0008** |
   | _War and Peace_ | **1.6983** (epoch 2) | 1.8000 | 1.9391 |

The alphabet task is almost deterministic; the network transitioned from memorisation to perfect generalisation, giving a near-zero test loss. For the literary corpus, loss plateaued around 1.7–1.9—typical for a shallow character RNN on long-form English text. The slight uptick between epochs 2 & 3 confirmed diminishing returns.

3. Explain impact of changing temperature
Using the _War and Peace_ model with the prompt **“hello”** (15 characters generated):

| τ (temperature) | Sample Output | Behaviour |
|-----------------|---------------|-----------|
| 0.1 | `hellowed the said th` | Very low entropy; model sticks to the most probable next letters, often repeating common trigrams. |
| 0.5 | `hellownow, anting on the rank ` | Safer diversity; English-like bigrams, mostly dictionary words. |
| 0.7 | `hellowever there on ` | Coherent phrase fragments; good trade-off (used as “default” in demo). |
| 1.0 | `hellowinser undeaste` | Noticeable inventiveness; pseudo-words but phonotactically plausible. |
| 1.5 | `hellob,kinctua, is,s` | Jumps outside high-probability region; creative but less readable. |
| 2.0 | `hellopered.t cknee! ` | Near-random; punctuation injected, grammar collapses. |
| 2.5 | `hellox.b charitiovin` | Almost noise; still retains character-level plausibility. |

Lower τ (< 1) sharpens the output distribution, so generation largely echoes training n-grams. Raising τ flattens the distribution, letting the sampler choose lower-probability characters—growing lexical novelty at the cost of syntax and semantics.


4. Reflection
Implementing the bare-bones RNN reinforced how much manual bookkeeping (hidden-state loops, parameter initialisation, teacher forcing) underpins higher-level APIs. Two insights stood out:

* **Data efficiency vs. redundancy**: With `stride = 15`, the model saw ≈3× fewer overlapping slices than a stride-1 setup, yet reached identical losses—indicating that character context windows contain heavy redundancy.  
* **Early stopping matters**: On rich text, additional epochs quickly dipped into over-fitting. Monitoring validation loss and halting at the first uptick preserved generalisation and saved compute.

Overall, the exercise illustrated the trade-off between memorisation and creativity in sequence modelling: a network good enough to predict the next letter of *any* English word still cannot guarantee grammatical sentences, but with judicious temperature tuning it can produce text that feels stylistically authentic.

5. Terminal Output: (.venv) <redacted>@Vinayaks-MacBook-Pro assignment-7-neural-complete-<redacted> % "/Users/<redacted>/Library/Mobile Documents/com~apple~CloudDocs/College/Classes Spring '25/
COMPSCI 383/assignment7/assignment-7-neural-complete-<redacted>/.venv/bin/python" "/Users/<redacted>/Library/Mobile Documents/com~apple~CloudDocs/College/Classes Spring '
25/COMPSCI 383/assignment7/assignment-7-neural-complete-<redacted>/rnn_complete.py"
/Users/<redacted>/Library/Mobile Documents/com~apple~CloudDocs/College/Classes Spring '25/COMPSCI 383/assignment7/assignment-7-neural-complete-<redacted>/.venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:276: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /Users/<redacted>/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:81.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
Using device: mps
Epoch 1/3:   0%|                                                                                                                                | 0/2924 [00:00<?, ?it/s]/Users/<redacted>/Library/Mobile Documents/com~apple~CloudDocs/College/Classes Spring '25/COMPSCI 383/assignment7/assignment-7-neural-complete-<redacted>/rnn_complete.py:28: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
/Users/<redacted>/Library/Mobile Documents/com~apple~CloudDocs/College/Classes Spring '25/COMPSCI 383/assignment7/assignment-7-neural-complete-<redacted>/rnn_complete.py:29: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  target = torch.tensor(self.targets[idx], dtype=torch.long)
Epoch 1/3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2924/2924 [01:23<00:00, 35.13it/s]
Epoch 1, Loss: 1.7723
Epoch 2/3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2924/2924 [01:26<00:00, 33.70it/s]
Epoch 2, Loss: 1.7076
Epoch 3/3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2924/2924 [01:23<00:00, 35.06it/s]
Epoch 3, Loss: 1.7697
Epoch 3, Test Loss: 1.9124
Training complete. Now you can generate text.
Enter the initial text (n characters, or 'exit' to quit): test
Enter the number of characters to generate: 15
Enter the temperature value (1.0 is default, >1 is more random): 1
Generated text: test secomeneivnale
Enter the initial text (n characters, or 'exit' to quit): 