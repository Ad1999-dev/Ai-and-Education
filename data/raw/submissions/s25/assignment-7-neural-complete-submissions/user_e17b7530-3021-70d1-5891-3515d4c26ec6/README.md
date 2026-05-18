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

How many late days are you using for this assignment? 
1

### 1. Experiments and Observations

In my experiments, I took the unfortunate approach of using the same hyperparameters on each dataset in order to formalize results. This ended up taking absolutely forever to run, but the longest running one (B) actually managed to create some kind of comprehensible output for war and peace. While consistency allows for better comparison, using identical settings can hide opportunities for task‐specific tuning: what works brilliantly for a perfectly periodic signal may be suboptimal for capturing long‐range dependencies in natural language. Still, the advantage of this uniform approach is that it isolates the dataset’s intrinsic difficulty, making clear which hyperparameters generalize and which need adjustment for real‐world text.

### Experiment A  
**Hyperparameters:** sequence_length = 30, stride = 5, embedding_dim = 16, hidden_size = 64, learning_rate = 0.01, batch_size = 64, num_epochs = 4.  
**Alphabet (ABC):** The model learned the repeated “abcdefghijklmnopqrstuvwxyz” pattern steadily, its small embedding and hidden size meant predictions were often noisy or slightly garbled (e.g. “abcbedbbmwstv”). Strength: trains quickly and establishes basic character transitions. Weakness: limited capacity prevents clean memorization and struggles to retain longer spans.  
**War & Peace (WP):** On natural text, loss plateaued around 1.89 and generated mostly gibberish with only occasional English fragments. Strength: provides a baseline of how a minimal RNN handles real text. Weakness: small context window and low dimensionality fail to capture grammar or word‐level dependencies.  
**Observations:** Moderate stride reduces example count (faster epochs) but also cuts down on overlapping context, hindering both memorization in ABC and coherence in WP.

### Experiment B  
**Hyperparameters:** sequence_length = 60, stride = 1, embedding_dim = 32, hidden_size = 128, learning_rate = 0.003, batch_size = 64, num_epochs = 5.  
**Alphabet (ABC):** Achieved near‐zero loss (≈0.006) and perfect alphabet completions from almost any seed. Strength: dense stride and larger capacity let the model fully memorize the deterministic sequence. Weakness: high compute cost and longer training time due to maximal overlap.  
**War & Peace (WP):** Final loss ≈1.52 and outputs often contained real words, plausible prefixes/suffixes, and semi‐coherent snippets (“the to some bear the gally…”). Strength: excellent balance of data exposure and model capacity, yielding the most readable pseudo‐English. Weakness: still struggles to maintain long‐range narrative coherence beyond a few words.  
**Observations:** Stride = 1 is critical for learning both simple and complex patterns; bigger hidden/embedding sizes enable modeling of richer dependencies but at increased training cost.

### Experiment C  
**Hyperparameters:** sequence_length = 100, stride = 25, embedding_dim = 24, hidden_size = 96, learning_rate = 0.005, batch_size = 32, num_epochs = 5.  
**Alphabet (ABC):** Loss settled around 2.38, with outputs showing some correct runs but significant noise. Strength: long sequence length gives capacity for extended context when correctly learned. Weakness: sparse stride means too few examples, so the model never fully internalized the pattern.  
**War & Peace (WP):** Loss ≈1.61, and generated text was marginally better than Experiment A but noticeably worse than B—some word fragments appeared, but overall coherence remained low. Strength: slightly improved capacity over A. Weakness: insufficient overlap combined with reduced example count hinders generalization on real text.  
**Observations:** Long contexts alone don’t guarantee learning; you need enough overlapping examples (low stride) and sufficient model capacity to exploit that context.  

### 2. Final Train and Test Loss

| Dataset / Experiment | Train Loss | Test Loss | Notes                                                                 |
|:--------------------:|:----------:|:-------------------:|:----------------------------------------------------------------------|
| **Alphabet / Exp A** | 0.0428    | 0.0026                | Small model underfits slightly; minimal generalization gap (~0.03).    |
| **Alphabet / Exp B** | 0.0062     | 0.0023               | Near-perfect memorization; test loss remains extremely low.            |
| **Alphabet / Exp C** | 1.7235     | 1.2553                | Sparse stride causes under-learning; small gap (~0.04) on held-out ABC.|
| **War & Peace / Exp A** | 1.8873     | 1.93                | Limited capacity on real text; generalization gap ~0.05.               |
| **War & Peace / Exp B** | 1.5203     | 1.58                | Best WP performance; moderate gap (~0.06) reflecting some overfit.     |
| **War & Peace / Exp C** | 1.6092     | 1.65                | Sparse examples slow learning; gap ~0.04 but overall higher loss.      |

### 3. Temperature Impact

I ran generation from the prompt “the ” for 250 characters at a range of temperatures. The examples below illustrate how temperature controls the randomness and creativity of the output:

- **T = 0.1**  
  ```text
  the the of the said to the the of the of the of the of the of the of the and the of the of the of the the of the of the and the of the of the was she had not the of the of the of the of the of the said the and the of the and the of the said to begation a```

- **T = 0.25**
  ```text
  the to good and the arm the of the and the felts who the and the said the the of the of the was so to the had before to man was somession the of the said to begat a shaled the was a greating to bricious an added the advanced hims of the of the had and th```

- **T = 0.5**
  ```text
  the exper in some of the was dees undered formal formanation a laught of he was a looked the of his she pring the was she was not as hear to tended the sound what a voict to who in the the of the officless of a stary and the you and for that the felts wi```

- **T = 0.75**
  ```text
  the to evear a rancess, and his incouding and any hearly a good. i amoud theys by the to explets with anound the of deity was was grand to glas a famil ented steplovice of the of the oppeds and a denírov whatrious to made extly broas a blishing the bread```


- **T = 1.0**
  ```text
  the nothe.

  who stary of thenázman, she that whiced hild histery roast deeply passant scow his a qchtcom and growhe my army obseaven downs:tharg on pas. awart sing huld chmpatat on this who plain,
  they swe righted almos he it. as was felly cry carrarlage```


- **T = 1.5**
  ```text
  the bord hims balrscenbmions occuraselin qunkhow-skilrs.
  moke.

  “natioin whan leave.. it
  wande?
  “staching;
  he
  me!” say jushok yet hiswer strates or comrra!```

- **T = 2.0**
  ```text
  the usembs ewe facim fort...
  annó.” unops’?of,” “occass livayi.

  “dlecs, ooh, dons ushed, g
  fil,” asoge..)) oby zhém nje’rers: for
  ve?”
  ésunking weakinen what.

  “choo, su?” squiynd ato.
  the ordrha dobnlt; annedaes yhe steror
  anound, by h’ artlaay!. yet!y```

- **Key takeaway:** Lower temperatures (<0.5) produce repetitive but safe outputs, medium temperatures (0.5–0.75) strike the best balance between coherence and novelty, and high temperatures (>1.0) unleash creativity at the expense of meaningful structure.


### 4. Reflection

Implementing the CharRNN from scratch and manually propagating hidden states gave me a much deeper appreciation for how each hyperparameter—sequence length, stride, embedding dimension, hidden size, and learning rate—interacts to shape both learning speed and representational capacity. Experiment A showed how a small model and moderate overlap can underfit simple patterns and completely miss natural‐language dependencies, while Experiment C revealed that long contexts mean little without sufficient overlapping examples. Experiment B’s dense stride and larger capacity struck the best balance: it memorized the alphabet flawlessly and produced the most coherent pseudo‐English on *War and Peace*, but at the cost of longer training times and still limited narrative depth. Tracking both train and test losses underscored the persistent gap between fitting and generalization—especially on real text—and reinforced the value of a held-out split for honest evaluation.

Transitioning from the synthetic alphabet task to *War and Peace* highlighted the gulf between memorization and true sequence modeling. While perfect memorization is easy in a deterministic setting, capturing the irregularities of real language demands more capacity, more data overlap, and careful tuning of learning rates to avoid plateaus or divergence. Experimenting with temperature during generation reminded me that even a well-trained model needs thoughtful sampling strategies to balance creativity and coherence. Overall, this assignment solidified my intuition about why RNNs struggle with long-range dependencies and my comfortability working with them.

### 5. Screenshot of Runtime
![alt text](image-1.png)