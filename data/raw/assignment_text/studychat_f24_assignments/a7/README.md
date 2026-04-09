# Assignment 7: Neural Complete

## Attribution

When using or sharing this material, please include the following attribution:

> Originally developed by Aditya Singh and Hunter McNichols as part of the course COMPSCI 383 Artificial Intelligence at the University of Massachusetts Amherst.  
> Shared under the CC BY 4.0 License.

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

In this assignment, you'll be building a Recurrent Neural Network (RNN) to create an autocomplete system. This will be a bit more advanced than the ones you've created earlier in the course.

The assignment is meant to **teach you more about RNNs**, aligning with what you’ve gone over in class, rather than test you on it. The Jupyter notebook will guide you step-by-step through building your own RNN network. Please make sure to thoroughly read through each part to develop a solid understanding of how RNNs are implemented from scratch.
