## TODO: Fill out your Final Report here

How many late days are you using for this assignment? 0

1. Describe your experiments and observations

I first asked 383GPT for some good benchmarks for the hyperparameters. After some testing, I found that the stride in particular made the epochs take much longer to complete. For the "abcdefghijklmnopqrstuvwxyz" dataset, around 7 epochs consistently resulted in the smallest loss overall. However, the stride had to be 1, for such a small sequence of characters (26). This took around 3 minutes to fully complete, but resulted in a very high accuracy model. For example, it correctly filled 2 sequences of the alphabet with only the starting letter of 'a'. The full parameter list for the alphabet was:

sequence_length = 500
stride = 1           
embedding_dim = 50     
hidden_size = 64       
learning_rate = 0.001  
num_epochs = 7       
batch_size = 64       
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

I had a similar approach to making parameters for War and Peace, but had to heavily increase stride and learning rate and decrease the num_epochs, just to get 1 epoch to finish in 10 minutes.

The full parameter list for War and Peace was:
sequence_length = 300 # Length of each input sequence
stride = 50            # Stride for creating sequences
embedding_dim = 100      # Dimension of character embeddings
hidden_size = 256       # Number of features in the hidden state of the RNN
learning_rate = 0.01    # Learning rate for the optimizer
num_epochs = 1        # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

2. Analysis on final train and test loss for both datasets

The final train and test loss for the alphabet was:
Test Loss: 0.0078
Epoch 7, Loss: 0.0075 from Epoch 1, Loss: 3.1088 

The final train and test loss for War and Peace was:
Test Loss: 481.7674
Epoch 1, Loss: 196.9084

The alphabet training was understandably much more precise than War and Peace, as there was much less variation in the source material.

3. Explain impact of changing temperature

The impact of changing temperature in the context of the alphabet was that random letters were selected to predict the sequence. For example, with initial text "abc" and 23 as the number of characters to generate, temperatures of 1 and 0.5 resulted in the correct sequence "abcdefghijklmnopqrstuvwxyz". On the other hand, a temperature of 1.5 resulted in the string "abcdepqrbcdefgefghijklmnop" and 2 resulted in the string "abcdefghijklmnopqrsuwwopqr". On the other hand, for War and Peace, nothing really resulted in a lucid response. For example, with inputs of "Pierre B" and 7 (one of the common names), a temperature of 1 resulted in the string "pierre b)n muvv", while a temperature of 0.5 resulted in an output of "pierre b)n muvv" and a temperature of 2 resulted in an output of "pierre blruvvvv".

4. Reflection

One of the biggest things was the impact of stride, which was not something I had previously thought about. I suppose it does make sense however, as it takes more time to traverse through the more overlapping sequences. RNNs are pretty well suited for language modelling, as they learn inherent patterns in language and writing.  