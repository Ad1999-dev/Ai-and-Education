Milestone 4. Final Report

Late days used: 2

For this assignment, I trained a character-level RNN on two datasets: (1) a synthetic sequence consisting of `"abcdefghijklmnopqrstuvwxyz"` repeated 100 times, and (2) the real-world text of *War and Peace*. Early attempts using the starter hyperparameters (`embedding_dim=2`, `hidden_size=1`, `sequence_length=1000`, `learning_rate=200`) failed to train effectively. The model consistently produced repetitive characters, and loss remained high. I realized that the sequence length was too large relative to the dataset (2600 characters), which resulted in very few usable training examples. Additionally, the model lacked the capacity to memorize or generalize patterns.

After tuning the parameters — increasing `embedding_dim` to 16, `hidden_size` to 128, and lowering `learning_rate` to 0.01, while increating the alphabet string to 1000x instead of 100, the model quickly learned the alphabet dataset. Final train and test loss on this synthetic set dropped to nearly 0, being 0.01 at the end and the model was able to complete the full alphabet sequence starting from just the character `'a'`.

On *War and Peace*, the model converged more slowly and achieved a final loss between 1.5 and 2.0, which is reasonable given the complexity of natural language. Generated text with temperature = 1.0 showed plausible pseudo-words and partial sentences. Lower temperatures (e.g., 0.5) produced repetitive but coherent character groupings, while higher temperatures (e.g., 1.5) led to more creative but erratic outputs. This assignment highlighted the sensitivity of RNNs to hyperparameters, the importance of dataset sizing relative to sequence length, and how temperature controls the creativity vs. coherence trade-off in text generation.
