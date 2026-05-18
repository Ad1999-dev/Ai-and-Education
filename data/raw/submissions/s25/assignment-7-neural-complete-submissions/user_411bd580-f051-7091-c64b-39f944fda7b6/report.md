# Assignment 7 - Project Report
**<redacted> <redacted>**  
May 2025

## Comparison of RNN Performance on Two Datasets

| **Hyperparameter**     | **Alphabet Sequence** | **_War and Peace_** |
|------------------------|------------------------|----------------------|
| Sequence Length        | 80                     | 150                  |
| Stride                 | 5                      | 10                   |
| Embedding Dimension    | 16                     | 32                   |
| Hidden Size            | 128                    | 256                  |
| Learning Rate          | 0.08                   | 0.0015               |
| Epochs                 | 10                     | 2                    |
| Batch Size             | 16                     | 64                   |
| Train Loss             | 0.0000                 | 1.3811               |
| Test Loss              | ~0.0000                | 1.4652               |
| Training Time          | < 1 minute             | ~5 minutes           |

**Table:** Hyperparameters and final performance metrics for both datasets.

---

## Description of Experiments and Observations

I trained a character-level recurrent neural network (RNN) on two datasets: a synthetic sequence created by repeating the English alphabet (`abcdefghijklmnopqrstuvwxyz`) 100 times, and the full text of *War and Peace* by Leo Tolstoy. These datasets differed dramatically in complexity, prompting different choices in hyperparameters and architecture. For the alphabet sequence, I experimented with lightweight models and aggressive learning rates. Using a hidden size of 128 and a sequence length of 80, the model converged quickly with a learning rate of 0.08. Within 10 epochs, it achieved perfect recall of the alphabet pattern. For *War and Peace*, I switched to a deeper model with a hidden size of 256, a sequence length of 150, and a more conservative learning rate of 0.0015 to avoid instability. This training ran for 2 epochs and took approximately five minutes to complete. The alphabet model consistently output exact substrings like “fghij” when prompted with “cde,” even at various temperatures. The *War and Peace* model, while far less precise, generated realistic-sounding English snippets and preserved character-level fluency. Sample generations included:  
> “around his own the childly living where put it”  
> “how this horse firad. the sk the staff action, sa”  

These results matched expectations: alphabet generation was fully deterministic, while literary text generation demanded probabilistic modeling of complex sequences.

---

## Analysis of Final Train and Test Loss

For the alphabet dataset, both training and test loss dropped to nearly zero (0.0000), which is expected given the simplicity and repetition in the data. This showed that the model learned the underlying pattern with perfect accuracy. In contrast, for *War and Peace*, the final training loss settled at 1.3811, with a slightly higher test loss of 1.4652. These values are typical for character-level models on natural language corpora, where the long-term structure is difficult to capture due to the limited memory of standard RNNs. The gap between train and test loss remained small, suggesting good generalization.

---

## Impact of Changing the Temperature Parameter

Temperature scaling had a significant impact on the quality and creativity of generated text. With the alphabet model, temperature changes had little visible effect due to the deterministic nature of the pattern. For the *War and Peace* model, temperature played a central role in shaping output. At a temperature of 1.0, generated text often included plausible English constructions, though lacking global coherence. Lowering the temperature to around 0.7 resulted in safer, more repetitive outputs (e.g., repeating the same clause or word fragments), while increasing it to values like 1.2 produced highly varied but grammatically unstable sequences. For example, a prompt with temperature = 1.2 produced:
> “the furst clarm in coud, the was somplected brudly of the hinst”

These variations helped me balance between structured and novel outputs, depending on the desired style.

---

## Reflection

This assignment deepened my understanding of sequence modeling and the sensitivity of RNNs to both hyperparameters and input structure. For synthetic data like repeated alphabets, even a small model with high learning rates is sufficient. However, real-world language demands a more nuanced architecture and careful tuning to extract meaningful patterns. I learned the importance of sequence length, hidden state size, and learning rate — particularly how a poor choice can lead to overfitting or unstable training.

## Screenshots of Tests
![Long Test](longTest.png)
![Short Test](shortTest.png)
