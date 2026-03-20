## Final Report

### Late-Day Usage

0 days

### 1. Experiments & Observations

We trained a character-level RNN on two datasets:

- **Dataset A** – the alphabet sequence “abcdefghijklmnopqrstuvwxyz” repeated 100×.
- **Dataset B** – _War and Peace_ by Leo Tolstoy (~3.3 MB).

For each dataset we fixed a single-layer RNN with embeddings and tuned only learning rate, hidden size, and batch size.

| Hyper-Parameter | Dataset A | Dataset B |
| --------------- | --------- | --------- |
| `embedding_dim` | 32        | 64        |
| `hidden_size`   | 64        | 256       |
| `learning_rate` | 0.001     | 0.002     |
| `batch_size`    | 32        | 128       |
| Epochs          | 30        | 1         |

Observations:

- Alphabet convergence was rapid: by epoch 7 the loss flattened near 0.01. Accuracy exceeded 99%, matching the deterministic pattern.
- In contrast, English text required far larger hidden states to capture long-range dependencies. Doubling `hidden_size` from 128 → 256 helped reduce loss.
- Increasing stride from 8 → 16 sped training 2× with negligible impact on validation loss, which means high overlap is less important for natural language.

### 2. Final Train/Test Loss

| Metric     | Dataset A  | Dataset B  |
| ---------- | ---------- | ---------- |
| Train Loss | **0.0004** | **1.6405** |
| Test Loss  | **0.0003** | **1.5142** |

Alphabet accuracy (exact-next-char) ≈ 100%. For _War and Peace_ the per-character accuracy ≈ 50%.

### 3. Temperature Effects on Generation

Using the _War and Peace_ model and a 45-character prefix “Well, Prince, so Genoa and Lucca are now just”, we generated 100 new characters under three temperatures:

| **T** | **Sample Output**                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------------- |
| 0.5   | “… as the countess and said the house and said and leave in his second speak of the regiment to the ho“ |
| 1.0   | “… bolg last the faces, he has visiting of the room for hard, so mean pulled himself! said yes, and af“ |
| 1.5   | “… agatinawaiced annwhichm... de is.if himself troops of biteng, all ruin after was ful any slee, emol” |

Lower T sharpens the distribution, producing repetitive grammar and frequent clauses seen in training. High T injects noise, yielding creative but often nonsensical words ("agatinawaiced", “annwhichm”).

### 4. Reflection & Insights

The two datasets demonstrated clear differences in learning complexity. The alphabet sequence required minimal capacity and converged perfectly within few epochs, while War and Peace demanded significantly larger models that still showed room for improvement. The alphabet model produced nearly perfect predictions with consistent patterns, whereas the literary text generated more varied outputs with different words and phrases but does not produce meaningful sentences due to their unstructured and nonsensical combinations. Most interesting was observing how temperature directly controls the trade-off between the text generation being deterministic and creative. Lower settings (T=0.5) produced repetitive but comprehensible text, while higher values (T=1.5) generated novel but often garbled words. Overall, the experiments showed how RNNs encode sequential context through hidden state propagation and how capacity requirements scale dramatically with task complexity.

### 5. Terminal Output Screenshots

#### Dataset A - Alphabet Sequence

![Training and test loss for dataset A with sample generated text](datasetA.png)

#### Dataset B - War and Peace

![Training and test loss for dataset B with sample generated text](datasetB.png)
