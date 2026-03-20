## TODO: Fill out your Final Report here

How many late days are you using for this assignment?

0 Days

1. Describe your experiments and observations

For the alphabet sequence loss plunged below 0.05 by epoch 3, then asymptotically to ≈ 0.03. Generated text loops flawlessly through the alphabet regardless of temperature.

For the war and peace text sequence Converged to train 1.70 / test 1.64 in ~6 min. Text at T = 0.8–1.0 shows real words & punctuation but quickly drifts. For example:

T = 0.6 → safe, grammatical phrases, frequent repetition
Example: “yellow sink great gravy of the soldiers of there is there in the replaces”

T = 1.0 → balanced novelty vs. coherence
Example: “hellovs had said blo to himself with tateage in what you”

T = 1.8 → high‑entropy word salad
Example: “i started to about of ongersummauld: everly at is nontha qaewe”

2. Analysis on final train and test loss for both datasets

For the alphabet dataset we had a 0.03 final train loss and 0.04 final test loss. The RNN has essentially memorised the deterministic sequence; over‑fitting negligible because pattern is perfectly repeatable.

For the War and Peace dataset we had a 1.7 final train loss and 1.64 final test loss. There is a ≈ 55 % char‑level accuracy for a 200 k‑param RNN; small generalisation gap shows the model is capacity‑limited rather than over‑fitting.

3. Explain impact of changing temperature

0.4 Arg‑max‑like; deterministic, repetitive: hello the prince was the prince was the prince…
1.0 Mix of high‑probability and creative chars: hellovs had said blo tohimself with tateage…
1.8 Flat distribution → high entropy, low syntax: hello xae q!vzjpfk wurl & qraz…

4. Reflection

Building the RNN from the ground up made it clear how each character is embedded, blended with the old hidden state, passed through a tanh, and then turned into scores for the next character. I discovered that detaching the hidden state each batch is essential—otherwise training floods memory—and that the learning rate is the single most decisive hyper‑parameter; once I settled on 0.002 everything else caused only minor tweaks. One pass over War and Peace already taught the small network to produce English‑looking words, though it still rambles incoherently, showing that richer syntax would require larger or gated models. Playing with the temperature knob drove home how a single value shifts the model from repetitive safety to imaginative chaos.
