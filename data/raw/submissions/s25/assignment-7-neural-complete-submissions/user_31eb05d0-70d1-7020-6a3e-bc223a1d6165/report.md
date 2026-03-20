## Final Report

How many late days are you using for this assignment?
- Zero

1. Describe your experiments and observations

I trained RNN on the two datasets (Alphabet*100, warandpeace.txt). 

The alphabet dataset was simple and had an obvious pattern, which the model was able to learn quickly and reproduce with very few error. 

The War and Peace dataset on the other hand was a lot more complex and took a very long time to train. It had to recognize a lot more than just a repeating letter pattern so it makes sense that it took a long time to train. The model was a lot less accurate but it was able to generate sentences that had some grammar and punctuation, even though they were not coherent at all. 

The biggest change I made was changing the learning rate to 0.01 from 200. It brought the loss from the alphabet dataset from 139 to 3. I experimented with the other ones too but hidden_size didn't change the loss by much and the rest did have impacts but I saw that they would cause problems when we moved to a larger dataset so I didn't mess with them.


2. Analysis on final train and test loss for both datasets

Alphabet dataset:
- Default Train Loss: 3.2277
- Tuned Test Loss: 3.2315
![alt text](image.png)

The generated text was very accurate and almost the exact same as the original alphabet dataset. There were definitely some mistakes though. 

War and Peace:

I feel like I should get bonus points for not crashing out over this. The Epoch took an hour to load.

- Default Train Loss: 19.4843
- Final Test Loss: 1.705
![alt text](image-1.png)

The generated text was definitely in broken English and had some punctuation, but it barely had any grammar and some of the words were not actual words. 

3. Explain impact of changing temperature

Temperature is the randomness of the model when looking through the oputput probabilities. A lower temperature means that the model is less random and picks more probable characters whereas a higher temperature has more randomness and has more variation. From what I saw, a lower temperature of around 0.5 had actual English words and the words definitely could be used together but it created repetetive sentences that had no meaning. A higher temperature of 1.5 produced straight up gibberish with non-existant words but at least there were words.

4. Reflection

Implementing the RNN was an interesting process and I think the most important part was understanding how important tuning the model is. The basic model that I started with had an extremely high loss function and I had to tweak various parameters to get the loss function down. The hardest part was seeing what the results were though because it took so long to reload the warandpeace dataset. 