# Rolling Average Word Embeddings (RAW Embeddings)
Since Mikolov's 2013 paper on word2vec many other word embeddings algorithms appeard (GloVe, FastText, ...). Although the algorithm are different the main idea behind the majority of them is similar : 

> Words that are often found in the same context should have similar vectors.

When I explain how amazing word embeddings are to my friends that don't have a scientifical background, I do not talk about matrix factorization (GloVe), softmax (word2vec)... I explain that you basicallly go through a corpus and everytime you find 2 words in the same context you bring their vector closer together. I thus started wondering how good such a simple algorithm would work. Although this seems like the simplest and most intuitive baseline for word embeddings, I haven't been able to find someone talking about it so I decided to test and see :thumbsup:.


## Main Idea
As prevously stated, the main idea would be to bring word embeddings closer to each other when their corresponding words are found close to each other in the corpus. This could easily be done by keeping a exponential rolling/moving average and for all words you encounter you make it closer to the rolling average. The high level pseudocode would thus be:

**Pseudocode:**
```python
wordEmbeddings = random unit vectors
rollingAverage = 0
initialize weight # how much closer you will bring the vector to the rolling average
initialize alpha # exponential smoothing parameter
for epoch in range(nEpoch):
  for word in corpus:
    # bring current word embedding closer to rolling average
    wordEmbeddings[word] = weighted_average(rollingAverage,wordEmbeddings[word],weight)
    wordEmbeddings[word] *= 1 + weight # has to rescale if not all will converge to 0
    rollingAverage = exponential_rolling_average(rollingAverage,wordEmbeddings[word],alpha)
  decrease weight # necessary for convergence
```

## Comparaison to current word embeddings

### Advantages
Before any coding here is what I hypothesize could be better with RAW embeddings:

#### Computationally efficiency: 
Word2vec is a relatively slow algorithm (1 layer NN for each word and iterates multiple times through the corpus) but uses little RAM. On the other hand GloVe is fast (iterates through the corpus only once to compute the co-occurence matrix) but memory expensive. Last year, FastText improved a lot over Word2Vec by using approximations (hierachical softmax / hashing trick ...). 

RAW embeddings could possibly be faster than word2vec/Fasttext and more memory efficient than GloVe as they do not have to construct a co-occurence matrix and they only have to compute a weighted average at each step.

#### More Interpretable : 

* **Norm = "strength" of meaning.** 
* **Angle = meaning.** 

Bringing word embeddings closer to the rolling average means that vectors will never increase of size: their norm will be constrained (although I have to rescale them at the begining so that they do not all converge to 0). The scale of the vector will thus represent how often it is found around similar words. For example stopwords like "and" that are found near all words will be approximatvely 0 (close to all vectors), which is what we want as they do not convey much meaning.  As the norm will convey the "strength" of the meaning, the angle will have to represent the meaning.

If this intuition turns out to be correct:
* Words with dissimilar meaning will have a cosine similarity close to 0.  
* Stopwords will have a norm close to 0. We thus do not have to remove them anymore.
* A superlative adjective like "stronger" should have the same angle as "strong" but a larger norm.

#### Less Hyperparameters : 
[Levy et al.](https://transacl.org/ojs/index.php/tacl/article/view/570/124) have shown that most current word embeddings can give the same accuracy on downstream tasks when trained with good hyperparameters. It is thus important to use the least possible hyperparameters. 

RAW Embeddings will only have 3 hyperparameters: the **number of epochs**, the **weight** used in the weighted average (by how much you will move the vector at each step), and the coefficient **$\alpha$** of the exponential smoothing in the rolling average.

Some hyperparameters that will be droped compared to *word2vec* are : *window size* (because exponential rolling average => infinte window with weighted average), *frequency threshold*, *definition of rare words*, *number of negative samples*, *context distribution smoothing*, *whether to add context words*.

#### Better Suited for Simple Vector Representation of Sentences : 
As a baseline, vector representation of sentences are often encoded by averaging or summing the word embeddings that constitute it. Although this works well in practice the encoded sentence will have a vector representation that could represent an other word (i.e sentence embeddings can only encode the major topic of the phrase). In RAW embeddings summing should work better as each of their dimension is constrained: i.e summing will make the vectors longer effectively encoding the fact that we are talking about a sentence and not a vector. **The longer the corpus / sentence is, the longer the summed vector will be**, which seems logical as we would like to encode more information.

Of course this could theoretically be learnt if we trained word embeddings at the same time as training the sentence encoder. But often we do it in 2 steps, so leaving the space of vectors with a norm higher than 1 for sentences and those with norm lower than 1 for words seems like a good idea.  

### Disadvantages
Of course even before testing out the idea I see some issues:

#### No Nice Statictical Interpretation
Besides the intuitive interpretation, there doesn't seem to be any statistical interpretation of how the vectors and built. the method is also extremely dependant on the initialization, indeed if we are not careful then all vectors will converge to the same value. Additionally without rescaling, all vectors will converge to 0 after an infinte number of steps, as the word embedding with the highest norm can never increase its norm. 

Although I haven't given it enough thought yet, I would be suprised if we cannot show that under random initialization the RAW Embeddings implicitely does the same as *word2vec* and *GloVe*. Indeed by making the current word embedding closer to the weighted average of previous word embeddings we effectively "try to predict the next word". By runing the algorithm in a forward and a backward pass this would mean that we effectively try to predict all the context window. If we are able to show that RAW Embeddings implicitely do the same thing as *word2vec* then they implicitely do the same thing as *GloVe* too due the transitivity and [this paper](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf).

#### Hard to Vectorize Computations
Although the computations are extremely simple, I do not see an easy way to vectorize the code. Indeed at each step we have to update the rolling average which depends on the current word embedding which depends itself on the rolling average. Everything thus seems to have to be done sequentially. 

Parallelization should still be possible by run each core on a different text. Indeed due the the large vocabulary, we will rarely be modifying the same word. 

In any case, I haven't given it enough thought yet as I first want to see if the alorithms works :smile: . 

#### Accuracy
The most important point : how well do RAW Embeddings work on downstream tasks. Hard to say without testin !

## Keep in Mind
Here are some notes to keep in mind:
* If all word embeddings are equal to the same value then they will never change. Although random initialization and rescaling should prohibit this in practice, the theory is still not satisfactory. One way of avoiding this would be to push the current word embedding further from X randomly sampled other vectors at each step. In the case of random initilaization this should be equivalent as rescaling the vector, as the expectation over all vectors will be 0 (i.e pushing away from 0 is the same as rescaling). But this would mean better theoretical propreties as the word embeddings would be independant of the initilaization when X -> inf and nEpoch -> inf.
* If we really wanted most sentences to have a norm higher than 1 we should constrain word embeddings to be positive (easily done by initializing with positive values for each dimensions). But the idea of letting negative dimensions for encoding words such as antonyms seems appealing.
* If you want to mimmic a *word2vec* symmetric window, you can run the algorithm both in a forward and backward pass of the corpus.
