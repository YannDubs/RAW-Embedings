# Rolling Average Word Embeddings (RAW Embeddings)
Since Mikolov's 2013 paper on word2vec many other word embeddings algorithms appeard (GloVe, FastText, ...). Although the algorithm are different the main idea behind the majority of them is similar : words that are often found in the same context should have similar vectors. When I explain how amazing word embeddings are to my friends that do not have a scientifical background, I do not talk about matrix factorization (GloVe), softmax (word2vec)... I explain that you basicallly go through a corpus and everytime you find 2 words in the same context you bring there vector closer together. The last time I explained that I wondered how good such a simple algorithm would work. Although this seems like the simplest and most intuitive baseline for word embeddings, I haven't been able to find someone talking abou it so I decided to implement it and see :).

## Main Idea
As prevously stated, the main idea would be to bring word embeddings closeer to each other when there corresponding words are found close to each other in the corpus. This could easily be done by keeping a rolling average and for all words you encounter you make it closer to the rolling average. The high level pseudocode would thus be:

**Pseudocode:**
```python
wordEmbeddings = random unit vectors
rollingAverage = 0
initialize weight # how much closer you will bring the vector to the rolling average
for epoch in range(nEpoch):
  for word in corpus:
    # bring current word embedding closer to rolling average
    wordEmbeddings[word] = weighted_average(rollingAverage,wordEmbeddings[word],weight)
    wordEmbeddings[word] *= 1 + weight # has to rescale if not all will converge to 0
    rollingAverage = rolling_average(rollingAverage,wordEmbeddings[word])
  decrease weight # necessary for convergence
```

## Comparaison to current word embeddings

### Advantages
Before any coding here is what I hypothesize could be better with RAW embeddings:

#### Computationally efficiency: 
Word2vec is a relatively slow algorithm (1 layer NN for each word and iterates multiple times through the corpus) but uses little RAM. On the other hand GloVe is fast (iterates through the corpus only once to compute the co-occurence matrix) but memory expensive. Last year, FastText improved a lot over Word2Vec by using approximations (hierachical softmax / hashing trick ...). 

RAW embeddings could possibly be faster than word2vec/Fasttext and more memory efficient than GloVe as they do not have to construct a co-occurence matrix and they only have to compute a weighted average at each step.

#### More Interpretable : 
By bringing word embeddings closer to the rolling average it means that vectors will never increase of size: their norm will be constrained (although I have to rescale them at the begining so that they do not all converge to 0). The scale of the vector will thus represent how often it is found around similar words. For example stopwords like "and" that are found near all words will have to be norm close to 0, which is what we want as they do not convey much meaning. This means that we will not have to remove stopwords any more. As the norm will convey the "strength" of the meaning, the angle will have to represent the meaning.

If this intuition turns out to be correct:
* Words with dissimilar meaning will have a cosine similarity close to 0.  
* Stopwords will have a norm close to 0.
* A superlative adjective like "stronger" should have the same angle as "strong" but a larger norm.

#### Less Hyperparameters : 
Word2vec is a relatively slow algorithm (1 layer NN for each word and iterates multiple times through the corpus) but uses little RAM. On the other hand GloVe is fast (iterates through the corpus only once to compute the co-occurence matrix) but memory expensive. Last year, FastText improved a lot over Word2Vec by using approximations (hierachical softmax / hashing trick ...). 

RAW embeddings could possibly be faster than word2vec and more memory efficient 

#### Better Suited for Simple Vector Representation of Sentences : 
As a baseline, vector representation of sentences are often simply computed by averaging or summing the word embeddings that constitute it. It is relatively suprising that such simple operations work well as summing these unconstrained word embeddings will give a vector which could represent an other word (i.e sentence embeddings can only encode the major topic of the phrase). In RAW embeddings summing should work well as each of their dimension is constrained: i.e summing will make the vectors longer effectively encoding the fact that we are talking about a sentence and not a vector. The longer the corpus / sentence is, the longer the summed vector will be, which seems logical as we would like to encode more information.

## Keep in Mind
Here are some notes to keep in mind:
* If all word embeddings would be equal to the same value then they will never change value. Although this shouldn't happen in practice due to the random initialization and rescaling, this is definitely not wanted. It might be a better idea to bring the current word vector closer to the rolling average but to push it further away form X randomly sampled other vectors. In the case of random initilaization this should be the same as rescaling the vector, as the average expectation of vectors will be 0 (i.e pushing away from 0 is the same as rescaling). But this would mean better theoretical propreties as the word embeddings would be independant of the initilaization when X -> inf and nEpoch -> inf.
