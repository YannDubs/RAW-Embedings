# Rolling Average Word Embeddings (RAW Embeddings)
Since Mikolov's 2013 paper on word2vec many other word embeddings algorithms appeard (GloVe, FastText, ...). Although the algorithm are different the main idea behind the majority of them is similar : words that are often found in the same context should have similar vectors. When I explain how amazing word embeddings are to my friends that do not have a scientifical background, I do not talk about matrix factorization (GloVe), softmax (word2vec)... I explain that you basicallly go through a corpus and everytime you find 2 words in the same context you bring there vector closer together. The last time I explained that I wondered how good such a simple algorithm would work. Although this seems like the simplest and most intuitive baseline for word embeddings, I haven't been able to find someone talking abou it so I decided to implement it and see :).

## Main Idea
As prevously stated, the main idea would be to bring word embeddings closeer to each other when there corresponding words are found close to each other in the corpus. The high level pseudocode would be:

**Pseudocode:**
```python
Initialize all word embeddings randomly
for epoch in range(nEpoch):
```

## Comparaison to current word embeddings

### Advantages
Before any coding here is what I hypothesize could be better with RAW embeddings:

### Computationally efficiency**: 
Word2vec is a relatively slow algorithm (1 layer NN for each word and iterates multiple times through the corpus) but uses little RAM. On the other hand GloVe is fast (iterates through the corpus only once to compute the co-occurence matrix) but memory expensive. Last year, FastText improved a lot over Word2Vec by using approximations (hierachical softmax / hashing trick ...). 

RAW embeddings could possibly be faster than word2vec and more memory efficient 
