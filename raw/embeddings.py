from raw.helpers import *

# should be cythonized
"""
cpdef np.ndarray[double,ndim=1] exp_rolling_average(np.ndarray[double,ndim=1] avg,
                                                    np.ndarray[double,ndim=1] vec,
                                                    double alpha):
    return avg * (1-alpha) + vec * alpha

cpdef np.ndarray[double,ndim=1] weighted_average(np.ndarray[double,ndim=1] new,
                                                 np.ndarray[double,ndim=1] old,
                                                 double weight):
    return new * weight + old * (1-weight)

cpdef np.ndarray[double,ndim=2] random_unit_vector(int npoints,int dim):
    cdef np.ndarray[double,ndim=2] vec = np.random.randn(dim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T
    
cpdef double get_weight(double weight, int count):
    return weight / np.log(2+count)
"""
def exp_rolling_average(avg,vec,alpha):
    return avg * (1-alpha) + vec * alpha

def weighted_average(new,old,weight):
        return new * weight + old * (1-weight)

def random_unit_vector(npoints,dim):
    vec = np.random.randn(dim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def get_weight(weight, count):
    return weight / np.log(2+count)


class Embedding:
    """
    Word embedding class, wrapper around a numpy array.
    
    Parameters
    ----------
    vocab : set of tokens, optional
        Vocabulary used to initialize the word embeddings.
        
    dim : int, optional
        Dimension of each word embedding.
        
    initializer : function, optional
        Function used to initialize each word embedding. 
        (nVec,dim) -> np.array of shape (nVec,dim).
        
    seed : int, optional
        Seed for randomness.    
    """
    def __init__(self, vocab=None, dim=100, initializer=np.random.randn, seed=123):
        self.seed = seed
        self.dim = dim
        self.hash = {} # could use real hash, for now dictionnary
        self.initializer = initializer
        self.embeddings = np.zeros((0,self.dim))
        
        np.random.seed(self.seed)
        if vocab is not None:
            self.add_embeddings(vocab)
            
    def _sample_vectors(self,n):
        """Samples n vectors with replacement."""
        idx = np.random.randint(low=0,high=len(self.embeddings),size=n)
        return self.embeddings[idx,:] 
    
    def _idx(self, word):
        """Retuns hash of a word."""
        return self.hash[word]
    
    def _word(self,idx):
        """Returns words associated with indices."""
        if not isinstance(idx,list):
            idx = [idx]
        words = [k for k,v in self.hash.items() if v in idx]
        return squeeze(words)
            
    def __getitem__(self, idx):
        """Gets the vector representation of the word or idx."""
        try:
            return self.embeddings[self._idx(idx)]
        except:
            return self.embeddings[idx]
    
    def gets(self,words):
        """Gets multiple words at once."""
        assert not isinstance(words,str)
        return self.embeddings[[self._idx(word) for word in words]]

    def __contains__(self, word):
        """Checks if a word is contained in the embedding."""
        return word in self.hash
    
    def __len__(self):
        """returns number of words in the vocabulary."""
        return self.embeddings.shape[0]
    
    @property
    def shape(self):
        return self.embeddings.shape
    
    def nearest_neigbours(self,word,k=5,metric="euclidean",**kwargs):
        """
        Finds the k most similar words to the given one.
        
        Parameters
        ----------
        word : str or int or np.ndarray[float,dim=1]
            Word for which to find the nearest neighbours.
            
        k : int, optional
            Number of nearest neighbours to return.
        
        metric : str or callable, optional
            The distance metric to use.  If a string, the distance function can be
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
            'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
            'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
            'wminkowski', 'yule'.
            
        **kwargs:
            Additional arguments to scipy.spatial.distance.cdist 
            
        Return
        ------
        out : OrderedDict
            Dictionnary of the k most simlar word and their distance to the query sorted by value.
        """
        if not isinstance(word,np.ndarray):
            word = self[word]
        return nearest_neighbours(self.embeddings, word, self._word, k=k, metric=metric, **kwargs)
    
    def intra_similarity(self,txt,metric="euclidean",**kwargs):
        """
        Returns a similarity matrix between each words of a phrase.
        
        Parameters
        ----------
        txt : str or list
            Words for which to compute intra-similarity.
            
        metric : str or callable, optional
            The scipy.spatial.distance.cdist metric to use.  Ex : 'cosine', 'dice', 'euclidean'.
            
        **kwargs:
            Additional arguments to scipy.spatial.distance.cdist 
            
        Return
        ------
        out : np.ndarray[float,ndim=2]
            Square similarity matrix of edge = len(txt) in order.
        """
        try :
            txt = txt.split()
        except :
            pass
        embeddings = self.gets(txt)
        return cdist(embeddings,embeddings,metric=metric,**kwargs)

    def nearest_analogy(self,pos,neg,k=5,metric="euclidean",**kwargs):
        """
        Analogy similarity.
        
        Parameters
        ----------
        pos : list
            Words to sum.
            
        neg : list
            Words to substract.
            
        metric : str or callable, optional
            The scipy.spatial.distance.cdist metric to use.  Ex : 'cosine', 'dice', 'euclidean'.
            
        **kwargs:
            Additional arguments to scipy.spatial.distance.cdist 
            
        Return
        ------
        out : OrderedDict
            Dictionnary of the k most simlar word and their distance to the query sorted by value.
            
        Example
        -------
            `pos=['Germany', 'Paris'], neg=['Berlin']` should give France.
        """
        querry = np.sum(self.gets(pos),axis=0) - np.sum(self.gets(neg),axis=0)
        return self.nearest_neigbours(querry,k=k,metric=metric,**kwargs)
                           
    def add_embeddings(self,vocab):
        """
        Adds initalized word embeddings from a vocabulary without modifying the previous word vectors.
        
        Parameters
        ----------
        vocab : set of tokens or string
            Vocabulary to add.
        """
        vocab = vocabularize(vocab)
        sizeVocab = len(self.hash)
        i = 0
        for w in vocab:
            if w not in self.hash:
                self.hash[w] = sizeVocab + i
                i += 1
                 
        nAdded = len(self.hash) - sizeVocab
        self.embeddings = np.concatenate((self.embeddings,self.initializer(nAdded,self.dim)), axis=0) 


# very dirty code : still dev mode
class RawEmbedding(Embedding):
    def __init__(self, vocab=None, dim=100, initializer=random_unit_vector, 
                 seed=123, weight=0.1, alpha=0.3, nEpoch=10):
        super(RawEmbedding,self).__init__(vocab=vocab, dim=dim, initializer=initializer, seed=seed)
        self.nEpoch = nEpoch
        self.alpha = alpha
        self.weight = weight
        self.countWords = {}
    
    def max_norm(self,k=5):
        norms = np.linalg.norm(self.embeddings,axis=1)
        bestIndices = k_argbest(norms,k,isMin=False)
        d = {self._word(i):float(n) for i,n in zip(bestIndices,norms[bestIndices])}    
        return sorted_dict(d) 
    
    def min_norm(self,k=5):
        norms = np.linalg.norm(self.embeddings,axis=1)
        bestIndices = k_argbest(norms,k)
        d = {self._word(i):float(n) for i,n in zip(bestIndices,norms[bestIndices])}    
        return sorted_dict(d) 
    
    def _compute_vector_rescale(self,rollingAverage,word,weight):       
        wv = self[word]
        tmp = weighted_average(rollingAverage,wv,weight) / (1 - weight)
        signs = np.sign(wv)
        return np.minimum(np.abs(wv),np.abs(tmp)) * signs
    
    def _compute_vector_repel1(self,rollingAverage,word,weight,nRepel=10):       
        wv = self.__getitem__(word)
        tmp = weighted_average(rollingAverage,wv,weight)
        print(tmp)
        stepSize = np.linalg.norm(wv-tmp)
        repellingVec = np.mean(em._sample_vectors(nRepel),axis=0)
        delta = repellingVec - tmp
        tmp = tmp + ((delta) / np.linalg.norm(delta) * stepSize / np.sqrt(self.dim))
        print(wv)
        print(tmp)
        print(stepSize)
        print(repellingVec)
        print(delta)
        assert False
        return tmp
    
    def _compute_vector_repel2(self,rollingAverage,word,weight,nRepel=10):       
        wv = self.__getitem__(word)
        tmp = weighted_average(rollingAverage,wv,weight)
        print(tmp)
        deltaNorm = np.abs(np.linalg.norm(wv)-np.linalg.norm(tmp))
        repellingVec = np.mean(em._sample_vectors(nRepel),axis=0)
        delta = repellingVec - tmp
        tmp = tmp + ((delta) / np.linalg.norm(delta) * deltaNorm)
        print(wv)
        print(tmp)
        print(stepSize)
        print(repellingVec)
        print(delta)
        assert False
        return tmp

    def train(self,files,method="rescaling"):
        for epoch in range(self.nEpoch):
            for file in files:
                with open(file, 'r',encoding="utf-8", errors = 'ignore') as myfile:
                    words = preprocessing(myfile.read()).split()
                self.add_embeddings(set(words))
                rollingAverage = random_unit_vector(1,self.dim).squeeze()
        
                for _ in range(2): # read other way around too
                    for w in words:
                        self.countWords[w] = self.countWords.get(w,0) + 1
                        weight = get_weight(self.weight,self.countWords[w])
                        if method == "rescaling":
                            updatedVec = self._compute_vector_rescale(rollingAverage,w,weight)
                        elif method == "repel1":
                            updatedVec = self._compute_vector_repel1(rollingAverage,w,weight)
                        elif method == "repel2":
                            updatedVec = self._compute_vector_repel2(rollingAverage,w,weight)
                        else:
                            raise ValueError('Unkown method : {}'.format(method))

                        self.embeddings[self._idx(w)] = updatedVec
                        rollingAverage = exp_rolling_average(rollingAverage,updatedVec,self.alpha) 


                    words = words[::-1]
        