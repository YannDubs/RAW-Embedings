import numpy as np
import string
from collections import OrderedDict
from scipy.spatial.distance import cdist

def squeeze(l):
    """Returns the element if list only contains it."""
    return l[0] if len(l) == 1 else l
    
def vocabularize(txt):
    """Make a set of vocabulary from a txt or list."""
    try : 
        txt = txt.split()
    except :
        pass
    return set(txt)

def unsqueeze(vec):
    """Goes from a numpy vector to an array with a single row."""
    if vec.ndim == 1:
        return np.expand_dims(vec,0)
    return vec

def sorted_dict(d, isReverse=False):
    """Sorts a dictionnary by value in a OrderedDict."""
    return OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse=isReverse))

def k_argbest(vec,k,isMin=True):
    """Returns effeciently the k best (max/min) values in an array."""
    vec = np.squeeze(vec)
    if not isMin:
        vec = -vec
    return np.argpartition(vec, k)[:k]
    
def nearest_neighbours(matrix, vec, mapper=lambda x : x, k=5,metric="cosine", **kwargs):
    """Retuns nearest neigbours between a word and embeddings."""
    vec = unsqueeze(vec)
    dists = cdist(matrix,vec,metric=metric,**kwargs)

    if len(matrix)-1 <= k:
        bestDists, bestIndices = dists, range(len(matrix))    
    else:
        bestIndices = k_argbest(dists,k+1)
        bestDists = dists[bestIndices]

    d = {mapper(i):float(d) for i,d in zip(bestIndices,bestDists) if d != 0}    
    return sorted_dict(d) 

def random_unit_vector(npoints,dim):
    vec = np.random.randn(dim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

def preprocessing(txt):
    """Preprocesses the text"""
    translator=str.maketrans('','',string.punctuation + string.digits)
    return txt.strip().replace('\n', '').lower().translate(translator)