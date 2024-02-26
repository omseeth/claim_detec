"""
This script helps with loading embeddings. Pretrained embeddings should be
saved at /embeddings/[file].
"""

import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path


class GloVe:
    """
    GloVe embeddings loaded from .txt file.
    """

    def __init__(self):
        self.debug = True
        self.embedding_path = (Path().cwd() / "./embeddings/glove.6B.300d.txt")
    
    def load_model(self):
        """
        Creates embedding vectors.

        return: dict where each word is key and its embedding the value.
        """

        glove = {}
        f = open(self.embedding_path, "r")
        
        if self.debug:
            print("Loading GloVe from file...")

        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove[word] = embedding
        
        return glove


class Word2Vec:
    """
    word2vec embeddings loaded from .bin.gz file. Vectors are being created
    with gensim.
    """

    def __init__(self):
        self.debug = True
        self.embedding_path = (Path().cwd() / "./embeddings/GoogleNews-vectors-negative300.bin.gz")

    def load_model(self):
        """
        Creates embedding vectors.

        return: KeyedVectors from gensim
        """
        
        if self.debug:
            print("Loading Word2Vec from file...")

        model = KeyedVectors.load_word2vec_format(self.embedding_path,
                                                  binary=True)
        
        return model


if __name__ == "__main__":

    embedding_test = GloVe()
    glove = embedding_test.load_model()
    print(list(glove.items())[:10])

    word2vec_test = Word2Vec()
    word2vec = word2vec_test.load_model()
    print(word2vec.most_similar('toronto'))
    print(word2vec['computer'])

