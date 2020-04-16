# -*- coding: utf-8 -*-
"""

"""

corpus = ['text text mining is interesting',
          'text mining is the same as data mining',
          'text and data mining have few differences']

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(corpus)

print(X.toarray()) ##aparecer um vetor das palavras do vetor

print(vec.get_feature_names()) ## numero de palavras do vetor

print(len(vec.get_feature_names())) ## numero da frequencia da palavra

print(vec.vocabulary_) ## a palvra a a frequencia da mesma