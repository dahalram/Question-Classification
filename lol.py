#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
import pickle

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
print y_train[0]
print X_train[0]

class RNNNumpy:

	# word_dim = 8000
	# label_di
	def __init__(self, word_dim):
		self.label_dim = label_dim
		self.hidden_dim = hidden_dim

def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros(self.label_dim)
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
    o = softmax(self.V.dot(s[-2]))
    return [o, s]

def forward_prop_step(x_t, s_t1_prev):
      # This is how we calculated the hidden state in a simple RNN. No longer!
      # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
       
      # Get the word vector
      x_e = E[:,x_t]
       
      # GRU Layer
      z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
      r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
      c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
      s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
       
      # Final output calculation
      # Theano's softmax returns a matrix with one row, we only need the row
      o_t = T.nnet.softmax(V.dot(s_t1) + c)[0]
 
      return [o_t, s_t1]



'''
	1. find the order and initialize Us and Ws accordingly
				f    :       W    if previous case: dim [0 = x]
							 U
				o    :       W
				             U
				g    :       W
				             U
				i    :       W
				             U

		T   =     len(x)   // x is the input sentence, which is the no. units of the
						   // network
		i   =     np.zeros(T, size)

		for sure     :      (
							 np.zeros(T, dimensions of c) => coz c(t) = c(t-1) .* f + g .* i 
							 np.zeros(T, dimensions of s) => coz s(t) = tanh(c(t)) .* o
							 				s -> dimension 1 X 100 in previous cases         ,



		z_t1   =    U.x_e   +   W.s(t-1)



'''

import math
import numpy as np

class RNNNumpy:

	'''
		0   -   f
		1   -   o
		2   -   g
		3   -   i
	'''

	def __init__(self, dim_):
		self.W = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim), 
									(4, hidden_dim, hidden_dim))
		self.U = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim),
									(4, hidden_dim, vector_dim))
		pass

# e**j / 

def softmax(k):
	val = []
	l = len(k)
	j = sum([np.exp(k[i]) for i in range(l)])
	val = [np.exp(k[i])/j for i in range(l)]
	return val

def forward_propagation(self, x):
	T = len(x)
	c = np.zeros((T, dim_c))
	s = np.zeros((T+1, dim_s))
	i = np.zeros((T, dim_i))
	# v -> 100 x 8000
	# x -> order of x -> sth x hidden_dim
	# 
	for t in range(T):
		c[t] = np.multiply(c[t-1], f) + np.multiply(g, i)
		s[t] = np.multiply(np.tanh(c[t]), o)
		f[t] = hard_softmax(self.U[0].dot(x[t])+s[t-1].dot(W[0]))
		i[t] = hard_softmax(self.)

def bptt(self, input_sentence):
	T = len(input_sentence)
	









































