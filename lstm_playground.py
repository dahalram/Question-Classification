#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import re
import os
import time
from datetime import datetime
from utils import *
import pickle

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '2000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '70'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.0025'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
print "Reading the input file"
inputfile = open("all_data.label",'r')

inputfile_1 = open("list",'r')
input_1 = inputfile_1.readlines()
freq_voc = []
for item in input_1:
    freq_voc.append(item.split('\n')[0])


label_dic = {}
input_ = inputfile.readlines()
i = 0
y = []
x = []
sent = []
counter = 0
for item in input_:
    # For extracting the label of each sentence
    tag_ = re.findall(r'\w+:',item)
    tag = tag_[0].split(':')[0]
    if tag not in label_dic:
        # If the label is not in my dictionary I will add it to the dictionary
        # label_dic changes labels into numerical values 
        label_dic[tag] = i
        i = i + 1
    # Changing the label into numbers and put them inside y
    y.append(label_dic[tag])
    counter += 1

    sent.append(re.sub(r'\w+:\w+','',item)[0:-2])

sentences = sent
#sentences = itertools.chain(*[nltk.sent_tokenize(pi.lower()) for pi in sent])

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for i,sent in enumerate(sentences)]
# This se
for ii, vall in enumerate(tokenized_sentences):
    for jj,item in enumerate(vall):
        if item in freq_voc:
            tokenized_sentences[ii][jj] = 'stop_words'


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
X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])

X_train = X[0:5452]
y_train = y[0:5452]

X_test = X[5452:]
y_test = y[5452:]

def hard_sigmoid(x):
    return max(0, min(1, (x+1)/2.))

class LSTMNumpy:

    def __init__(self, word_dim, label_dim = 6, hidden_dim=128, bptt_truncate=-1):

        # Assign instance variables
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.word_dim = word_dim
        self.label_dim = label_dim

        # Randomly initialize the network parameters
        self.U1 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.U2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.U3 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.U4 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))

        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (label_dim, hidden_dim))
        
        self.W1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.W2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.W3 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.W4 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


def forward_propagation(self, x):

    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s_t because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s_t = np.zeros((T+1, self.hidden_dim))
    s_t[-1] = np.zeros(self.hidden_dim)
    
    i_t = np.zeros((T+1, self.hidden_dim))
    i_t[-1] = np.zeros(self.hidden_dim)

    f_t = np.zeros((T+1, self.hidden_dim))
    f_t = np.zeros(self.hidden_dim)

    o_t = np.zeros((T+1, self.hidden_dim))
    o_t = np.zeros(self.hidden_dim)

    g_t = np.zeros((T+1, self.hidden_dim))
    g_t = np.zeros(self.hidden_dim)

    # The outputs at each time step. Again, we save them for later.
    op_t = np.zeros(self.label_dim)
    
    # For each time step...
    for a in np.arange(T):
        # LSTM layer
        i_t[a] = hard_sigmoid(self.U[:,x_t[a]] + W[0].dot(s_t_prev) + b[0]) # adding a bias
        f_t[a] = hard_sigmoid(self.U[:,x_t[a]] + W[1].dot(s_t_prev) + b[1])
        o_t[a] = hard_sigmoid(self.U[:,x_t[a]] + W[2].dot(s_t_prev) + b[2])

        g_t[a] = np.tanh(self.U[:,x_t[a]] + W[3].dot(s_t_prev) + b[3])
        c_t[a] = c_t1_prev * f_t + g_t * i_t
        s_t[a] = np.tanh(c_t) * o_t

    op_t = softmax(self.V.dot(s_t[-2])) 


    # Get the word embedding vector
    # x_e = U[:, x_t] # CHECK TODO

    # LSTM layer
    # i_t = hard_sigmoid(self.U[0].dot(x_e) + W[0].dot(s_t_prev) + b[0]) # adding a bias
    # f_t = hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t_prev) + b[1])
    # o_t = hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t_prev) + b[2])

    # g_t = np.tanh(U[3].dot(x_e) + W[3].dot(s_t_prev) + b[3])
    # c_t = c_t1_prev * f_t + g_t * i_t
    # s_t = np.tanh(c_t) * o_t

    # op_t = softmax(self.V.dot(s_t) + c)[0] # Bias term c

    return [op_t, s_t]

LSTMNumpy.forward_propagation = forward_propagation

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=0)

LSTMNumpy.predict = predict

def calculate_total_loss(self, x, y):
    L = 0.0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[y[i]]
        # Add to the loss based on how off we were
        L += -1.0 * (np.log(correct_word_predictions))
    return L

def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    # y -> all y_train[i]
    N = len(y)
    return self.calculate_total_loss(x,y)/N
 
LSTMNumpy.calculate_total_loss = calculate_total_loss
LSTMNumpy.calculate_loss = calculate_loss

def bptt(self, x, y):
    T = len(x)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[y] -= 1.

    # For each output backwards...
    dLdV += np.outer(delta_o, s[-1].T)
    # Initial delta calculation
    delta_t = self.V.T.dot(delta_o) * (1 - (s[-1] ** 2))
    # Backpropagation through time (for at most self.bptt_truncate steps)
    for bptt_step in np.arange(max(0, (T-1)-self.bptt_truncate), (T-1)+1)[::-1]:
        #print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
        dLdW += np.outer(delta_t, s[bptt_step-1])              
        dLdU[:,x[bptt_step]] += delta_t
        # Update delta for next step
        delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
 
LSTMNumpy.bptt = bptt

# Performs one step of SGD.
def numpy_sgd_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

LSTMNumpy.numpy_sgd_step = numpy_sgd_step




































