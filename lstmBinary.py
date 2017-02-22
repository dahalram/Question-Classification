import numpy as np
import random
from random import shuffle
import tensorflow as tf

## Training input data
train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
    	temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

## Training output data
train_output = []
for i in train_input:
    count = 0
    for j in i:
		if j[0] == 1:
			count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

## Generate the test data
NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print "test and training data loaded"

## Data dimensions: [Batch Size, Sequence Length, Input Dimension]
data = tf.placeholder(tf.float32, [None, 20,1]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 21])


num_hidden = 24
num_layers = 2
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

## Transpose the output to switch batch size with sequence size. 
## Take the values of outputs only at sequence's last input, 
## 		which means in a string of 20 we're only interested in the output we got at the 20th character
## 		and the rest of the output for previous characters is irrelevant.
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 1000
no_of_batches = int(len(train_input)) / batch_size


epoch = 200
for i in range(epoch):
	ptr = 0
	for j in range(no_of_batches):
		inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
		ptr+=batch_size
		sess.run(minimize,{data: inp, target: out})
	print "Epoch: ", str(i)

incorrect = sess.run(error,{data: test_input, target: test_output})
print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()




