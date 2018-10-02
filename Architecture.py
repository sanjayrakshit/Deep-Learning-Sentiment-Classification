from load import Load_data
from config import *

import tensorflow as tf 
import time

load = Load_data(batch_size, sequence_length)
load.create_ids()
load.parse_file()

vocab_size = len(load.wids) + 2

tf.reset_default_graph()
x = tf.placeholder(dtype=tf.int64, shape=[None, sequence_length], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="y")


def get_embedding():
	with tf.variable_scope('get_embedding', reuse=tf.AUTO_REUSE):
		word_embedding = tf.get_variable('word_embedding', [vocab_size, word_embedding_size])
		return word_embedding


def get_lstm_cell():
	lstm_single_layer = tf.contrib.rnn.LSTMCell(rnn_cell_size, name='LSTM_CELLS', state_is_tuple=True)
	dropout = tf.contrib.rnn.DropoutWrapper(lstm_single_layer, output_keep_prob=dropratio_lstm)
	return dropout


def create_rnn():
	cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for _ in range(rnn_layer_size)], state_is_tuple=True)
	return cell


def model():
	# getting the embedding
	word_embedding = get_embedding()
	embedded_words = tf.nn.embedding_lookup(word_embedding, x)
	
	# creating the rnn structure
	CELL = create_rnn()
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
		OUTPUT, FINAL_STATE = tf.nn.dynamic_rnn(CELL, embedded_words, dtype=tf.float32)

	OUTPUT = tf.transpose(OUTPUT, [1,0,2]) # Step 1 to get the value from last time step
	OUTPUT = tf.gather(OUTPUT, OUTPUT.get_shape()[0]-1) # Step 2 to get the value from last time step

	# fully connected layers
	dense = tf.contrib.layers.fully_connected(OUTPUT, num_outputs=dense_layer_size,\
	activation_fn=tf.nn.leaky_relu, weights_initializer=tf.keras.initializers.he_normal(),\
	biases_initializer=tf.zeros_initializer())
	
	dense = tf.contrib.layers.dropout(dense, keep_prob=dropratio_dense)

	prediction = tf.contrib.layers.fully_connected(dense, num_outputs=num_classes, activation_fn=tf.nn.softmax,
												weights_initializer=tf.truncated_normal_initializer(),
												biases_initializer=tf.zeros_initializer())
	return prediction


y_hats = model()
print(y_hats)
print(y_hats.get_shape())

print("The trainable variables are ... ")
print(tf.trainable_variables())

with tf.name_scope('x_ent'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hats)\
	+ tf.add_n([l2_reg_const*tf.nn.l2_loss(V) for V in tf.trainable_variables()]))
	tf.summary.scalar('x_ent', cost)

with tf.variable_scope('train_step', reuse=tf.AUTO_REUSE):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    corrects = tf.equal(tf.argmax(y, 1), tf.argmax(y_hats, 1))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


no_of_tr_batches = int(len(load.train)/batch_size)
no_of_ts_batches = int(len(load.test)/batch_size)
train_loss = []
test_loss = []

sess = tf.Session()

some_time = str(time.time())

summ = tf.summary.merge_all()
name = "{}-{}-lr{}".format(rnn_cell_size, rnn_layer_size, learning_rate)
writer_tr = tf.summary.FileWriter(f'./logs/T_{name}_{some_time}')
writer_ts = tf.summary.FileWriter(f'./logs/V_{name}_{some_time}')
writer_tr.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

iter_train = iter_test = 0

for i in range(epoch):
	print('epoch:',i+1)
	time.sleep(1)
	cc = 0
    
    # for training data
	for j in range(no_of_tr_batches):
		xx, yy = load.get_train_batch(i=j)
		_, c, s = sess.run([train_step, cost, summ], {x: xx, y:yy})
		cc += c
		writer_tr.add_summary(s, iter_train)
		iter_train += 1
		print(f'{j+1} / {no_of_tr_batches}', end='\r')
	train_loss.append(cc/no_of_tr_batches)
	print('')
    
    # for validation data
	cc = 0
	for j in range(no_of_ts_batches):
		xx, yy = load.get_test_batch(i=j)
		c, s = sess.run([cost, summ], {x: xx, y:yy})
		writer_ts.add_summary(s, iter_test)
		iter_test += 1
		cc += c
		print(f'{j+1} / {no_of_ts_batches}', end='\r')
	test_loss.append(cc/no_of_ts_batches)
	print('')

print('Train loss:', train_loss[-1])
print('Validation loss:', test_loss[-1])
temp_test = list(zip(*load.test))
print('Validation accuracy:', sess.run(
	accuracy, {
		x: temp_test[0], y: temp_test[1]
	}
))
print('='*40)