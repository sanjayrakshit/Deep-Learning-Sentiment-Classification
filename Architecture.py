from load import Load_data
from config import *

import tensorflow as tf 
import time, pickle, sys


load = None
if sys.argv[-1] == "0":
	load = Load_data(batch_size, sequence_length)
	load.create_ids()
	load.parse_file()
	with open("object.Load_data", "wb") as f:
		pickle.dump(load, f)
else:
	with open("object.Load_data", "rb") as f:
		load = pickle.load(f)

vocab_size = len(load.wids) + 2

tf.reset_default_graph()
x = tf.placeholder(dtype=tf.int64, shape=[None, None], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, None], name="y")


def get_lstm_cell():
	lstm_single_layer = tf.contrib.rnn.LSTMCell(rnn_cell_size, name='LSTM_CELLS', state_is_tuple=True)
	dropout = tf.contrib.rnn.DropoutWrapper(lstm_single_layer, output_keep_prob=dropratio_lstm)
	return dropout


def create_rnn():
	cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for _ in range(rnn_layer_size)], state_is_tuple=True)
	return cell


def model():
	# getting the embedding
	with tf.name_scope("embedding"):
		word_embedding = tf.Variable(tf.truncated_normal([vocab_size, word_embedding_size], -0.1, 0.1))
		embedded_words = tf.nn.embedding_lookup(word_embedding, x)
	
	# creating the rnn structure
	CELL = create_rnn()
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
		OUTPUT, FINAL_STATE = tf.nn.dynamic_rnn(CELL, embedded_words, dtype=tf.float32)

	# fully connected layers
	dense = tf.contrib.layers.fully_connected(OUTPUT[:, -1], num_outputs=dense_layer_size,\
	activation_fn=tf.nn.leaky_relu, weights_initializer=tf.keras.initializers.he_normal(),\
	biases_initializer=tf.zeros_initializer())
	
	dense = tf.contrib.layers.dropout(dense, keep_prob=dropratio_dense)

	prediction = tf.contrib.layers.fully_connected(dense, num_outputs=num_classes, activation_fn=tf.nn.sigmoid,
												weights_initializer=tf.truncated_normal_initializer(),
												biases_initializer=tf.zeros_initializer())
	return prediction


y_hats = model()
print(y_hats)
print(y_hats.get_shape())

print("The trainable variables are ... ")
print(*tf.trainable_variables(), sep='\n')

with tf.name_scope('cost'):
	# cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hats)
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y), logits=y_hats)\
	# + tf.add_n([l2_reg_const*tf.nn.l2_loss(V) for V in tf.trainable_variables()]))
	# cost = tf.losses.mean_squared_error(y_hats, y)
	cost = tf.reduce_mean(tf.square(tf.subtract(y_hats, y))) + tf.reduce_mean([l2_reg_const*tf.nn.l2_loss(V) for V in tf.trainable_variables()])
	tf.summary.scalar('cost', cost)

with tf.variable_scope('train_step', reuse=tf.AUTO_REUSE):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    # corrects = tf.equal(tf.argmax(y, 1), tf.argmax(y_hats, 1))
    # accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    # tf.summary.scalar('accuracy', accuracy)
	correct_pred = tf.equal(tf.cast(tf.round(y_hats), tf.float32), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	tf.summary.scalar('accuracy', accuracy)


no_of_tr_batches = int(len(load.train)/batch_size)
no_of_ts_batches = int(len(load.test)/batch_size)
train_loss = []
test_loss = []
train_acc = []
val_acc = []

sess = tf.Session()
# session_conf = tf.ConfigProto(
#       intra_op_parallelism_threads=1,
#       inter_op_parallelism_threads=1)
# sess = tf.Session(config=session_conf)

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
	cc = 0; aa = 0
    # for training data
	for j in range(no_of_tr_batches):
		xx, yy = load.get_train_batch(i=j)
		_, c, s, a = sess.run([train_step, cost, summ, accuracy], {x: xx, y:yy})
		cc += c
		aa += a
		writer_tr.add_summary(s, iter_train)
		iter_train += 1
		print(f'{j+1} / {no_of_tr_batches}', end='\r')
	train_acc.append(aa/no_of_tr_batches)
	train_loss.append(cc/no_of_tr_batches)
	print('')
    
    # for validation data
	cc = 0; aa = 0
	for j in range(no_of_ts_batches):
		xx, yy = load.get_test_batch(i=j)
		c, s, a = sess.run([cost, summ, accuracy], {x: xx, y:yy})
		writer_ts.add_summary(s, iter_test)
		iter_test += 1
		cc += c
		aa += a
		print(f'{j+1} / {no_of_ts_batches}', end='\r')
	val_acc.append(aa/no_of_ts_batches)
	test_loss.append(cc/no_of_ts_batches)
	print('')

	print('Train loss:', train_loss[-1])
	print('Validation loss:', test_loss[-1])
	print('Train acc:', train_acc[-1])
	print('Val acc:', val_acc[-1])
	print('='*40)