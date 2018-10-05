from load import Load_data
from config import *

import tensorflow as tf 
import time, pickle, sys, numpy as np
from tqdm import tqdm
from collections import namedtuple



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


def make_lstm_cell(rnn_cell_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_cell_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop


def build_rnn(vocab_size, word_embedding_size, batch_size, rnn_cell_size, rnn_layer_size, 
            learning_rate, dense_layer_size):
    '''Build the Recurrent Neural Network'''

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    keep_prob_dense = tf.placeholder(tf.float32, name='keep_prob_dense')

    # Create the embeddings
    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.truncated_normal((vocab_size, word_embedding_size), -0.1, 0.1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        cell = tf.contrib.rnn.MultiRNNCell([make_lstm_cell(rnn_cell_size, keep_prob) for _ in range(rnn_layer_size)])
    
    # Set the initial state
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 initial_state=initial_state)    
    
    # Create the fully connected layers
    with tf.name_scope("fully_connected"):
        
        # Initialize the weights and biases
        # weights = tf.truncated_normal_initializer(stddev=0.1)
        # biases = tf.zeros_initializer()
        
        dense = tf.contrib.layers.fully_connected(
            inputs = outputs[:, -1],
            num_outputs = dense_layer_size,
            activation_fn = tf.nn.leaky_relu,
            weights_initializer = tf.keras.initializers.he_normal(),
            biases_initializer = tf.zeros_initializer())
        dense = tf.contrib.layers.dropout(dense, keep_prob_dense)
        
    # Make the predictions
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense, 
                                                        num_outputs = num_classes, 
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer = tf.truncated_normal_initializer(),
                                                        biases_initializer = tf.zeros_initializer())
        tf.summary.histogram('predictions', predictions)
    
    # Calculate the cost
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    
    # Train the model
    with tf.name_scope('train'):    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Determine the accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'keep_prob', 'keep_prob_dense', 'initial_state', 'final_state','accuracy',
                    'predictions', 'cost', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph

no_of_tr_batches = int(len(load.train)/batch_size)
no_of_ts_batches = int(len(load.test)/batch_size)

def train(model, epochs, log_string):
    '''Train the RNN'''

    saver = tf.train.Saver()
    
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        valid_loss_summary = []
        
        # Keep track of which batch iteration is being trained
        iteration = 0

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/t_{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('./logs/v_{}'.format(log_string))

        for e in range(epochs):
            state = sess.run(model.initial_state)
            
            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []
            
            for i in tqdm(range(no_of_tr_batches), total=no_of_tr_batches):
                x, y = load.get_train_batch(i=i)
                feed = {model.inputs: x,
                        model.labels: y,
                        model.keep_prob: dropratio_lstm,
                        model.keep_prob_dense: dropratio_dense,
                        model.initial_state: state}
                summary, loss, acc, state, _ = sess.run([model.merged, 
                                                            model.cost, 
                                                            model.accuracy, 
                                                            model.final_state, 
                                                            model.optimizer], 
                                                        feed_dict=feed)                
                
                # Record the loss and accuracy of each training batch
                train_loss.append(loss)
                train_acc.append(acc)
                
                # Record the progress of training
                train_writer.add_summary(summary, iteration)
                
                iteration += 1
            
            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc) 

            val_state = sess.run(model.initial_state)
            for i in tqdm(range(no_of_ts_batches), total=no_of_ts_batches):
                x, y = load.get_test_batch(i=i)
                feed = {model.inputs: x,
                        model.labels: y,
                        model.keep_prob: 1,
                        model.keep_prob_dense: 1,
                        model.initial_state: val_state}
                summary, batch_loss, batch_acc, val_state = sess.run([model.merged, 
                                                                        model.cost, 
                                                                        model.accuracy, 
                                                                        model.final_state], 
                                                                        feed_dict=feed)
                
                # Record the validation loss and accuracy of each epoch
                val_loss.append(batch_loss)
                val_acc.append(batch_acc)
            
            # Average the validation loss and accuracy of each epoch
            avg_valid_loss = np.mean(val_loss)    
            avg_valid_acc = np.mean(val_acc)
            valid_loss_summary.append(avg_valid_loss)
            
            # Record the validation data's progress
            valid_writer.add_summary(summary, iteration)

            # Print the progress of each epoch
            print("Epoch: {}/{}".format(e, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc),
                  "Valid Loss: {:.3f}".format(avg_valid_loss),
                  "Valid Acc: {:.3f}".format(avg_valid_acc))

            # Stop training if the validation loss does not decrease after 3 epochs
            if avg_valid_loss > min(valid_loss_summary):
                print("No Improvement.")
                stop_early += 1
                if stop_early == 3:
                    break   
            
            # Reset stop_early if the validation loss finds a new low
            # Save a checkpoint of the model
            else:
                print("New Record!")
                stop_early = 0
                checkpoint = "./models/sentiment_{}.ckpt".format(log_string)
                saver.save(sess, checkpoint)

[_, unique] = ("%0.4f" %(time.time() / 1000)).split('.')
log_string = "{}-{}-{}-{}".format(rnn_cell_size, rnn_layer_size, sys.argv[1], unique)
model = build_rnn(vocab_size, word_embedding_size, batch_size, rnn_cell_size, rnn_layer_size, 
learning_rate, dense_layer_size)            
train(model, epochs, log_string)


# def make_predictions(lstm_size, multiple_fc, fc_units, checkpoint):
#     '''Predict the sentiment of the testing data'''
    
#     # Record all of the predictions
#     all_preds = []

#     model = build_rnn(n_words = n_words, 
#                       embed_size = embed_size,
#                       batch_size = batch_size,
#                       lstm_size = lstm_size,
#                       num_layers = num_layers,
#                       dropout = dropout,
#                       learning_rate = learning_rate,
#                       multiple_fc = multiple_fc,
#                       fc_units = fc_units) 
    
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         # Load the model
#         saver.restore(sess, checkpoint)
#         test_state = sess.run(model.initial_state)
#         for _, x in enumerate(get_test_batches(x_test, batch_size), 1):
#             feed = {model.inputs: x,
#                     model.keep_prob: 1,
#                     model.initial_state: test_state}
#             predictions = sess.run(model.predictions, feed_dict=feed)
#             for pred in predictions:
#                 all_preds.append(float(pred))
                
#     return all_preds