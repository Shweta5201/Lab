from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use("Agg")
import os
import os.path
import argparse
import numpy as np
import tensorflow as tf

from math import sqrt
from reader import Reader
from rrnncell import RRNNCell

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

RATING_MATRIX_FILE = "rating_matrix.txt"
TRAIN_FILE = "train.data"
TEST_FILE= "test.data"

class RRNNGlobal(object):

    def __init__(self, num_users, num_items, num_units, 
        k, lr=0.001, rho=0.9, chunk_size=32, batch_size=None,
        is_training=True, cell_act=None, debug=False):

        # input: batch x num_inputs+1 x chunk_size
        self._inputs = tf.placeholder(dtype=tf.int32, 
            shape=[None, chunk_size, 2], name='inputs')
        # targets: batch x chunk_size
        self._targets = tf.placeholder(dtype=tf.int64, 
            shape=[None, chunk_size], name='targets')
        # seq_lengths: batch
        self._seq_len = tf.placeholder(dtype=tf.float32,
            shape=[None], name='seq_lengths')
        # batch_users: batch
        # To initialize cell state with user embedding
        #self._batch_users =  tf.placeholder(dtype=tf.int32,
        #    shape=[None], name='batch_users')
        #self._batch_users = tf.zeros_like(self._seq_len, 
        #  dtype=tf.int32, name='batch_users')

        # when assigned using numpy array by feeding to the model
        #u_emb = tf.Variable(
        #    tf.constant(0.0, shape=[num_users+1, emb_dim]), name='user_embeddings')
        #u_latent_factors = tf.placeholder(tf.float32, [num_users+1, emb_dim])
        #u_emb_init = u_emb.assign(u_latent_factors)

        #i_emb = tf.Variable(
        #    tf.constant(0.0, shape=[num_items+1, emb_dim]), name='item_embeddings')
        #i_latent_factors = tf.placeholder(tf.float32, [num_items+1, emb_dim])
        #i_emb_init = i_emb.assign(i_latent_factors)

        emb_dim = num_units
        with tf.variable_scope("Embeddings"):
          u_emb = tf.get_variable("user_embeddings",
            [num_users+1, emb_dim])
          i_emb = tf.get_variable("item_embeddings",
            [num_items+1, emb_dim])

        users = self._inputs[:,:,0]
        items = self._inputs[:,:,1]

        # shape(u_input) = batch_size x chunk_size x emb_dim
        u_input = tf.nn.embedding_lookup(u_emb, users)
        i_input = tf.nn.embedding_lookup(i_emb, items)
        #item_hist = tf.reshape(
        #  tf.cast(item_hist,tf.float32),[-1,num_items])
        #i_input = tf.matmul(item_hist, i_emb)
        #i_input = tf.reshape(i_input, [-1,chunk_size,num_units])

        inputs = tf.concat([u_input,i_input], 2)

        if debug:
          inputs = tf.Print(inputs, [inputs, tf.shape(inputs)],
            "Inputs for RNN cell",summarize=5*10*64*2)
        
        cell = RRNNCell(num_units, cell_act, debug)
        #cell = tf.contrib.rnn.GRUCell(num_units)
        #cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2,
        #  input_keep_prob=0.5, output_keep_prob=1.0)
        #cell = tf.contrib.rnn.MultiRNNCell([cell_1, cell_2])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        # Initializing hidden state with user embedding
        #self._initial_state = tf.nn.embedding_lookup(u_emb, self._batch_users)

        if debug:
          self._initial_state = tf.Print(self._initial_state,
            [self._initial_state, tf.shape(self._initial_state),tf.shape(self._batch_users)],
            "Initial hidden states for RNN")

        hidden_state, self._final_state = tf.nn.dynamic_rnn(cell, inputs, 
            sequence_length=self._seq_len, initial_state=self._initial_state)

        if debug:
          hidden_state = tf.Print(hidden_state, [hidden_state],
            "Hidden states from RNN")
        
        """ Trying NADE scores"""
        hidden_state = tf.reshape(hidden_state, [-1,chunk_size,emb_dim,1])
        with tf.variable_scope("Output_Connections"):
          weights_V = tf.get_variable("output_weights",
            [num_items+1, emb_dim, k+1])
          bias_B = tf.get_variable("output_bias",[num_items+1, k+1])
        _v = tf.nn.embedding_lookup(weights_V, items)
        _b = tf.nn.embedding_lookup(bias_B, items)       
        # shape(dot_vh) = batch_size x chunk_size x 1 x num_ratings+1
        dot_vh = tf.reduce_sum(tf.multiply(_v, hidden_state), 2)#, keep_dims=True)
        scores = tf.add(dot_vh,_b)
        # shape(scores) = batch_size x chunk_size x num_ratings+1
        logits = scores

        #logits = tf.layers.dense(inputs=hidden_state, units=k+1)
 
        if debug:
          logits = tf.Print(logits, [logits], "Output layer logits")

        ratings = tf.squeeze(tf.argmax(input=logits, axis=2))
        se = tf.to_float(tf.squared_difference(self._targets, ratings))
        logits = tf.reshape(logits, [-1, k+1])
        targets = tf.reshape(self._targets, [-1])
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits)
        sign_masks = tf.to_float(tf.sign(targets))
        masked = loss * sign_masks
        masked = tf.reshape(masked, [batch_size, chunk_size])
        se = tf.reshape(se, [-1])
        se = tf.reduce_sum(se * sign_masks)
        
        self._predictions = {
            "rating": ratings,
            "se": se,
        }
        self._cost = tf.reduce_mean(masked)
        #loss = tf.nn.weighted_cross_entropy_with_logits(targets=self._targets, logits=logits

        self._isTraining = is_training
        if not is_training:
          self._train_op = tf.no_op()
          return

        loss = k * tf.reduce_sum(masked)      # Optimizing on weighted cross-entropy 
        optimizer = tf.train.AdadeltaOptimizer()#RMSPropOptimizer(lr, decay=rho, epsilon=1e-8)#
        self._train_op = optimizer.minimize(loss)

    @property    
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def seq_lengths(self):
        return self._seq_len

    @property
    def batch_users(self):
        return None#self._batch_users

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def outputs(self):
        return self._predictions

    @property
    def loss(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def phase(self):
      if self._isTraining:
        return "Training"
      else:
        return "Validation"



def run_batch(sess,model,iterator,initial_state,msl,log):
    """"Runs on all chunks of a batch"""
    costs = 0
    se = 0
    state = initial_state
    chunk = 0
    chunk_log = msl/10
    for inputs, targets, seqLens in iterator:
      fetches = [model.final_state, model.outputs,
        model.loss, model.train_op]
      feed_dict = {}
      feed_dict[model.inputs] = inputs
      feed_dict[model.targets] = targets
      feed_dict[model.seq_lengths] = seqLens
      feed_dict[model.initial_state] = state
      state, outputs, loss, _ = sess.run(fetches, feed_dict)
      costs = np.add(costs,loss)
      se = np.add(se,outputs["se"])
      chunk += seqLens
      if chunk > chunk_log:
        log.write("{} loss: {}\n".format(model.phase,costs))
        print("{} loss: {}, MSE: {}".format(model.phase,costs,se/seqLens))
        chunk = 0
    if chunk > 0:
      log.write("{} loss: {}\n".format(model.phase,costs))
      print("{} loss: {}, MSE: {}\n".format(model.phase,costs,se/seqLens))
#      print("outputs: ",outputs["rating"])
#      print("targets: ",targets)
#      print(seqLens)
    return state, costs, se

def run_epoch(train_model, test_model, sess,
    train_data, test_data, log):
    """Runs on all batches"""
    train_error = 0
    train_se = 0
    train_iter = train_data.iterate_batches()
    for batch in train_iter:
      # Initializing cell hidden states
      state = sess.run(train_model.initial_state)#,
        #{train_model.batch_users: b_u})
      # Training each batch of users
      state, loss, se = run_batch(sess, train_model, batch, state,
        train_data.max_seq_len,log)
      train_se = np.add(train_se,se)
      train_error = np.add(train_error,loss)
    train_RMSE = np.sqrt(train_se/train_data.max_seq_len)

    print("Starting testing...")    
    test_error = 0
    test_se = 0
    test_iter = test_data.iterate_batches()
    for batch in test_iter:
      state = state
        #,{test_model.batch_users: b_u})
      state, loss, se = run_batch(sess, test_model, batch, state,
        test_data.max_seq_len,log)
      test_se = np.add(test_se,se)
      test_error = np.add(test_error,loss)
    test_RMSE = np.sqrt(test_se/test_data.max_seq_len)
    return train_error, test_error, train_RMSE, test_RMSE

def main(args):

    rmf_path = os.path.join(args.data_path,RATING_MATRIX_FILE)
    train_file = os.path.join(args.data_path,TRAIN_FILE)
    test_file = os.path.join(args.data_path,TEST_FILE)

    print(rmf_path)
    print(train_file)
    print(test_file)

    rm = Reader.get_rating_matrix(rmf_path)
    num_users, num_items = rm.shape
    train_data = Reader(train_file, args.batch_size,
      args.chunk_size,args.num_ratings,num_users,num_items)
    test_data = Reader(test_file, args.batch_size,
      args.chunk_size,args.num_ratings, num_users, num_items)
    epochs = args.num_epochs

    print(rm.shape)
    del rm
    print("Number of users: ",num_users)
    print("Number of items: ",num_items)

    if args.chunk_size == 0:
      cs = min(train_data.max_seq_len, test_data.max_seq_len)
    else:
      cs = min(args.chunk_size, train_data.max_seq_len,
             test_data.max_seq_len)
    print(cs) 
    settings = {
        "batch_size": args.batch_size,
        "chunk_size": cs,
        "lr": args.learning_rate,
        "rho": args.decay,
        "num_users": num_users,
        "num_items": num_items,
        "num_units": args.num_units,
        "k": args.num_ratings,
        "cell_act":args.cell_activation,
        #"debug": True,
    }

    print("Model configuration:\n",settings)

    train_errors = list()
    test_errors = list()
    _train_rmse = list()
    _test_rmse = list()

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_normal_initializer(
                mean=0, stddev=1/sqrt(args.num_units))
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = RRNNGlobal(is_training=True, **settings)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = RRNNGlobal(is_training=False, **settings)

        # Initializing all model weights
        tf.global_variables_initializer().run()

        with open("log.txt","w+",0) as log:
          log.write("Rating matrix file path : [{}]\n".format(rmf_path))
          log.write("Training data file : [{}]\n".format(train_file))
          log.write("Testing data file : [{}]\n".format(test_file))
          log.write("Model configuration:\n{}\n".format(settings))
          
          for i in range(1,epochs+1):
            train_err, test_err,train_rmse, test_rmse = run_epoch(
              train_model,test_model,session,train_data,test_data,log)
            log.write("Epoch {}: Training error {}\n".format(i,train_err))
            log.write("Epoch {}: Test error {}\n".format(i,test_err))
            log.write("Epoch {}: Training RMSE {}\n".format(i,train_rmse))
            log.write("Epoch {}: Test RMSE {}\n".format(i,test_rmse))
            print("Epoch {}: Training error {}\n".format(i,train_err))
            print("Epoch {}: Test error {}\n".format(i,test_err))
            print("Epoch {}: Training RMSE {}\n".format(i,train_rmse))
            print("Epoch {}: Test RMSE {}\n".format(i,test_rmse))
            train_errors.append(train_err)
            test_errors.append(test_err)
            _train_rmse.append(train_rmse)
            _test_rmse.append(test_rmse)

          x = range(epochs)  
          f, axarr = plt.subplots(2, sharex=True)
          axarr[0].plot(x, train_errors)
          axarr[0].plot(x, test_errors)
          axarr[0].scatter(x, train_errors)
          axarr[0].scatter(x, test_errors)
          axarr[0].set_title('Cross-entopy loss')
          axarr[1].plot(x, _train_rmse)
          axarr[1].plot(x, _test_rmse)
          axarr[1].scatter(x, _train_rmse)
          axarr[1].scatter(x, _test_rmse)
          axarr[1].set_title('RMSE')
          plt.savefig("results.png")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path at which train and test data is present")
    parser.add_argument("num_ratings",type=int,
      help="number of ratings of the rating system used")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--num-units", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--cell-activation", type=str, default="tanh")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(args)
