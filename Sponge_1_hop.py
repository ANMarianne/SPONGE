#!/usr/bin/env python
# coding: utf-8

# fix random seed
rand_seed = 1234
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(rand_seed)# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(rand_seed)# 3. Set `numpy` pseudo-random generator at a fixed value
import sys
import numpy as np
np.random.seed(rand_seed)# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(rand_seed) 
from tensorflow import keras
from tensorflow.keras import layers
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
from sklearn.neighbors import NearestNeighbors
import sklearn
from joblib import Parallel, delayed
import zipfile
import time
import csv



tf.compat.v1.flags.DEFINE_string('f','','')

#Experiments parameters


# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_neigh', 3, 'Number of neighbors')
flags.DEFINE_integer('data_id', 1, 'id of the dataset to call')
flags.DEFINE_integer('val_num', 50, 'Number of validation datapoints')
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2. -1 means not to use this layer')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (keep probability) 0, 0.25, 0.5')
flags.DEFINE_float('lr', 1e-2, 'Learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('es_patience', 10, 'Patience for early stopping')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('dataset', 'databiggauss.npz', 'Data file path  0.17657')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('length_scale', 0.5, 'Length scale of the RBF kernel. 1, .5, .1, .05')



# Defining the metrics

def mse_loss(logits, labels):
    """ Computes the mse loss of a batch.

    Args:
      logits: Model output of size [batch, 1]
      labels: True labels of the batch [batch, 1]

    Returns:
      loss: Float 32
    """

    loss = tf.reduce_mean((logits-labels)**2)
    return loss


def clean_nan(lists):
    nan_ind=list(np.argwhere(np.isnan(lists)).flatten())
    return nan_ind


#   Preparing the dataset for the input pipeline

def load_kriging_data(file_path, num_neigh, num_val, log_y=False):
    """ Load the data from a file path.

    Args:
      file_path: Path where the data is stored
      num_neigh: number of nearest neighbours
      num_val: the number of validation data
      log_y: boolean specifying whether to assume a log gaussian.

    Returns:
      features: Collapsed train and test Input data
      y_tilde: Nonzero entries correspond to training targets
      y_train_test: The collapsed train and test targets
      ind_nbs: Indices of points corresponding to the neighbors
      Ntrain: The number of training instances
    """

    # Loading the data from the file path
    data = np.load(file_path)

    # Collecting the train and the test set
    X_train = np.ndarray.astype(data['Xtrain'], np.float32)
    X_test = np.ndarray.astype(data['Xtest'], np.float32)
    Y_train = (np.ndarray.astype(data['Ytrain'], np.float32)).reshape(-1, 1)
    Y_test = (np.ndarray.astype(data['Ytest'], np.float32)).reshape(-1, 1)

    print(X_train.shape)
    Ntrain = X_train.shape[0]

    # merge train and test # get features and coordinates
    coords_features = np.concatenate([X_train, X_test], axis=0)
    coords = coords_features[:, 0:2]
    features = coords_features

    # feature normalization as sugested in the original code of the authors
    fmean = np.mean(features[0:Ntrain], axis=0, keepdims=True)
    fstd = np.std(features[0:Ntrain], axis=0, keepdims=True)
    features = (features - fmean) / (fstd )

    y_train_test = np.concatenate([Y_train, Y_test], axis=0)

    # take the log of y coordinates if necessary
    if log_y:
        y_train_test = np.log(y_train_test)

    y_tilde = y_train_test.copy()
    y_tilde[Ntrain-num_val:] = 0

    # getting neighbors from training data and test data.
    knn = NearestNeighbors(n_neighbors=num_neigh).fit(coords[0:Ntrain])
    ind_nbs_train_val = knn.kneighbors(return_distance=False)
    ind_nbs_test = knn.kneighbors(coords[Ntrain:], return_distance=False)

    ind_nbs = np.concatenate([ind_nbs_train_val, ind_nbs_test], axis=0)

    return features, y_tilde, y_train_test, ind_nbs, Ntrain


class MyConstraints(tf.keras.constraints.Constraint):

 def __call__(self, w):
   w = w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
   return w


# Build a graph convolutional layer
class MyGraphconvNeigh(layers.Layer):
  def __init__(self, hidden_out):
    """ Instatiating the graph convolutional layer.

    Args:
      hidden_out: output of the dense
    """
    super(MyGraphconvNeigh, self).__init__()

    self.num_outputs = hidden_out

  def build(self, input_shape):
    """ calling the graph conv layer builds it to create weights.

    Args:
        input_shape: input shape
    """

    tf.random.set_seed(rand_seed)
    if len(input_shape)==2:
      self.graphconvNeig = self.add_weight("GConvNeigh",
                                  shape=[input_shape[1][-1],
                                         self.num_outputs])#, dtype=tf.float64)
    else:
      self.graphconvNeig = self.add_weight("GConvNeigh",
                                  shape=[input_shape[-1],
                                         self.num_outputs])

  def call(self, inputs):
    """ Executed when apply the layer to inputs.

    Args:
      inputs: tuple of input (3d of shape [batch, K+1, hidden]) and normalised adj

    Returns:
      output : tuple of output(3d of shape [batch, K+1, hidden_out]) and normalised adj
    """
    if len(inputs)==2:
      Adj, H = inputs
      Adj = Adj[:,0,:]      #uncomment for one ls
      result0 = Adj @ H
      output = result0 @ self.graphconvNeig
      #Adj_new = tf.reshape(Adj[:, 0, 0:num_neigh +1 ], [Adj.shape[0],1,num_neigh +1]) # initial line
    else:
      output =  inputs @ self.graphconvNeig
    return Adj, output


# Build a graph convolutional layer
class MyAdjconvLayer(layers.Layer):
  def __init__(self):
    """ Instatiating the graph convolutional layer.

    Args:
      hidden_out: output of the dense
    """
    super(MyAdjconvLayer, self).__init__()

  def build(self, input_shape):
    """ calling the graph conv layer builds it to create weights.

    Args:
        input_shape: input shape
    """

    tf.random.set_seed(rand_seed)
    if len(input_shape)==2:
      self.adjconv = self.add_weight("AConv",
                                  shape=[input_shape[0][1],
                                         1],trainable=True, constraint=MyConstraints())#, regularizer='l1')
    else:
      self.adjconv = self.add_weight("AConv",
                                  shape=[input_shape[1],
                                         1],trainable=True,constraint=MyConstraints())

  def call(self, inputs):
    """ Executed channel wise convolution when apply the layer to inputs.

    Args:
      inputs: tuple of input (3d of shape [batch, K+1, hidden]) and normalised adj

    Returns:
      output : tuple of output(3d of shape [batch, K+1, hidden_out]) and normalised adj
    """

    adjconv = self.adjconv/tf.reduce_sum(self.adjconv)


    if len(inputs)==2:
      adj, H = inputs
      batch, _, n1, n2,  = adj.shape
      D1 = tf.reshape(tf.tensordot(adj, adjconv, axes=[[1],[0]]), [batch, n1, n2])
      D2 = (tf.cast(tf.reshape(tf.reduce_sum(D1, axis=2), [-1,n1,1]), tf.float32))
      output = D1/D2
    else:
      output =  inputs @ self.adjconv
    return output, H


# Build a graph convolutional layer
class MyGraphconvLayer(layers.Layer):
  def __init__(self, hidden_out):
    """ Instatiating the graph convolutional layer.

    Args:
      hidden_out: output of the dense
    """
    super(MyGraphconvLayer, self).__init__()

    self.num_outputs = hidden_out

  def build(self, input_shape):
    """ calling the graph conv layer builds it to create weights.

    Args:
        input_shape: input shape
    """

    tf.random.set_seed(rand_seed)
    self.graphconv = self.add_weight("GConv",
                                  shape=[input_shape[-1][-1],
                                         self.num_outputs])

  def call(self, inputs):
    """ Executed when apply the layer to inputs.

    Args:
      inputs: tuple of input (3d of shape [batch, K+1, hidden]) and normalised adj

    Returns:
      output : tuple of output(3d of shape [batch, K+1, hidden_out]) and normalised adj
    """

    A_bar, H = inputs
    input_shape = H.shape
    if len(input_shape)==3:
        result = tf.einsum('bij,bjk->bik', A_bar, H)

    else:
        result = tf.matmul(A_bar, H)

    output = tf.matmul(result, self.graphconv)

    return output, A_bar


# Build a KCN model
class KCN(keras.Model):
  def __init__(self, hidden_sizes, output_dim, kcn_bool=True):
    """ Instatiating the model.

    Args:
      hidden_sizes: list of hidden output sizes
      dropout_rate: prob to drop neurones
      kcn_bool: if False KCN=GCN, if true, KCN base then provide
    """
    super(KCN, self).__init__(name='')

    self.kcn_bool = kcn_bool

    self.hidden0 = MyAdjconvLayer()                  #comment for one ls

    self.hidden1 = MyGraphconvNeigh(hidden_sizes[0])

    if self.kcn_bool == True:
        tf.random.set_seed(rand_seed)
        self.dense = tf.keras.layers.Dense(output_dim, use_bias=False)

  def call(self, input_tensor, training=False):
    """ Executed when apply the layer to inputs.

    Args:
      input_tensor: output of the GCN, 3d tensor of shape [batch, K+1, hidden_prev]
      training: boolean specifying the mode
    Returns:
      output : 3d tensor of shape [batch, K+1, output_dense] if training=true, otherwise[K+1, out]
    """


    # x, A_bar = self.hidden1(input_tensor)
    x = self.hidden0(input_tensor)          # comment for one ls
    A_bar, x = self.hidden1(x)                  # comment for one ls
    # if training==True:
    #     x = tf.nn.dropout(x, FLAGS.dropout)
    x = tf.nn.relu(x)

    if self.kcn_bool == True:
        if len(x.shape)==3:
            x = self.dense(x[:, 0])
        else:
            x = self.dense(tf.reshape(x[0], [1,-1]))
    return x


def adjacency_matrix(coords, length_scale):
    """Compute the adjacency matrix from the coordinates.

    Args:
        coords: coords matrix(shape [num_data, 2])
        length_scale: length scale of the kernel
    Returns:
        adj : adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
    """

    adj = sklearn.metrics.pairwise.rbf_kernel(coords, gamma=1.0 / (2.0 * length_scale * length_scale)) 
    # adj = sklearn.metrics.pairwise.laplacian_kernel(coords, gamma=1.0 / (np.sqrt(2.0) * length_scale)) 
    return adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix from the coordinates.

    Args:
      adj: coords matrix(shape [num_data, 2])

    Returns:
      adj_normalized : normalized adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
    """

    Deg = np.sum(adj, axis=1)
    degpow = 1./(np.sqrt(Deg))
    degpow[np.isnan(degpow)] = 0.0
    degpow = np.diag(degpow)
    adj_normalized = degpow @ adj @ degpow
    return adj_normalized

def normalize_adj_3d(adj):
    """Symmetrically normalize adjacency matrix from the coordinates.

    Args:
      adj: coords matrix(shape [num_data, 2])

    Returns:
      adj_normalized : normalized adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
    """

    batch, n1, n2 = adj.shape
    Deg = (tf.cast(tf.reshape(tf.reduce_sum(adj, axis=2), [-1,n1]), tf.float32))
    degpow = 1./(np.sqrt(Deg))
    degpow[np.isnan(degpow)] = 0.0
    degpows = []
    for i in range(batch):
      bla = np.diag(degpow[i])
      degpows.append(bla)
    degpows = tf.convert_to_tensor(np.stack(degpows, axis=0)) 
    adj_normalized = degpows @ adj @ degpows
    return adj_normalized

def asym_normalize_adj(adj):
    """Asymmetrically normalize adjacency matrix from the coordinates.

    Args:
      adj: coords matrix(shape [num_data, 2])
    Returns:
      adj_normalized : normalized adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
    """

    Deg = np.sum(adj, axis=1)
    degpow = 1./(Deg)
    degpow[np.isnan(degpow)] = 0.0
    degpow = np.diag(degpow)
    adj_normalized = degpow @ adj
    return adj_normalized


def row_norm_adj(adj):
    """Symmetrically normalize adjacency matrix from coordinates, return 1st row

    Args:
      adj: coords matrix (shape [num_daya, 2])
    Returns:
      one_row: first row of the normalized adjacency matrix (shape [num_neigh+1] )
    """

    one_row = np.zeros(shape=(adj.shape[-1], ))
    left_deg = np.sum(adj[0,:])
    leftdegpow = 1./(np.sqrt(left_deg))
    for k in range(adj.shape[-1]):
       right_deg = np.sum(adj[k,:])
       rightdegpow = 1./(np.sqrt(right_deg))
       one_row[k] = leftdegpow * adj[0, k] * rightdegpow
    return one_row


# Build the input of the training/test pipeline
def build_input_h(ind, neigh_ind, feat, ytilde, length_scale):
  """Compute the input matrix H and the normalize adjacency matrix for the input.

    Args:
        ind: Indices of points at which topredict
        neigh_ind: Indices of neighbors
        Xtrain: Training inputs
        ytilde: Training Targets
        length_scale: Length scale value

    Returns:
        adj_t : adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
        H_t : Input to the pipeline 
  """

  _, dim_feat = feat.shape
  _, num_neigh = neigh_ind.shape
  # H_t = np.zeros((len(ind), num_neigh+1, dim_feat))
  H_t = np.zeros((len(ind), num_neigh+1, dim_feat+2)) # data with covariates
  Adj_t = np.zeros((len(ind), len(length_scale), 1, num_neigh+1))
  j=0
  for i in ind:
    ind_array_i = neigh_ind[i]
    first_col = np.zeros(num_neigh+1).reshape(-1,1)
    first_col[1:] = ytilde[ind_array_i]
    sec_col = np.zeros(num_neigh+1).reshape(-1,1)
    sec_col[0] = 1.

    # third_col = np.zeros((num_neigh+1, dim_feat-2))';
    # third_col[0] = feat[i, 2:].reshape(1, -1)  # only with covariates
    # third_col[1:] = feat[ind_array_i][:, 2:]   # only with covariates
    # H_t[j] = np.concatenate([first_col.reshape(-1,1), sec_col.reshape(-1,1), third_col], axis=1)

    third_col = np.zeros((num_neigh+1, dim_feat))
    third_col[0] = feat[i, :].reshape(1, -1)  # only with covariates
    third_col[1:] = feat[ind_array_i]#[:, :2]   # only with covariates
    H_t[j] = np.concatenate([first_col.reshape(-1,1), sec_col.reshape(-1,1), third_col], axis=1)
    # H_t[j] = np.concatenate([first_col.reshape(-1,1), sec_col.reshape(-1,1)], axis=1)

    #compute the normalized adjacency matrix per datapoint
    coords = np.concatenate([feat[i, 0:2].reshape(1, -1), feat[ind_array_i][:, 0:2]] , axis=0)
    for k in range(len(length_scale)):
      Adj_t[j, k] = adjacency_matrix(coords, length_scale[k])[0, :]
    j += 1
  return (Adj_t, H_t)


def evaluate(arange_nt, model, ind_neigh, features, y, length_scale):
    """ Evaluation function.

    Args:
      dataset: one of validation or test dataset
      model: model used

    Returns:
      avg_loss : loss value
    """

    x_batch = build_input_h(arange_nt, ind_neigh, features, y, length_scale)
    logits = model(x_batch, training=False)  # Logits for this minibatch
    avg_mse = mse_loss(logits, y[arange_nt])

    return logits, avg_mse.numpy()


def train_eval(datanum, dataname, num_valid, len_scale=0.5):
    """ Train and evaluate the model.

    Args:
      datanum: Data ID
      dataname: name of dataset
      num_valid: number of validation datapoints
      len_scale: length scale to use for the kernel

    Returns:
      model: trained model 
      test_mse: mse on the test set
      train_loss_list: Training losses at various epochs 
      best_val: The best mse on validation data 
    """

    # Prepare the training dataset.
    print('loading the input and preparing '+dataname + '...')
    time1=time.time()
    features, y_tilde, y_train_test, ind_neigh, Ntrain = load_kriging_data(dataname, FLAGS.num_neigh, num_valid, log_y=False)

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr)

    print('features', features.shape)
    print('y_train_test', y_train_test.shape)
    print('ind_neigh', ind_neigh.shape)

    time2=time.time()
    y_train_test = np.reshape(y_train_test, (-1, 1))
    y_tilde = np.reshape(y_tilde, (-1, 1))

    print('done with the dataset!' , time.time() - time1)

    hiddens = [FLAGS.hidden1, FLAGS.hidden2]
    model = KCN(hiddens, output_dim=y_train_test.shape[1])

    best_val = 1e21
    train_loss_list = []
    val_mse_list = []

    for epoch in range(FLAGS.epochs):
        ind_arr_train = np.arange(Ntrain - num_valid)

        # random.shuffle(ind_arr_train)
        t = time.time()

        # Initialize the metrics
        train_loss = 0.0
        val_mse = 0.0
        test_mse = 0.0

        num_steps = 0 
        # Iterate over the batches of the dataset.
        for i in range(0, len(ind_arr_train), FLAGS.batch_size):
            ind_array = ind_arr_train[i:i+FLAGS.batch_size]
            loss_value = 0.0
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                x_batch_train = build_input_h(ind_array, ind_neigh, features, y_tilde, len_scale)


                logits = model(x_batch_train, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = mse_loss(y_tilde[ind_array], logits)
                # Weight decay loss
                for var in model.trainable_weights:
                    loss_value += FLAGS.weight_decay * tf.nn.l2_loss(var)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss +=loss_value
            num_steps +=1
        
        train_loss = train_loss/num_steps
        train_loss_list.append(train_loss)

        # Evaluate on the validation set
        arange_ev = np.arange(Ntrain-num_valid, Ntrain)
        # print('array of validation indices', arange_ev)
        # val_mse = 1.
        _, val_mse = evaluate(arange_ev, model, ind_neigh, features, y_train_test, len_scale)
        val_mse_list.append(val_mse)

        # # save weigths at this stage
        # if val_mse<best_val:
        #   best_val = val_mse
        #   model.save_weights("KCN_experiments/checkpoints.ckpt")
        # # Print results

        if (epoch + 1)%25==0.:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
                  "val_mse=", "{:.5f}".format(val_mse), "time=", "{:.5f}".format(time.time() - t))
        
        #if not np.logical_or(np.isnan(val_mse), np.isnan(train_loss)):# and val_mse<best_val:
        #    model.save_weights("KCN_experiments_vs_corr/checkpoints"+str(datanum)+".ckpt")
        #    #best_val = val_mse
        #    # continue
        #else:
        #   print("Nan values arose..., restoring previous weights")
        #   model.load_weights("KCN_experiments_vs_corr/checkpoints"+str(datanum)+".ckpt")
        #   break
        # if epoch > FLAGS.es_patience and np.mean(val_mse_list[-3:]) > np.mean(
        #         val_mse_list[-(FLAGS.es_patience + 3):-3]):
        #     print("Early stopping...")
        #     break

        last_valid_error = np.mean(val_mse_list[-4:])

    print("Optimization Finished!")

    t1 = time.time()
    best_val = val_mse_list[-1]
    # Restore the weights
    # model.load_weights("KCN_experiments/checkpoints.ckpt")
    arange_te = np.arange(Ntrain, len(y_train_test))
    # print('array of test indices', arange_te)
    logits_test, test_mse = evaluate(arange_te, model, ind_neigh, features, y_train_test, len_scale)
    # %rm -rf KCN_experiments
    exp_time = time.time() - t1
    print("test_mse=", "{:.5f}".format(test_mse), "time=", "{:.5f}".format(exp_time))
    return model, test_mse, train_loss_list, best_val, len_scale, logits_test, exp_time


def data_result(jj):
    """ Iterate over various length scales for each dataset.

    Args:
      jj: dataset id

    Returns:
      best_test_mse: Test mse obtained with the best length scale
      best_n: the best length scale
    """
    print('Dataset number ', jj)

    best_val_list = []
    last_tr_list = []
    mse_list = []
    div_list = []
    list_ns = []
    best_val1 = 1e12
    best_val2 = 1e12

    ns=3
    ls_list = [10**ns, 10**(ns-1), 10**(ns-2), 10**(ns-3)]
    model, test_mse, tr_loss_list, best_val, _, _, _ = train_eval(jj, 'content/100_datamse_sg/smalldatagauss_'+str(jj)+'_mse.npz',num_valid=7, len_scale=ls_list, len_scale_list=[], num_cv=0, cross_val=False)
    list_ns.append(ns)
    last_tr_list.append(tr_loss_list[-1].numpy())
    best_val_list.append(best_val)
    mse_list.append(test_mse)

    while(len(best_val_list)<9 and best_val2==1e12):
        ls_list.remove(ls_list[0])
        ls_list.append(ls_list[-1]*1e-1)
        print('list of knots: ', ls_list)
        model, test_mse, tr_loss_list, best_val, _, _, _ = train_eval(jj, 'content/100_datamse_sg/smalldatagauss_'+str(jj)+'_mse.npz', num_valid=7, len_scale=ls_list, len_scale_list=[], num_cv=0, cross_val=False)
        list_ns.append(ns-1)
        best_val_list.append(best_val)
        last_tr_list.append(tr_loss_list[-1].numpy())
        mse_list.append(test_mse)
        if len(best_val_list)>3:
            if (best_val1 == 1e12 and best_val_list[-3]<np.mean(best_val_list[-3:])):
                best_val1 = best_val_list[-3]
            else:
                if(best_val_list[-3]<best_val1 and best_val_list[-3]<=best_val_list[-4] and best_val_list[-3]<=best_val_list[-2]):
                      best_val2 = best_val_list[-3]

    ind_nan = clean_nan(mse_list)
    if len(ind_nan)!=0:
        best_val_list = [k for j, k in enumerate(best_val_list) if j not in ind_nan]
        last_tr_list = [k for j, k in enumerate(last_tr_list) if j not in ind_nan]
        mse_list = [k for j, k in enumerate(mse_list) if j not in ind_nan]

    id_val = np.argmin(best_val_list)
    best_n = list_ns[id_val]
    best_test_mse = mse_list[id_val]
    best_ranges = [10**(best_n), 10**(best_n-1), 10**(best_n-2), 10**(best_n-3)]

    with open('mult_kcn_sg_mse_3.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([jj, best_n, best_test_mse])


    return best_test_mse, best_n, len(ind_nan)




if __name__ == '__main__':
    
    test_mse, ls, num_nans= data_result(jj=sys.argv[1])
    print(num_nans)
