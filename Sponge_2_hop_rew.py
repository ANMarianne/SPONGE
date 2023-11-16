#!/usr/bin/env python
# coding: utf-8


# fix random seed
rand_seed = 1234
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(rand_seed)# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(rand_seed)# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(rand_seed)# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(rand_seed)
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
import sklearn
import scipy
import csv
import time

tf.compat.v1.flags.DEFINE_string('f','','')



#Experiments parameters
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_neigh', 5, 'Number of neighbors')
flags.DEFINE_integer('val_num', 50, 'Number of validation datapoints')
flags.DEFINE_integer('data_id', 1, 'id of the dataset to call')
flags.DEFINE_integer('hidden1', 5, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2. -1 means not to use this layer')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (keep probability) 0, 0.25, 0.5')
flags.DEFINE_float('lr', 1e-2, 'Learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('es_patience', 10, 'Patience for early stopping')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('dataset', 'databiggauss.npz', 'Data file path  0.17657')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('length_scale', 0.5, 'Length scale of the RBF kernel. 1, .5, .1, .05')


num_neigh = 5



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
      y_train_test_mean: mean of training targets
      y_train_test_std: Standard deviation of training targets
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
    features = (features - fmean) / (fstd)

    y_train_test = np.concatenate([Y_train, Y_test], axis=0)

    y_train_test_mean = np.mean(y_train_test[0:Ntrain], axis=0, keepdims=True)
    y_train_test_std = np.std(y_train_test[0:Ntrain], axis=0, keepdims=True)
    y_train_test = (y_train_test - y_train_test_mean) / (y_train_test_std)


    # take the log of y coordinates if necessary
    if log_y:
        y_train_test = np.log(y_train_test)

    y_tilde = y_train_test.copy()
    y_tilde[Ntrain-num_val:] = 0

    # getting neighbors from training data and test data. search one more because the first is the pt itself
    knn = NearestNeighbors(n_neighbors=num_neigh).fit(coords[0:Ntrain])
    ind_nbs_train_val = knn.kneighbors(return_distance=False)
    ind_nbs_test = knn.kneighbors(coords[Ntrain:], return_distance=False)

    ind_nbs = np.concatenate([ind_nbs_train_val, ind_nbs_test], axis=0)

    return features, y_tilde, y_train_test, ind_nbs, Ntrain, y_train_test_mean , y_train_test_std


def gauss_numb(coord1, coord2, length_scale):
    """Compute the correlation number from the coordinates.

    Args:
        coord1: coords of the first location
        coord2: coords of the second location
        length_scale: length scale of the kernel

    Returns:
         numb : correlation number between coord1 and coord2
    """

    numb =np.exp(-((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2)/(2*length_scale*length_scale))
    return numb

def inv_numb(coord1, coord2, length_scale):
    """Compute the correlation number from the coordinates.

    Args:
        coord1: coords of the first location
        coord2: coords of the second location
        length_scale: length scale of the kernel

    Returns:
        numb : correlation number between coord1 and coord2
    """
    dist = (coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2
    if dist==0.:
        numb=1.
    else:
        numb =1/((dist)**length_scale)
    return numb


def expo_numb(coord1, coord2, length_scale):
    """Compute the correlation number from the coordinates.

    Args:
      coord1: coords of the first location
      coord2: coords of the second location
      length_scale: length scale of the kernel

    Returns:
      numb : correlation number between coord1 and coord2
    """

    numb = np.exp(-((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2)**0.5/(length_scale))
    return numb


# helping function to build the input of the training/test pipeline
def adj_input_i(i, neigh_ind, feat, ytilde, length_scale, expo=False):
    """Compute the input matrix H and the normalize adjacency matrix for the input.

    Args:
        i: index of the center point for which we want a prediction
        neigh_ind: indices of neighbors for each points
        Xtrain: train input
        ytilde: targets
        length_scale: length scale value

    Returns:
        adj_t : adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
        H_t : Input to the pipeline 
    """

    _, dim_feat = feat.shape
    _, num_neigh = neigh_ind.shape 

    if expo==True:
        adj_ = expo_numb
    else:
        adj_ = gauss_numb

    coords = feat[:, 0:2]

    ind_list = []
    number_neigh = 1+num_neigh+num_neigh*num_neigh
    b_i = [1.*np.eye(num_neigh+1, number_neigh) for k in range(len(length_scale))]
    neigh_i = neigh_ind[i] 
    ind_list.append(i)
    neigh_j = neigh_i.copy()

    # collect the neighb
    [ind_list.append(i1) for i1 in neigh_i]
    # j = 0
    for p in range(num_neigh+1):

        #rescale with x
        pairw_dist_all = sklearn.metrics.pairwise_distances(neigh_j.reshape(-1, 1))
        pairw_dist = np.triu(pairw_dist_all, k=1)
        arr_pairw_dist = pairw_dist[np.triu_indices(num_neigh, k=1)]
        med_here = np.median(arr_pairw_dist)
        ind_here = np.where(np.logical_and(pairw_dist>0., pairw_dist<=0.2*med_here))
        if len(ind_here[1])==0:
            ind_unique = [0]
        else:
            ind_unique  = np.unique(ind_here[1]) +1

        for q in range(1, num_neigh+1):
            for k in range(len(length_scale)):
                if p==0:
                    b_i[k][p,num_neigh*p+q] = adj_(coords[ind_list[p],:], coords[ind_list[num_neigh*p+q],:], length_scale[k])
                else:
                    b_i[k][p,num_neigh*p+q] = adj_(coords[ind_list[p],:], coords[ind_list[num_neigh*p+q],:], length_scale[k])*np.exp(-np.abs(ytilde[ind_list[p]]-ytilde[ind_list[num_neigh*p+q]]))
                if q in ind_unique:
                    small_dist = np.min(pairw_dist[ind_here[0], q-1])
                    b_i[k][p,num_neigh*p+q] *= (expit(small_dist)-1)/(expit(small_dist)+1)

        if p < num_neigh:
            neigh_j = neigh_ind[neigh_i[p]] 
            [ind_list.append(j1) for j1 in neigh_j]
    Adj_i = np.stack(b_i, axis=0)   

    first_col = ytilde[ind_list]
    first_col[0] = 0.
    second_col = np.zeros_like(first_col)
    second_col[0] = 1
    # third_col = feat[ind_list]#[:, :2]
    H_i = np.concatenate((first_col, second_col), axis=1)
    # H_i = np.concatenate((first_col, second_col, third_col), axis=1)

    return (Adj_i, H_i)


# Build the input of the training/test pipeline
def build_input_h(ind, neigh_ind, feat, ytilde, length_scale, expo=False):
    """Compute the input matrix H and the normalize adjacency matrix for the input.

    Args:
        i: index of the center point for which we want a prediction
        neigh_ind: indices of neighbors for each points
        Xtrain: train input
        ytilde: targets
        length_scale: length scale value

    Returns:
        adj_t : adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
        H_t : Input to the pipeline 
    """

    Adj_t = []
    H_t = []

    for i in ind:
        Adj_i, H_i = adj_input_i(i, neigh_ind, feat, ytilde, length_scale, expo=expo)
        Adj_t.append(Adj_i)
        H_t.append(H_i)
    Adj_t = np.stack(Adj_t, axis=0)
    H_t = np.stack(H_t, axis=0)
    return (Adj_t, H_t)


def clean_nan(lists):
    nan_ind=list(np.argwhere(np.isnan(lists)).flatten())
    return nan_ind


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
        """ calling the graph conv layer builds it to create weights"""

        tf.random.set_seed(rand_seed)
        if len(input_shape)==2:
            self.graphconvNeig = self.add_weight("GConvNeigh",
                                      shape=[input_shape[1][-1],
                                         self.num_outputs])
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
            # Adj = Adj[:,:,:,0]      #uncomment for one ls
            result0 = Adj @ H
            output = result0 @ self.graphconvNeig
            Adj_new = tf.reshape(Adj[:, 0, 0:num_neigh +1 ], [Adj.shape[0],1,num_neigh +1]) 
        else:
            output =  inputs @ self.graphconvNeig
        return Adj_new, output


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
        if len(input_shape)==2:
            self.graphconv = self.add_weight("GConv",
                                        shape=[input_shape[1][-1],
                                               self.num_outputs])
        else:
            self.graphconv = self.add_weight("GConv",
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
                result0 = inputs[0] @ inputs[1]
                output = result0 @ self.graphconv
            else:
                output =  inputs @ self.graphconv 
            return output


# Build a graph convolutional layer
class MyAdjconvLayer(layers.Layer):
    def __init__(self):
        """ Instatiating the graph convolutional layer.

        Args:
          hidden_out: output of the dense
        """
        super(MyAdjconvLayer, self).__init__()

    def build(self, input_shape):
        """ calling the graph conv layer builds it to create weights"""

        tf.random.set_seed(rand_seed)
        if len(input_shape)==2:
            self.adjconv = self.add_weight("AConv",
                                        shape=[input_shape[0][1],
                                               1],trainable=True, constraint=MyConstraints())
        else:
            self.adjconv = self.add_weight("AConv",
                                      shape=[input_shape[1],
                                             1],trainable=True)

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
        self.hidden2 = MyGraphconvLayer(hidden_sizes[1])

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

        x = self.hidden0(input_tensor)          # comment for one ls
        A, x = self.hidden1(x)                  # comment for one ls
        # A, x = self.hidden1(input_tensor)     # uncomment for one ls
        x = tf.nn.relu(x)

        x = self.hidden2((A,x))
        x = tf.nn.relu(x)

        if self.kcn_bool == True:
            x = self.dense(x) #before
            x = x[:, 0, :] #before
        return x


# Defining the metrics

def mse_loss(logits, labels):
    """ Computes the mse loss of a batch.

    Args:
      logits: Model output of size [batch, 1]
      labels: True labels of the batch [batch, 1]
    Returns:
      loss: Float 32
    """

    loss = tf.reduce_mean((logits-labels)**2)   # mse loss
    # loss = tf.reduce_mean(tf.abs(logits-labels))   # mae loss
    return loss



def evaluate(arange_nt, model, ind_neigh, features, y, length_scale, expo, ymean, ystd, val=False):
    """ Evaluation function.

    Args:
      dataset: one of validation or test dataset
      model: model trained
    Returns:
      avg_loss : loss value
    """

    xtot = build_input_h(arange_nt, ind_neigh, features, y, length_scale, expo=expo)
    logits = model(xtot, training=False)
    avg_mse = mse_loss(logits*ystd+ymean, y[arange_nt]*ystd+ymean)
    # avg_mse = tf.reduce_mean(tf.abs(logits-y[arange_nt]))
    if val:
        return avg_mse.numpy()
    else:
        return avg_mse.numpy(), logits*ystd+ymean


# In[11]:


def train_eval(datanum, dataname, num_valid, len_scale=[0.5], expo=False):
    """ Train and evaluate the model.

    Args:
      datanum: Data ID
      dataname: name of dataset
      num_valid: number of validation datapoints
      len_scale: length scale to use for the kernel
      expo: Whether to use exponential kernel

    Returns:
      model: trained model 
      test_mse: mse on the test set
      train_loss_list: Training losses at various epochs 
      best_val: The best mse on validation data 
    """

    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr)

    # Prepare the training dataset.
    print('loading the input and preparing '+dataname + '...')
    time1=time.time()

    features, y_tilde, y_train_test, ind_neigh, Ntrain, ymean, ystd = load_kriging_data(dataname, num_neigh, num_valid, log_y=False)

    print('features', features.shape)
    print('y_train_test', y_train_test.shape)
    print('ind_neigh', ind_neigh.shape)

    time2=time.time()
    y_train_test = np.reshape(y_train_test, (-1, 1))
    y_tilde = np.reshape(y_tilde, (-1, 1))

    print('done with the dataset!' , time.time() - time1)

    hiddens = [5, 10]
    model = KCN(hiddens, output_dim=y_train_test.shape[1])

    ind_arr_train = np.arange(Ntrain - num_valid)
    #np.random.seed(rand_seed) 
    #random.shuffle(ind_arr_train)

    best_val = 1e21

    train_loss_list = []
    val_mse_list = []
    out = 0

    for epoch in range(FLAGS.epochs):
        t = time.time()

        # Initialize the metrics
        train_loss = 0.0
        val_mse = 0.0
        test_mse = 0.0

        num_steps = 0 
        # Iterate over the batches of the dataset.
        for i in range(0, len(ind_arr_train), FLAGS.batch_size):
            # print('inside', i)
            ind_array = ind_arr_train[i:i+FLAGS.batch_size]
            # ind_array = ind_arr_train[i:i+2]
            loss_value = 0.0
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                x_batch_train = build_input_h(ind_array, ind_neigh, features, y_tilde, len_scale, expo=expo)


                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = mse_loss(y_tilde[ind_array]*ystd+ymean, logits*ystd+ymean)
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
        val_mse = evaluate(arange_ev, model, ind_neigh, features, y_train_test, len_scale, expo, ymean, ystd, val=True)
        val_mse_list.append(val_mse)

        # # save weigths at this stage
        # if val_mse<best_val:
        #   best_val = val_mse
        #   model.save_weights("KCN_neigh_2/checkpoints.ckpt")
        # Print results
        if (epoch + 1)%25==0.:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{}".format(train_loss),
                  "val_mse=", "{}".format(val_mse), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.es_patience and np.mean(val_mse_list[-3:]) > np.mean(
        #         val_mse_list[-(FLAGS.es_patience + 3):-3]):
        #     print("Early stopping...")
        #     break

        if not np.logical_or(np.isnan(val_mse), np.isnan(train_loss)):
            model.save_weights("KCN_neigh_2_xy/checkpoints"+str(datanum)+".ckpt")
          # continue
        else:
            if epoch == 0:
                break
            else:
                print("Nan values arose..., restoring previous weights")
                model.load_weights("KCN_neigh_2_xy/checkpoints"+str(datanum)+".ckpt")
                break

        # last_valid_error = np.mean(val_mse_list[-4:])

    print("Optimization Finished!")

    t1 = time.time()
    # Restore the weights
    # model.load_weights("KCN_neigh_2/checkpoints.ckpt")

    best_val = val_mse_list[-1]
    arange_te = np.arange(Ntrain, len(y_train_test))
    if epoch==0:
        print('training unsuccessful')
        test_mse = logits_test = np.nan
    else:
        test_mse, logits_test = evaluate(arange_te, model, ind_neigh, features, y_train_test, len_scale, expo, ymean, ystd)
    print(test_mse)
    # %rm -rf KCN_neigh_2
    print("test_mse =", "{}".format(test_mse), "time =", "{:.5f}".format(time.time() - t1))
    return model, test_mse, train_loss_list, val_mse_list, best_val, logits_test



def dataxy_results(i):
    """ Iterate over various length scales for each dataset.

    Args:
      i: dataset id

    Returns:
      best_test_mse: Test mse obtained with the best length scale
      best_n: the best length scale
    """
    print('Dataset number ', i)

    best_val_list = []
    last_tr_list = []
    mse_list = []
    div_list = []
    list_ns = []
    best_val1 = 1e12
    best_val2 = 1e12

    # ls_list = [1e2, 1e1, 1e0, 1e-1]

    data = np.load('content/100_datamse_sg/smalldatagauss_'+str(i)+'_mse.npz')

    # Collecting the train and the test set
    X_train = np.ndarray.astype(data['Xtrain'], np.float32)
    Y_train = np.ndarray.astype(data['Ytrain'], np.float32)
    iqr_xtr = scipy.stats.iqr(sklearn.metrics.pairwise_distances(X_train[0:2]))
    iqr_ytr = scipy.stats.iqr(Y_train)

    ns = int(np.round(iqr_ytr*iqr_xtr))+1
    ls_list = [10**ns, 10**(ns-1), 10**(ns-2), 10**(ns-3)]
    model_egn, test_mse, tr_loss_list, val_loss_list, best_val, _ = train_eval(i, 'content/100_datamse_mg/mediumdatagauss_'+str(i)+'_mse.npz', 25, len_scale=ls_list, expo=False)
    list_ns.append(ns)
    last_tr_list.append(tr_loss_list[-1].numpy())
    best_val_list.append(best_val)
    mse_list.append(test_mse)

    while(len(best_val_list)<6 and best_val2==1e12):
#     while len(best_val_list)<7 :
        ls_list.remove(ls_list[0])
        ls_list.append(ls_list[-1]*1e-1)
        print('list of knots: ', ls_list)
        model_egn, test_mse, tr_loss_list, val_loss_list, best_val, _ = train_eval(i, 'content/100_datamse_mg/mediumdatagauss_'+str(i)+'_mse.npz', 25, len_scale=ls_list, expo=False)
        list_ns.append(ns-1)
        best_val_list.append(best_val)
        last_tr_list.append(tr_loss_list[-1].numpy())
        mse_list.append(test_mse)
        if len(best_val_list)>3:
            if (best_val1 == 1e12 and best_val_list[-3]<np.mean(best_val_list[-3:])):
                best_val1 = best_val_list[-3]
            else:
                if(best_val_list[-3]<best_val1 and best_val_list[-3]<=best_val_list[-4] and best_val_list[-3]\<=best_val_list[-2]):
                      best_val2 = best_val_list[-3]
        # continue

    ind_nan = clean_nan(mse_list)
    if len(ind_nan)!=0:
        best_val_list = [k for j, k in enumerate(best_val_list) if j not in ind_nan]
        last_tr_list = [k for j, k in enumerate(last_tr_list) if j not in ind_nan]
        mse_list = [k for j, k in enumerate(mse_list) if j not in ind_nan]

    id_val = np.argmin(best_val_list)
    best_test_mse =  mse_list[id_val]
    best_n = list_ns[id_val]

    
    with open('kcn_neigh2xy_sg_mse.csv', 'a+', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Data no', 'length scale', 'test mse'])
        writer.writerow([i, best_n, best_test_mse])

    return best_test_mse, best_n


if __name__ == '__main__':
    
    test_mse, ls = dataxy_results(i=sys.argv[1])

