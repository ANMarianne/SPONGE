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
from sklearn.neighbors import NearestNeighbors
import sklearn
from patsy import dmatrix
from patsy import build_design_matrices
from math import factorial
import time
import csv
import sys



tf.compat.v1.flags.DEFINE_string('f','','')


#Experiments parameters
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_neigh', 5, 'Number of neighbors')
flags.DEFINE_integer('val_num', 50, 'Number of validation datapoints')
flags.DEFINE_integer('hidden1', 5, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2. -1 means not to use this layer')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (keep probability) 0, 0.25, 0.5')
flags.DEFINE_float('lr', 1e-2, 'Learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('es_patience', 10, 'Patience for early stopping')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('dataset', 'databiggauss.npz', 'Data file path  0.17657')
# flags.DEFINE_string('dataset', '/home/mnjifon/Documents/Thesis/Dominic Litterature/aaai20-kcn-data/data/precip.npz', 'Data file path  0.17657')
# flags.DEFINE_string('dataset', '/home/mnjifon/Documents/Thesis/Dominic Litterature/aaai20-kcn-data/data/birds.npz', 'Data file path 0.48596')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('length_scale', 0.5, 'Length scale of the RBF kernel. 1, .5, .1, .05')


num_neigh = 5


def comp_dist(datas):
    shapes = datas.shape
    lenghts = max(shapes[0], shapes[1])
    max_dist = 0
    for i in range(lenghts-1):
        for j in range(i+1, lenghts):
            dist = (np.sum((datas[i,:] - datas[j,:])**2))**(0.5)
            if dist > max_dist:
              max_dist = dist
    return max_dist


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

    # feature normalization as sugested in the original code of the authors
    fmean = np.mean(coords_features[0:Ntrain], axis=0, keepdims=True)
    fstd = np.std(coords_features[0:Ntrain], axis=0, keepdims=True)
    coords_features = (coords_features - fmean) / (fstd + 0.01)

    y_train_test = np.concatenate([Y_train, Y_test], axis=0)

    # take the log of y coordinates if necessary
    if log_y:
        y_train_test = np.log(y_train_test)

    y_tilde = y_train_test.copy()
    y_tilde[Ntrain-num_val:] = 0

    coords = coords_features[:, 0:2]
    features = coords_features

    # Maximum distance between samples
    dists = sklearn.metrics.pairwise_distances(coords, metric='euclidean')
    # max_dist = comp_dist(coords_features[:, 0:2])
    
    max_dist = np.max(dists)
    print('maximum distance', max_dist)

    # getting neighbors from training data and test data.
    knn = NearestNeighbors(n_neighbors=num_neigh).fit(coords[0:Ntrain])
    ind_nbs_train_val = knn.kneighbors(return_distance=False)
    ind_nbs_test = knn.kneighbors(coords[Ntrain:], return_distance=False)
    ind_nbs = np.concatenate([ind_nbs_train_val, ind_nbs_test], axis=0)

    return features, y_tilde, y_train_test, ind_nbs, Ntrain, max_dist


def adj_input_i(i, neigh_ind, feat, ytilde, knots_r, max_dist, lapl_b=False):
    """Compute the input matrix H and the normalize adjacency matrix for the input.

    Args:
        i: index of the center point for which we want a prediction
        neigh_ind: indices of neighbors for each points
        feat: train input
        ytilde: targets
        knots_r: 
        max_dist:
        lapl_b: 

    Returns:
        adj_t : adjacency matrix (shape [num_data, num_neigh+1, num_neigh+1])
        H_t : Input to the pipeline 
    """

    _, dim_feat = feat.shape
    _, num_neigh = neigh_ind.shape

    knots =  tuple(i*max_dist for i in knots_r)

    coords = feat[:, 0:2]

    ind_list = []
    computed_values = []

    number_neigh = 1+num_neigh+num_neigh*num_neigh
    depth = len(knots)+3+1
    # depth = 9
    b_i = np.zeros((num_neigh+1, number_neigh, depth))
    neigh_i = neigh_ind[i]

    # append index of the actual pt interest as the first in ind_list
    ind_list.append(i)

    # append the next 5 indices of the closest neighbor to the list
    [ind_list.append(i1) for i1 in neigh_i]

    # Distances accumulated
    dist_list = []
    index_list_x = []
    index_list_y = []
    # index_list_z0 = []
    index_list_z1 = []
    index_list_z2 = []
    index_list_z3 = []
    # index_list_z4 = []

    #loop through the neighboors of the point of interest
    for p in range(num_neigh+1):  

        # distance between a point and itself
        coords_here = np.concatenate([(coords[ind_list[p],:]).reshape(1,-1), (coords[ind_list[p],:]).reshape(1,-1)], axis=0)
        # l2_dist = sklearn.metrics.pairwise_distances(coords_here, metric='euclidean')[0,1]
        # dist_list.append(l2_dist)
        dist_list.append(coords_here)
        index_list_x.append(p)
        index_list_y.append(p)
        # index_list_z0.append(p)
        index_list_z1.append(p*(num_neigh+1))

        # select each index of the neighborhood list
        for q in range(1, num_neigh+1):  
            coords_herepq =  np.concatenate([(coords[ind_list[p],:]).reshape(1,-1), (coords[ind_list[num_neigh*p+q],:]).reshape(1,-1)], axis=0)
            # l2_distpq = sklearn.metrics.pairwise_distances(coords_herepq, metric='euclidean')[0,1]
            # dist_list.append(l2_distpq)
            dist_list.append(coords_herepq)
            index_list_x.append(p)
            index_list_y.append(num_neigh*p+q)
            index_list_z3.append(p)
            # index_list_z4.append(num_neigh*p+q)
            index_list_z2.append(num_neigh*p+q)

        if p < num_neigh:
            neigh_j = neigh_ind[neigh_i[p]]
            [ind_list.append(j1) for j1 in neigh_j]
  
    # computed_values = build_design_matrices(desmat_inf, {"train": dist_list}, return_type='dataframe')[0].values 
    dist_list = np.concatenate(dist_list, axis=0) 
    l2_dist_mat = sklearn.metrics.pairwise_distances(dist_list, metric='euclidean')
    l2_dist = l2_dist_mat[np.array([index_list_x]), np.array([index_list_y])].reshape(-1,)
    computed_values = dmatrix("bs(train, knots=knots, degree=3, include_intercept=True,lower_bound=0., upper_bound=max_dist)-1", {"train": l2_dist})
  

    # b_i[np.array([np.unique(index_list_x)]), np.array([np.unique(index_list_x)])] = computed_values[index_list_z1]
    # b_i[np.array([index_list_z3]), np.array([np.unique(index_list_z2)])] = computed_values[index_list_z2]

    for p in range(num_neigh+1): 
        b_i[p,p,:] = 1. # computed_values[p*(num_neigh+1)] #1. #0.
        for q in range(1, num_neigh+1):
            b_i[p,num_neigh*p+q,:] = computed_values[num_neigh*p+q]

    # normalized before
    # Adj_i = b_i/((np.sum(b_i, axis=2)+1e-6).reshape(-1,number_neigh,1))

    Adj_i = b_i
    # print('Adj_i', Adj_i[:,:,2])
    tf.debugging.check_numerics(Adj_i, 'Adj_i has nan values')
    # H_i = ytilde[ind_list]
    # H_i[0] = 0
    first_col = ytilde[ind_list]
    first_col[0] = 0.
    second_col = np.zeros_like(first_col)
    second_col[0] = 1
    third_col = feat[ind_list]#[:, 2:]
    H_i = np.concatenate((first_col, second_col, third_col), axis=1)
    # H_i = np.concatenate((first_col, second_col), axis=1)


    return (Adj_i, H_i)


# Build the input of the training/test pipeline
def build_input_h(ind, neigh_ind, feat, ytilde, knots, max_dist, lapl_b=False):
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

    Adj_t = []
    H_t = []

    for i in ind:
        Adj_i, H_i = adj_input_i(i, neigh_ind, feat, ytilde, knots, max_dist, lapl_b=lapl_b)
        Adj_t.append(Adj_i)
        H_t.append(H_i)
    Adj_t = np.stack(Adj_t, axis=0)
    H_t = np.stack(H_t, axis=0)

    return (Adj_t, H_t)


class MyConstraints(tf.keras.constraints.Constraint):

    def __call__(self, w):
        w = w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
        return w #/tf.reduce_sum(w)


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
                                                        self.num_outputs])#, dtype=tf.float64)
            tf.debugging.check_numerics(self.graphconvNeig, 'graph conv first neigh has nan values')
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
            result0 = Adj @ H
            output = result0 @ self.graphconvNeig
            Adj_new = tf.reshape(Adj[:, 0, 0:FLAGS.num_neigh +1], [Adj.shape[0],1,FLAGS.num_neigh +1])
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
        """ calling the graph conv layer builds it to create weights"""

        tf.random.set_seed(rand_seed)
        if len(input_shape)==2:
            self.graphconv = self.add_weight("GConv",
                                              shape=[input_shape[1][-1],
                                                      self.num_outputs])
            tf.debugging.check_numerics(self.graphconv, 'graph conv second neigh has nan values')

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
                                           shape=[input_shape[0][-1],
                                                  1],trainable=True,constraint=MyConstraints())#,regularizer='l2')
            tf.debugging.check_numerics(self.adjconv, 'adj convolution has nan values')
        else:
            self.adjconv = self.add_weight("AConv",
                                           shape=[input_shape[1],
                                                  1],trainable=True)#,constraint=MyConstraints()) 

    def call(self, inputs):
        """ Executed channel wise convolution when apply the layer to inputs. 

        Args:
            inputs: tuple of input (3d of shape [batch, K+1, hidden]) and normalised adj
        Returns:
            output : tuple of output(3d of shape [batch, K+1, hidden_out]) and normalised adj
        """

        adjc = self.adjconv/tf.reduce_sum(self.adjconv)
        if len(inputs)==2:
            adj, H = inputs
            batch, n1, n2, _ = adj.shape
            D1 = tf.reshape(tf.tensordot(adj, adjc, axes=[[3],[0]]), [batch, n1, n2])
            tf.debugging.check_numerics(D1, 'unnormalized adj has nan values')
            D2 = tf.cast(tf.reshape(tf.reduce_sum(D1, axis=2), [-1,n1,1]), tf.float32)
            output = D1/D2
            tf.debugging.check_numerics(output, 'normalized adj has nan values')
        else:
            output =  inputs @ adjc 
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


        self.hidden0 = MyAdjconvLayer()
        self.hidden1 = MyGraphconvNeigh(hidden_sizes[0])
        self.hidden2 = MyGraphconvLayer(hidden_sizes[1])

        if self.kcn_bool == True:
            tf.random.set_seed(rand_seed) 
            self.dense = tf.keras.layers.Dense(output_dim, use_bias=False)
                                               #kernel_initializer=tf.keras.initializers.Ones())#tf.keras.initializers.GlorotUniform(seed=rand_seed))


    def call(self, input_tensor, training=False):
        """ Executed when apply the layer to inputs.

        Args:
            input_tensor: output of the GCN, 3d tensor of shape [batch, K+1, hidden_prev]
            training: boolean specifying the mode
        Returns:
            output : 3d tensor of shape [batch, K+1, output_dense] if training=true, otherwise[K+1, out]
        """

        x = self.hidden0(input_tensor)
        A, x = self.hidden1(x)
        x = tf.nn.relu(x)

        x = self.hidden2((A,x))
        x = tf.nn.relu(x)

        if self.kcn_bool == True:
            x = self.dense(x)
            tf.debugging.check_numerics(x , 'if nothing happened before, then maybe the pb is the dense')
            x = x[:, 0, :]

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

    loss = tf.reduce_mean((logits-labels)**2)
    return loss


def evaluate(arange_nt, model, ind_neigh, features, y, knots, max_dist, lapl):
    """ Evaluation function.

    Args:
        dataset: one of validation or test dataset
        model: model used
    Returns:
        avg_loss : Loss value
    """

    xtot = build_input_h(arange_nt, ind_neigh, features, y, knots, max_dist, lapl_b=lapl)
    logits = model(xtot, training=False)
    avg_mse = mse_loss(logits, y[arange_nt])
    return avg_mse.numpy()


def clean_nan(lists):
    nan_ind=list(np.argwhere(np.isnan(lists)).flatten())
    return nan_ind


def train_eval(dataname, num_valid, knots=[], lapl=False):

    # Prepare the training dataset.
    print('loading the input and preparing '+dataname + '...')
    time1=time.time()

    features, y_tilde, y_train_test, ind_neigh, Ntrain, max_dist = load_kriging_data(dataname, FLAGS.num_neigh, num_valid, log_y=False)

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

    optimizer = keras.optimizers.Adam(learning_rate=FLAGS.lr)

    best_val = 1e21

    train_loss_list = []
    val_mse_list = []
    out = 0
  
    # for epoch in range(300):
    for epoch in range(FLAGS.epochs):
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
                x_batch_train = build_input_h(ind_array, ind_neigh, features, y_tilde, knots, max_dist, lapl_b=lapl)


                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = mse_loss(y_tilde[ind_array], logits)
                # Weight decay loss
                # for var in model.trainable_weights:
                #     loss_value += FLAGS.weight_decay * tf.nn.l2_loss(var)
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
        val_mse = evaluate(arange_ev, model, ind_neigh, features, y_train_test, knots, max_dist, lapl)
        val_mse_list.append(val_mse)

        # # # save weigths at this stage
        # if val_mse<best_val:
        #   best_val = val_mse
        #   model.save_weights("KCN_neigh_2/checkpoints.ckpt")

        # Print results
        if (epoch + 1)%25==0.:
          print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
                "val_mse=", "{:.9f}".format(val_mse), "time=", "{:.5f}".format(time.time() - t))

        # if epoch > FLAGS.es_patience and np.mean(val_mse_list[-3:]) > np.mean(
        #         val_mse_list[-(FLAGS.es_patience + 3):-3]):
        #     print("Early stopping...")
        #     break

        last_valid_error = np.mean(val_mse_list[-4:])

    print("Optimization Finished!")

    t1 = time.time()

    # Restore the weights
    # model.load_weights("KCN_neigh_2/checkpoints.ckpt")
    best_val = val_mse_list[-1]
    arange_te = np.arange(Ntrain, len(y_train_test))
    test_mse = evaluate(arange_te, model, ind_neigh, features, y_train_test, knots, max_dist, lapl)
    # %rm -rf KCN_neigh_2
    print("test_mse=", "{}".format(test_mse), "time=", "{:.5f}".format(time.time() - t1))
    print('maximum distances between samples', max_dist)
    return model, test_mse, train_loss_list, val_mse_list, best_val


def data_results(i):
    """ Iterate over various length scales for each dataset.

    Args:
      jj: dataset id

    Returns:
      best_test_mse: Test mse obtained with the best length scale
      best_n: the best length scale
    """

    print('Dataset number ', i)

    best_val_list = []
    mse_list = []
    last_tr_list = []
    best_val1 = 1e12
    best_val2 = 1e12

    knots_list = [1e-1]
    _, test_mse , tr_loss_list, _, best_val = train_eval('content/100_data_vsg_cov_mse/verysmalldatacov_mse'+str(i)+'.npz', 3, knots=knots_list)
    best_val_list.append(best_val)
    mse_list.append(test_mse)
    last_tr_list.append(tr_loss_list[-1].numpy())

    while(len(best_val_list)<10 and best_val2==1e12):
        knots_list.append(knots_list[-1]*1e-1)
        print('list of knots: ', knots_list)
        _, test_mse , tr_loss_list, _, best_val = train_eval('content/100_data_vsg_cov_mse/verysmalldatacov_mse'+str(i)+'.npz', 3, knots=knots_list)
        best_val_list.append(best_val)
        mse_list.append(test_mse)
        last_tr_list.append(tr_loss_list[-1].numpy())
        if len(best_val_list)>3:
            if (best_val1 == 1e12 and best_val_list[-3]<np.mean(best_val_list[-3:])):
                best_val1 = best_val_list[-3]
                # print('best knots so far', knots_list[:-2])
            else:
                if(best_val_list[-3]<best_val1 and best_val_list[-3]<=best_val_list[-4] and best_val_list[-3]<=best_val_list[-2]):
                    best_val2 = best_val_list[-3]
                    # continue


    ind_nan = clean_nan(mse_list)
    if len(ind_nan)!=0:
        best_val_list = [k for j, k in enumerate(best_val_list) if j not in ind_nan]
        last_tr_list = [k for j, k in enumerate(last_tr_list) if j not in ind_nan]
        mse_list = [k for j, k in enumerate(mse_list) if j not in ind_nan]

    id_val = np.argmin(best_val_list)
    best_test_mse =  mse_list[id_val]
    best_ranges = knots_list[:id_val+1]
    print('best test mse', best_test_mse, 'best ranges', best_ranges)



    with open('kcn_neigh2_bspl_cov_mse_vsg.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i, best_ranges, best_test_mse])

    return best_test_mse, best_ranges



if __name__ == '__main__':

    test_mse, ls = data_results(i=sys.argv[1])
