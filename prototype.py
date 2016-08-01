import tensorflow as tf
import numpy as np

# Let's say the objective funtion is 
f(\theta) = || W \theta - y ||^2, where the elements of W and y are from Gaussian.  
# g has to be the same size as the parameter size. 
g, state = lstm(input_t, hidden_state) # here, input_t is the gradient of a hidden state at time t w.r.t. the hidden
# update equation
param = param + g


# The loss L(\phi) can be computed by double for loop.
# For each loop, a different function is randomly sampled from a distribution of f. 
# Then, theta_t will be computed by the update equation (


# So, overall, what I need to implement is the two-layer coordinate-wise LSTM cell. 

T = 10 # number of random function samplings. L will be averaged over T.
num_of_small_t = 100 # Each function was optimized for 100 steps
num_of_unroll_in_m = 20  # the trained optimizers were unrolled for 20 steps. This will be used when we actually use this rnn to generate a search direction for the next time step. 
hidden_size = 20
num_layers = 2

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
state = tf.zeros([1, 2*self.hidden_size])  
# or initial_state = cell.zero_state(batch_size, tf.float32)
cell_output, hidden_state = cell(grad_f_t, state)
# how to get grad_f_t? 
grad_f_t = tf.gradients(f)??? 

# update parameters
W = W + cell_output_whole # cell_output_whole[i] = i th cell_output

# Write the above stuff more cleanly:
# IMPORTANT: we need create cell for each coordinate.
# initialize parameter W, and state for LSTM-optimizer; t = 0
# compute f_grad_t
# get g_(t+1), hidden_state = cell(f_grad_t, state)
# cell_output_whole[i] = ith g_(t+1)
# update parameters: W_t+1 = W_t + cell_output_whole 


# create the list to store lstm-optimizers
lstm_list = []
for k in range(num_of_coordinates): # with tf.name_scope("id") as scope: ? or I could do the same thing with tf.get_variable(kkk, name=string_that_determins_id) ?
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    state = tf.zeros([1, 2*self.hidden_size])

# Experiment 1:
# Assuming f(x) = ||Wx - y||^2
for i in range(T):
    W = get_new_W(); y = get_new_y()
    theta = init_theta()
    f = tf.square(tf.matmul(W, theta) - y)
    sum_of_f_t = 0
    num_of_coordinates = 10
    for t in range(num_of_small_t):
        # here recurrence happens

        grad_f_t = tf.gradients(f) #or something like that
        for k in range(num_of_coordinates):
            with tf.name_scope("%s_%d" % (lstm, k)) as scope:
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
                state = tf.zeros([1, 2*self.hidden_size])
                g_new, hidden_state = cell(grad_f_t[k], state)
                theta[k] = theta[k] + g_new
        f_evaluate_at_theta_t = tf.square(tf.matmul(W, theta) - y)
        sum_of_f_t = sum_of_f_t + f_evaluated_at_theta_t 
        
    L = L + sum_of_f_t  
L = L / T # This is the final loss

# Is this right? I don't think so. There should be some variables that I need to delte in terms of backprop gradient computation.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(L)


class LSTM_optim(object):
    __init__():

