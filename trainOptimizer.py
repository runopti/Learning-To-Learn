import numpy as np
import tensorflow as tf

import argparse    
parser = argparse.ArgumentParser(description='Define parameters to learn optimizer.')
parser.add_argument('--n_samplings', type=int, default=10)
parser.add_argument('--n_unroll', type=int, default=20) 
parser.add_argument('--n_dimension', type=int, default=3) 
parser.add_argument('--n_hidden', type=int, default=5) 
parser.add_argument('--n_layers', type=int, default=2) 

args = parser.parse_args()

T = args.n_samplings # number of random function samplings. L will be averaged over T.
n_unroll_in_m = args.n_unroll # the trained optimizers were unrolled for 20 steps.
num_of_coordinates = n_dimension = args.n_dimension # the dimension of input data space
hidden_size = args.n_hidden # This will be used when we actually use this rnn to generate a search direction for the next time step.
num_layers = args.n_layers # n_layer LSTM architecture


def trainOptimizer():
    g = tf.Graph()
    ### BEGIN: GRAPH CONSTRUCTION ###
    with g.as_default():
        cell_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        for i in range(num_of_coordinates):
            cell_list.append(tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)) # num_layers = 2 according to the paper.
        
        loss = 0
        for t in range(T):
            # random sampling of one instance of the quadratic function
            W = tf.truncated_normal([n_dimension, n_dimension]); y = tf.truncated_normal([n_dimension, 1])
            theta = tf.truncated_normal([n_dimension, 1])
            f = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y))
            batch_size = 1
            state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(num_of_coordinates)] 
            sum_f = 0
            g_new_list = []
            grad_f = tf.gradients(f, theta)[0]
            for i in range(num_of_coordinates):
                cell = cell_list[i]; state = state_list[i]
                grad_h_t = tf.slice(grad_f, begin=[i,0], size=[1,1])
                for k in range(n_unroll_in_m):
                    if k > 0: tf.get_variable_scope().reuse_variables() 
                    cell_output, state = cell(grad_h_t, state) # g_new should be a scalar b/c grad_h_t is a scalar
                    softmax_w = tf.get_variable("softmax_w", [hidden_size, 1])
                    softmax_b = tf.get_variable("softmax_b", [1])
                    g_new_i = tf.matmul(cell_output, softmax_w) + softmax_b
                
                g_new_list.append(g_new_i)
                state_list[i] = state # for the next t
            
            # update parameter 
            g_new = tf.reshape(tf.squeeze(tf.pack(g_new_list)), [n_dimension, 1]) # should be a [n_dimension, 1] tensor
            theta = tf.add(theta, g_new)
            
            f_at_theta_t = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y))
            sum_f = sum_f + f_at_theta_t 
    
            loss += sum_f
        
        loss = loss / T

        tvars = tf.trainable_variables() # should be just the variable inside the RNN
        grads = tf.gradients(loss, tvars)
        lr = 0.001 # Technically I need to do random search to decide this
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        ###  END: GRAPH CONSTRUCTION ###
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_epoch = 100
            for epoch in range(max_epoch):
                cost, _ = sess.run([loss, train_op])
                print("Epoch %d : loss %f" % (epoch, cost))
            
            print("Saving the trained model...")
            saver = tf.train.Saver()
            saver.save(sess, "model", global_step=0)

            import pickle
            import time
            print("Extracting variables...")
            now = time.time()
            variable_dict = {}
            for var in tf.trainable_variables():
                print(var.name)
                print(var.eval())
                variable_dict[var.name] = var.eval()
            print("elapsed time: {0}".format(time.time()-now))
            with open("variable_dict.pickle", "wb") as f:
                pickle.dump(variable_dict,f)


if __name__ == "__main__":
    trainOptimizer()
        


