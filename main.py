import tensorflow as tf
import numpy as np

import argparse        
parser = argparse.ArgumentParser(description='Define parameters to learn optimizer.')
parser.add_argument('--n_samplings', type=int, default=10) 
parser.add_argument('--n_unroll', type=int, default=20) 
parser.add_argument('--n_dimension', type=int, default=3)
parser.add_argument('--n_hidden', type=int, default=5) 
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--optim_method', type=str, default="lstm") 

args = parser.parse_args()

T = args.n_samplings # number of random function samplings. L will be averaged over T.
n_unroll_in_m = args.n_unroll # the trained optimizers were unrolled for 20 steps.
num_of_coordinates = n_dimension = args.n_dimension # the dimension of input data space
hidden_size = args.n_hidden # This will be used when we actually use this rnn to generate a search direction for the next time step.
num_layers = args.n_layers # n_layer LSTM architecture
max_epoch = args.max_epoch

def get_inputs(n_dimension, n):
    theta_param = np.random.randn(n_dimension, 1)
    
    W_inputs = np.random.randn(n, n_dimension, n_dimension)
    y_inputs = np.zeros([n, n_dimension, 1])
    for i in range(n):
        y_inputs[i] = np.dot(W_inputs[i], theta_param)
    
    return W_inputs, y_inputs

def restore_trained_optimizer_variables():
    variable_dict = {}
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("./model-0.meta")
            saver.restore(sess, "./model-0")
            # I need to extract all model's parameters here and then construct another graph
            for var in tf.trainable_variables():
                print(var.name) # # Getting the parameters inside the RNN. 
                print(var.eval()) # store these values in numpy array.
                # mapping by names I think?
                variable_dict[var.name] = var.eval() # like this?
    return variable_dict


def build_optimizer_graph():
    ### BEGIN: GRAPH CONSTRCUTION  ###
    grad_f = tf.placeholder(tf.float32, [n_dimension, 1])
    
    cell_list = []
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    for i in range(num_of_coordinates):
        cell_list.append(tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)) # num_layers = 2 according to the paper.        
    batch_size = 1
    state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(num_of_coordinates)] 
    sum_f = 0
    g_new_list = []
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
        # state_list[i] = state # for the next t # I don't need this list right..? b/c I'm not using t...T thing.

    # Reshaping g_new
    g_new = tf.reshape(tf.squeeze(tf.pack(g_new_list)), [n_dimension, 1]) # should be a [10, 1] tensor         

    
    return g_new, grad_f


def build_training_graph(method):
    n = T
    W = tf.placeholder(tf.float32, shape=[n, n_dimension, n_dimension])
    y = tf.placeholder(tf.float32, shape=[n, n_dimension, 1])
    theta = tf.Variable(tf.truncated_normal([n_dimension, 1]))
    if method == "lstm":
        g_new = tf.placeholder(tf.float32, shape=[n_dimension, 1])
    
    loss = 0
    for i in range(n):
        W_i = tf.reshape(tf.slice(W, begin=[i, 0, 0], size=[1, n_dimension, n_dimension]), [n_dimension, n_dimension])
        y_i = tf.reshape(tf.slice(y, begin=[i, 0, 0], size = [1, n_dimension, 1]), [n_dimension, 1])
        f = tf.reduce_sum(tf.square(tf.matmul(W_i, theta) - y_i)) # make this faster using tensor only
        loss += f
    loss /= (n*n_dimension)
    
    f_grad = tf.gradients(loss, theta)[0]
    
    if method == "SGD":
        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        # train_op = tf.train.AdamOptimizer().minimize(loss)
        return loss, train_op, W, y
    
    if method == "lstm":
        new_value = tf.add(theta, g_new)
        train_op = tf.assign(theta, new_value) # just to make it compatiable with method == "SGD case. 

        return loss, f_grad, train_op, g_new, W, y

def main():
    # variable name convention
    # **_ph : placeholders
    # **_op : nodes other than placeholders.
    # **_val: actual numpy values.
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            if args.optim_method == "lstm":
                loss_op, f_grad_op, train_op, g_new_ph, W_ph, y_ph = build_training_graph(method=args.optim_method)
                g_op, f_grad_ph = build_optimizer_graph()
            elif args.optim_method =="SGD":
                loss_op,  train_op, W_ph, y_ph = build_training_graph(method=args.optim_method)
            else:
                print("Define optimization method.")
                exit()

            sess.run(tf.initialize_all_variables())
            
            # Restore the trained optimizer in order to get the values
            # variable_dict = restore_trained_optimizer_variables()
            import pickle
            with open("variable_dict.pickle","rb") as f:
                variable_dict = pickle.load(f)

            ## INITIALIZATION BEGIN ##
            for var in tf.trainable_variables():
                # assign values using the numpy arrays to the current graph.     
                if var.name in variable_dict:
                    assign_op = var.assign(variable_dict[var.name]) # the inside param has to be a np array like this: var.assign(np.ones(12))
                    sess.run(assign_op)
            
            W_val, y_val = get_inputs(n_dimension, T)
            ## INITIALIZATION DONE ##     
            
            cost_list = []  
            if args.optim_method == "lstm":
                g_new_val = np.zeros([n_dimension,1])
                for epoch in range(max_epoch):
                    loss_val, f_grad_val, _ = sess.run([loss_op, f_grad_op, train_op], feed_dict={g_new_ph: g_new_val , W_ph: W_val, y_ph: y_val})
                    g_new_val = sess.run(g_op, feed_dict={f_grad_ph: f_grad_val})
                    print(loss_val)
                    cost_list.append(loss_val)
            if args.optim_method == "SGD":
                for epoch in range(max_epoch):
                    loss_val, _ = sess.run([loss_op, train_op], feed_dict={W_ph: W_val, y_ph: y_val})
                    print(loss_val)
                    cost_list.append(loss_val)


            with open("cost_list_" + args.optim_method + ".pickle","wb") as f:
                pickle.dump(cost_list, f)

            # % matplotlib inline
            import matplotlib.pyplot as plt
            plt.plot(range(len(cost_list)), cost_list)
            imagename = "figure_" + args.optim_method + ".png"
            plt.savefig(imagename)

if __name__ == "__main__": 
    main()
