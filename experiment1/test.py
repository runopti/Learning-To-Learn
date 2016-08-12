import numpy as np
import tensorflow as tf

def test():
    T = 100 # 5 # 100 # number of random function samplings. L will be averaged over T.
    # num_of_small_t = 100 # Each function was optimized for 100 steps
    n_unroll_in_m = 20  # the trained optimizers were unrolled for 20 steps. 
    num_of_coordinates = n_dimension = 10
    # This will be used when we actually use this rnn to generate a search direction for the next time step. 
    hidden_size = 20
    num_layers = 2
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
                #print(theta_i)
                #if t == 0 or t == 1:
                #    grad_h_t = tf.truncated_normal([1,1])
                #    print("ok")
                #else:
                    #print("ok2")
                #grad_h_t = tf.gradients(f, theta_i)[0] # should be zero at the first iteration b/c state is a zero vector.
                #grad_h_t = tf.Print(grad_h_t, [grad_h_t], message="This is grad_ht_t: ", summarize=100)
                #print(grad_h_t) # why this is None...there is no relation b/w state and theta_i???
                # the return value of tf.gradients is a sum of each dim of state so grad_h_t should be a scalar
                for k in range(n_unroll_in_m):
                    if k > 0: tf.get_variable_scope().reuse_variables() 
                    cell_output, state = cell(grad_h_t, state) # g_new should be a scalar b/c grad_h_t is a scalar
                    softmax_w = tf.get_variable("softmax_w", [hidden_size, 1])
                    softmax_b = tf.get_variable("softmax_b", [1])
                    g_new_i = tf.matmul(cell_output, softmax_w) + softmax_b
                
                g_new_list.append(g_new_i)
                state_list[i] = state # for the next t
            
            # update parameter 
            g_new = tf.reshape(tf.squeeze(tf.pack(g_new_list)), [n_dimension, 1]) # should be a [10, 1] tensor
            theta = tf.add(theta, g_new)
            
            f_at_theta_t = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y))
            sum_f = sum_f + f_at_theta_t 
    
            loss += sum_f
        
        loss = loss / T

        tvars = tf.trainable_variables() # should be just the variable inside the RNN
        grads = tf.gradients(loss, tvars)
        lr = 0.001 #?? Need to do random search
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        ###  END: GRAPH CONSTRUCTION ###
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_epoch = 100
            for epoch in range(max_epoch):
                cost, _ = sess.run([loss, train_op])
                print(cost)
            
            print("Saving the trained model...")
            saver = tf.train.Saver()
            saver.save(sess, "model-0", global_step=0)

test()

