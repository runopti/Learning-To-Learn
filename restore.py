import numpy as np
import tensorflow as tf

def restore_trained_optimizer_variables():
    variable_dict = {}
    g1 = tf.Graph()
    with g1.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("./model-0-0.meta")
            saver.restore(sess, "./model-0-0")
            # I need to extract all model's parameters here and then construct another graph
            for var in tf.trainable_variables():
                print(var.name) # # Getting the parameters inside the RNN. 
                print(var.eval()) # store these values in numpy array.
                # mapping by names I think?
                variable_dict[var.name] = var.eval() # like this?
    return variable_dict

import time
import pickle
with tf.Graph().as_default():
    with tf.Session() as sess:
    #g_op, f_grad_ph = build_optimizer_graph()
        now = time.time() 
        variable_dict = restore_trained_optimizer_variables()
        print(variable_dict)
        print("elapsed time: {0}".format(time.time()-now))
        
        with open("variable_dict_test.pickle", "wb") as f:
            pickle.dump(variable_dict,f)
        
