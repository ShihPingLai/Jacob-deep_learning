#!/usr/bin/python3
'''
Abstract:
    This is a program to demo how to code deep learning code.
Usage:
    std_code.py
Editor:
    Jacob975

20170104
####################################
update log
20180104 version alpha 1
    The code demo good
20180204 version alpha 2
    Move to python3 instead of python2
'''
import tensorflow as tf
import time

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #-----------------------------------
    # compare the difference between normal variable and tensorflow node.
    x = 1
    y = x + 9
    print (y)
    x = tf.constant(1, name = "x")
    y = tf.Variable(x+9, name = "y")    # y save the key of the node.
    model = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(model)
    print (sess.run(y))
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
