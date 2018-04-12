#!/usr/bin/python3
'''
Abstract:
    This is a program to demo how to trace a datum 
Usage:
    test_trace.py
Editor:
    Jacob975

##################################
#   Python3                      #
#   This code is made in python3 #
##################################

20180412
####################################
update log
20180412 version alpha 1:
    1. 
'''
import tensorflow as tf
import time
import load_lib
from sys import argv

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #-----------------------------------
    # initialize variables
    data = None
    tracer = None
    cls_pred = None
    cls_true = None
    sub_name = "source_sed_MaxLoss16"
    time_stamp = None
    # load argv
    if len(argv) != 2:
        print ("Error!\nUsage: test_tracer.py [directory]")
        exit()
    time_stamp = argv[1]
    # load tracer
    failure, data, tracer = load_lib.load_arrangement(sub_name, time_stamp)
    if not failure:
        print ("load data and tracer success")
    failure, cls_pred = load_lib.load_cls_pred(sub_name, time_stamp)
    if not failure:
        print ("load cls_pred success")
    failure, cls_true = load_lib.load_cls_true(sub_name, time_stamp)
    if not failure:
        print ("load cls_true success")
    # test
    print ("length of training data set: {0}".format(len(data.train.images)))
    print ("length of validation data set: {0}".format(len(data.validation.images)))
    print ("length of test data set: {0}".format(len(data.test.images)))
    print ("{0} | {1}\n".format(len(cls_pred), len(cls_true)))
    # confusion matrix
    failure, cm = load_lib.plot_confusion_matrix(cls_true, cls_pred)
    if not failure:
        print ("confusion matrix success")
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
