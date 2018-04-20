#!/usr/bin/python3
'''
Abstract:
    This is a program to demo how to trace a datum 
Usage:
    test_trace.py [DIR] [keyword]
Example:
    test_tracer.py . MaxLoss15
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
    1. The code work 
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
    #----------------------------------------
    # initialize variables
    data = None
    tracer = None
    cls_pred = None
    cls_true = None
    directory = None
    #----------------------------------------
    # load argv
    if len(argv) != 3:
        print ("Error!\nUsage: test_tracer.py [directory] [keyword]")
        exit()
    directory = argv[1]
    keyword = argv[2]
    #----------------------------------------
    # load tracer
    failure, data, tracer = load_lib.load_arrangement(keyword, directory)
    if not failure:
        print ("load data and tracer success")
    #----------------------------------------
    # load cls_pred
    failure, cls_pred = load_lib.load_cls_pred(keyword, directory)
    if not failure:
        print ("load cls_pred success")
    #----------------------------------------
    # load cls_true
    failure, cls_true = load_lib.load_cls_true(keyword, directory)
    if not failure:
        print ("load cls_true success")
    #----------------------------------------
    # test
    print ("### data number ###")
    print ("length of training data set: {0}".format(len(data.train.images)))
    print ("length of validation data set: {0}".format(len(data.validation.images)))
    print ("length of test data set: {0}".format(len(data.test.images)))
    print ("({0} | {1})\n".format(len(cls_pred), len(cls_true)))
    # confusion matrix
    print ("### confusion matrix ###")
    failure, cm = load_lib.confusion_matrix(cls_true, cls_pred)
    if not failure:
        print ("confusion matrix success")
    print (cm)
    # print data and the corresponding shuffle tracer of the first data
    print ("### The first datum in dataset ###")
    print ("data.test.images: {0}".format(data.test.images[0]))
    print ("shuffle tracer: {0}".format(tracer.test[0]))
    print ("true label: {0}".format(cls_true[0]))
    print ("predict label: {0}".format(cls_pred[0]))
    # print data and the corresponding shuffle tracer of the last data
    print ("### The final datum in dataset ###")
    print ("data.test.images: {0}".format(data.test.images[-1]))
    print ("shuffle tracer: {0}".format(tracer.test[-1]))
    print ("true label: {0}".format(cls_true[-1]))
    print ("predict label: {0}\n".format(cls_pred[-1]))
    #-----------------------------------
    print ("number of stars: {0}".format(len(cls_true[cls_true == 0])))
    print ("number of galaxies: {0}".format(len(cls_true[cls_true == 1])))
    print ("number of YSOs: {0}".format(len(cls_true[cls_true == 2])))
    gala2yso = tracer.test[(cls_true == 1) &(cls_pred == 2)]
    print (gala2yso)
    print ("number of galc to yso: {0}".format(len(gala2yso)))
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
