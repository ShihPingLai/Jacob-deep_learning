#!/usr/bin/python3
'''
Abstract:
    This is a program to save the arrangement of random data process. 
Usage:
    import save_lib.py
Editor:
    Jacob975

##################################
#   Python3                      #
#   This code is made in python3 #
##################################

20170104
####################################
update log

20180321 version alpha 1:
    1. save the arrangement of random data process.

'''
import tensorflow as tf
import time
import numpy as np
import os


def save_arrangement(argv, time_stamp, data, index):
    # save the index to the origin.
    if not os.path.exists(time_stamp):
        os.makedirs(time_stamp)
    np.save("{0}/training_index_{1}.npy".format(time_stamp, argv[1][:-4]), index.train)
    np.savetxt("{0}/training_index_{1}.txt".format(time_stamp, argv[1][:-4]), index.train)
    np.save("{0}/validation_index_{1}.npy".format(time_stamp, argv[1][:-4]), index.validation)
    np.savetxt("{0}/validation_index_{1}.txt".format(time_stamp, argv[1][:-4]), index.validation)
    np.save("{0}/test_index_{1}.npy".format(time_stamp, argv[1][:-4]), index.test)
    np.savetxt("{0}/test_index_{1}.txt".format(time_stamp, argv[1][:-4]), index.test)
    # save the distribution
    np.save("{0}/training_set_{1}.npy".format(time_stamp, argv[1][:-4]), data.train.images)
    np.savetxt("{0}/training_set_{1}.txt".format(time_stamp, argv[1][:-4]), data.train.images)
    np.save("{0}/training_label_{1}.npy".format(time_stamp, argv[1][:-4]), data.train.labels)
    np.savetxt("{0}/training_label_{1}.txt".format(time_stamp, argv[1][:-4]), data.train.labels)
    np.save("{0}/validation_set_{1}.npy".format(time_stamp, argv[1][:-4]), data.validation.images)
    np.savetxt("{0}/validation_set_{1}.txt".format(time_stamp, argv[1][:-4]), data.validation.images)
    np.save("{0}/validation_labels_{1}.npy".format(time_stamp, argv[1][:-4]), data.validation.labels)
    np.savetxt("{0}/validation_labels_{1}.txt".format(time_stamp, argv[1][:-4]), data.validation.labels)
    np.save("{0}/test_set_{1}.npy".format(time_stamp, argv[1][:-4]), data.test.images)
    np.savetxt("{0}/test_set_{1}.txt".format(time_stamp, argv[1][:-4]), data.test.images)
    np.save("{0}/test_labels_{1}.npy".format(time_stamp, argv[1][:-4]), data.test.labels)
    np.savetxt("{0}/test_labels_{1}.txt".format(time_stamp, argv[1][:-4]), data.test.labels)
    return 0

def save_cls_pred(argv, time_stamp, cls_pred):
    np.save("{0}/test_cls_pred_{1}.npy".format(time_stamp, argv[1][:-4]), cls_pred)
    np.savetxt("{0}/test_cls_pred_{1}.txt".format(time_stamp, argv[1][:-4]), cls_pred)
    return 0

def save_cls_true(argv, time_stamp, cls_true):
    np.save("{0}/test_cls_true_{1}.npy".format(time_stamp, argv[1][:-4]), cls_true)
    np.savetxt("{0}/test_cls_true_{1}.txt".format(time_stamp, argv[1][:-4]), cls_true)
    return 0
