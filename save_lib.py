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
20180322 version alpha 2:
    1. rearrange save condition.

20180323 version alpha 3:
    1. add explaination
'''
import tensorflow as tf
import time
import numpy as np
import os

# This is used to save data, label, and tracer
def save_arrangement(argv, time_stamp, data, tracer):
    # time_stamp is used to create a uniq folder
    # argv is used to denote filename
    # data contains data set and label
    # tracer is just tracer
    # create folder
    if not os.path.exists(time_stamp):
        os.makedirs(time_stamp)
    # if train set is not null, save data, labels and tracers
    if len(data.train.images):
        np.save("{0}/training_tracer_{1}.npy".format(time_stamp, argv[1][:-4]), tracer.train)
        np.savetxt("{0}/training_tracer_{1}.txt".format(time_stamp, argv[1][:-4]), tracer.train)
        np.save("{0}/training_set_{1}.npy".format(time_stamp, argv[1][:-4]), data.train.images)
        np.savetxt("{0}/training_set_{1}.txt".format(time_stamp, argv[1][:-4]), data.train.images)
        np.save("{0}/training_label_{1}.npy".format(time_stamp, argv[1][:-4]), data.train.labels)
        np.savetxt("{0}/training_label_{1}.txt".format(time_stamp, argv[1][:-4]), data.train.labels)
    if len(data.validation.images):
        np.save("{0}/validation_tracer_{1}.npy".format(time_stamp, argv[1][:-4]), tracer.validation)
        np.savetxt("{0}/validation_tracer_{1}.txt".format(time_stamp, argv[1][:-4]), tracer.validation)      
        np.save("{0}/validation_set_{1}.npy".format(time_stamp, argv[1][:-4]), data.validation.images)
        np.savetxt("{0}/validation_set_{1}.txt".format(time_stamp, argv[1][:-4]), data.validation.images)
        np.save("{0}/validation_labels_{1}.npy".format(time_stamp, argv[1][:-4]), data.validation.labels)
        np.savetxt("{0}/validation_labels_{1}.txt".format(time_stamp, argv[1][:-4]), data.validation.labels)
    if len(data.test.images):
        np.save("{0}/test_tracer_{1}.npy".format(time_stamp, argv[1][:-4]), tracer.test)
        np.savetxt("{0}/test_tracer_{1}.txt".format(time_stamp, argv[1][:-4]), tracer.test)
        np.save("{0}/test_set_{1}.npy".format(time_stamp, argv[1][:-4]), data.test.images)
        np.savetxt("{0}/test_set_{1}.txt".format(time_stamp, argv[1][:-4]), data.test.images)
        np.save("{0}/test_labels_{1}.npy".format(time_stamp, argv[1][:-4]), data.test.labels)
        np.savetxt("{0}/test_labels_{1}.txt".format(time_stamp, argv[1][:-4]), data.test.labels)
    return 0

# This is used to saving pred label
def save_cls_pred(argv, time_stamp, cls_pred):
    # time_stamp is used to create a uniq folder
    # argv is used to denote filename
    # cls_pred is predicted label
    np.save("{0}/test_cls_pred_{1}.npy".format(time_stamp, argv[1][:-4]), cls_pred)
    np.savetxt("{0}/test_cls_pred_{1}.txt".format(time_stamp, argv[1][:-4]), cls_pred)
    return 0

# THis is used to saving true label
def save_cls_true(argv, time_stamp, cls_true):
    # time_stamp is used to create a uniq folder
    # argv is used to denote filename
    # cls_pred is true label
    np.save("{0}/test_cls_true_{1}.npy".format(time_stamp, argv[1][:-4]), cls_true)
    np.savetxt("{0}/test_cls_true_{1}.txt".format(time_stamp, argv[1][:-4]), cls_true)
    return 0
