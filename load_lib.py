#!/usr/bin/python3
'''
Abstract:
    This is a program to load the tracer and data from files. 
Usage:
    import load_lib.py
Editor:
    Jacob975

##################################
#   Python3                      #
#   This code is made in python3 #
##################################

20170411
####################################
update log

20180411 version alpha 1:
    1. The library work
'''
import tensorflow as tf
import time
import numpy as np
import os

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

from astro_mnist import DataSet, shuffled_tracer

# This is used to load data, label, and tracer
def load_arrangement(sub_name, 
                    time_stamp, 
                    data, 
                    tracer,
                    reshape=False,
                    dtype=dtypes.float32,
                    seed=None,
                    ):
    # time_stamp is used to create a uniq folder
    # sub_name is used to denote filename
    # data contains data set and label
    # tracer is just tracer
    # create folder
    if not os.path.exists(time_stamp):
        print("Directory not found")
        return 1
    # if directory is not null, load data, labels and tracers
    try:
        train_tracer    = np.load("{0}/training_tracer_{1}.npy".format(time_stamp, sub_name))
        train_data      = np.load("{0}/training_set_{1}.npy".format(time_stamp, sub_name))
        train_labels    = np.load("{0}/training_label_{1}.npy".format(time_stamp, sub_name))
        valid_tracer    = np.load("{0}/validation_tracer_{1}.npy".format(time_stamp, sub_name))
        valid_data      = np.load("{0}/validation_set_{1}.npy".format(time_stamp, sub_name))
        valid_labels    = np.load("{0}/validation_labels_{1}.npy".format(time_stamp, sub_name))
        test_tracer     = np.load("{0}/test_tracer_{1}.npy".format(time_stamp, sub_name))
        test_data       = np.load("{0}/test_set_{1}.npy".format(time_stamp, sub_name))
        test_labels     = np.load("{0}/test_labels_{1}.npy".format(time_stamp, sub_name))
    except:
        print("data or label or tracer aren't completed")
        return 1
    options = dict(dtype=dtype, reshape=reshape, seed=seed)
    # generate tracer
    tracer = shuffled_tracer(train_tracer, valid_tracer, test_tracer)
    # generate data and index
    train = DataSet(train_data, train_labels, **options)
    validation = DataSet(valid_data, valid_labels, **options)
    test = DataSet(test_data, test_labels, **options)
    data = base.Datasets(train=train, validation=validation, test=test)
    return 0

# This is used to loading pred label
def load_cls_pred(sub_name, time_stamp, cls_pred):
    # time_stamp is used to create a uniq folder
    # sub_name is used to denote filename
    # cls_pred is predicted label
    try:
        cls_pred = np.load("{0}/test_cls_pred_{1}.npy".format(time_stamp, sub_name))
    except:
        print("test_cls_pred not found")
        return 1
    return 0

# THis is used to loading true label
def load_cls_true(sub_name, time_stamp, cls_true):
    # time_stamp is used to create a uniq folder
    # sub_name is used to denote filename
    # cls_pred is true label
    try:
        cls_true = np.load("{0}/test_cls_true_{1}.npy".format(time_stamp, sub_name))
    except:
        print ("test_cls_true not found")
        return 1
    return 0
