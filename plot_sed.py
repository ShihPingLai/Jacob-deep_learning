#!/usr/bin/python3
'''
Abstract:
    This is a program to show the data with different true and prediction 
Usage:
    plot_sed.py [keyword] [true label] [pred label]
Example:
    plot_sed.py MaxLoss15 1 2
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
import numpy as np
import time
import load_lib
import collections
from sys import argv
from glob import glob
import matplotlib.pyplot as plt

def get_sed(detected_occurance, n, data, tracer):
    # initialize variables
    normed_by_band = [dict() for i in range(8)]
    for key in detected_occurance:
        if detected_occurance[key] >= n:
            selected_data = data[np.where(tracer == key)] 
            ind_of_peak = np.argmax(selected_data)
            if ind_of_peak >= 8:
                continue
            else:
                normed_by_band[ind_of_peak][key] = selected_data
    return normed_by_band

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #----------------------------------------
    # initialize variables and constants
    data = None
    tracer = None
    cls_pred = None
    cls_true = None
    collected_tracer_in_confusion_matrix = np.array([])
    collected_sed_in_confusion_matrix = np.array([])
    normed_by_band = None
    n = 5
    true_ = pred_ = ["star", "gala", "yso"]
    #----------------------------------------
    # load argv
    if len(argv) != 4:
        print ("Error!\nUsage: plot_sed.py [keyword] [true label] [pred label]")
        exit()
    keyword = argv[1]
    true_label = int(argv[2])
    pred_label = int(argv[3])
    #----------------------------------------
    data_list = glob("AI*test_on*")
    for directory in data_list:
        print ("#################################")
        print ("start to loading data saved in {0}".format(directory))
        # load tracer
        failure, data, tracer = load_lib.load_arrangement(keyword, directory)
        if not failure:
            print ("load data and tracer success")
        # load cls_pred
        failure, cls_pred = load_lib.load_cls_pred(keyword, directory)
        if not failure:
            print ("load cls_pred success")
        # load cls_true
        failure, cls_true = load_lib.load_cls_true(keyword, directory)
        if not failure:
            print ("load cls_true success")
        # confusion matrix
        print ("### confusion matrix ###")
        failure, cm = load_lib.confusion_matrix(cls_true, cls_pred)
        if not failure:
            print ("confusion matrix success")
        print (cm)
        #-----------------------------------
        star_length = len(cls_true[cls_true == 0])
        print ("number of stars: {0}".format(len(cls_true[cls_true == 0])))
        gala_length = len(cls_true[cls_true == 1])
        print ("number of galaxies: {0}".format(len(cls_true[cls_true == 1])))
        yso_length = len(cls_true[cls_true == 2])
        print ("number of YSOs: {0}".format(len(cls_true[cls_true == 2])))
        tracer_in_confusion_matrix = tracer.test[(cls_true == true_label) &(cls_pred == pred_label)]
        collected_tracer_in_confusion_matrix = np.append(collected_tracer_in_confusion_matrix, tracer_in_confusion_matrix)
        print ("number of gala to yso: {0}".format(len(tracer_in_confusion_matrix)))
        # save tracer_in_confusion_matrix
        np.save("{0}/{1}_tracer_true_{2}_pred_{3}.npy".format(directory, keyword, true_[true_label], pred_[pred_label]), 
                tracer_in_confusion_matrix)
        np.savetxt("{0}/{1}_tracer_true_{2}_pred_{3}.txt".format(directory, keyword, true_[true_label], pred_[pred_label]), 
                tracer_in_confusion_matrix)
    # save collected_tracer_in_confusion_matrix
    np.save("all_tracer_true_{0}_pred_{1}.npy".format(true_[true_label], pred_[pred_label]), collected_tracer_in_confusion_matrix)
    np.savetxt("all_tracer_true_{0}_pred_{1}.txt".format(true_[true_label], pred_[pred_label]), collected_tracer_in_confusion_matrix)
    # sort object by band
    print("detect the occurance")
    detected_occurance = collections.Counter(collected_tracer_in_confusion_matrix)
    print("select by band")
    normed_by_band = get_sed(detected_occurance, n, data.test.images, tracer.test)
    # plot the sed band by band
    for ind, peak_at in enumerate(normed_by_band):
        if len(peak_at) == 0:
            continue
        result_plt = plt.figure("sed of true: {0}, pred: {1}, peak at {2} band".format(true_[true_label], pred_[pred_label], ind))
        for key, value in peak_at.items():
            plt.plot(range(1, 17), value[0])
        result_plt.savefig("sed_true_{0}_pred_{1}_peak_at_{2}_band.png".format(true_[true_label], pred_[pred_label], ind))
    #----------------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
