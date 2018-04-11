#!/usr/bin/python3
'''
Abstract:
    This is a program to test load_lib.py
Usage:
    test_load_lib.py
Editor:
    Jacob975

##################################
#   Python3                      #
#   This code is made in python3 #
##################################

20180411
####################################
update log

20180411 version alpha 1:
    Both test_load_lib.py and load_lib.py work
'''
import tensorflow as tf
import time
import load_lib

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #-----------------------------------
    data = None
    tracer = None
    sub_name = "source_sed_MaxLoss16"
    time_stamp = 'Friday, 23. March 2018 11:16AM'
    if not load_lib.load_arrangement(sub_name, time_stamp, data, tracer):
        print ("success")
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
