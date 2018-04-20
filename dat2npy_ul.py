#!/usr/bin/python3
'''
Abstract:
    This is a program to convert .dat files to npy files
    there are two options
    One is source, please uncomment the upper script to use
    Another is labels, please uncomment the lower scrpt to use
    ################
    Attention:
        The output data are upper-lower type.
Usage:
    dat2npy_ul.py [file name 1] [file name 2] [file name 3] ...
    [file name]:
        The file you want to processed.
        The first will be labeded as 0, the second will be labeded as 1, so as on.
Example:

    You have a data file "Toy.dat", "Toy2.dat", and "Toy3.dat"

    Then, do this cmd.
    $ dat2npy_ul.py Toy.dat Toy2.dat Toy3.dat
    you get two series of files, one is data, another is label
    each series are sort by number of zero in a data.

Editor:
    Jacob975

20170123
####################################
update log
20180123 version alpha 1
    The code is copy from dat2npy.py
20180320 version alpha 2
    1. add tracer to dat data set
20180322 version alpha 3
    1. rename tracer
'''
import tensorflow as tf
import time
import re           # this is used to apply multiple spliting
import numpy as np
from sys import argv
from dat2npy import no_observation_filter

# how many element in a data vector
data_width = 16

# the def is used to read a list of data with the same class.
def read_well_known_data(data_name):
    f = open(data_name, 'r')
    data = []
    for ind, line in enumerate(f.readlines(), start = 0):
        # skip if no data or it's a hint.
        if not len(line) or line.startswith('#'):
            continue
        row = re.split('[,\n\[\]]+', line)
        # clean the empty element
        row_c = [x for x in row if x != ""]
        if len(row_c) != data_width:
            print (line[:-1])
            print ("the row {0} is wrong.\n".format(ind))
            continue
        data.append(row_c)
    f.close()
    return data

def normalize(inp):
    # take norm
    h = len(inp)
    norm = np.amax(inp, axis=1)
    outp = inp / norm.reshape(h,1)
    outp[outp < 0 ]= -9.99e+02
    outp.reshape(-1, data_width)
    return outp

# input data format:
# [S1, S2, S3, ..., S8, E1, E2, E3, ..., E8]
# output data format:
# [Si+E1, S2+E2, S3+E3, ..., S8+E8, S1-E1, S2-E2, ..., S8-E8]

def upperlower(inp):
    sig = inp[:,:8]
    err = inp[:,8:]
    upp = sig + err
    low = sig - err
    outp = np.hstack((upp, low))
    return outp

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #----------------------------------
    # read argv
    data_name_list = argv[1:]
    print ("The command is:\n {0}".format(argv))
    print ("data to be processed: {0}".format(data_name_list))
    #-----------------------------------
    # Load data
    sum_data = [[] for x in range(data_width)]
    sum_label = [[] for x in range(data_width)]
    sum_tracer = [[] for x in range(data_width)]
    for ind, data_name in enumerate(data_name_list, start = 0):
        print ("##############################")
        print ("data name = {0}".format(data_name))
        print ("label = {0}".format(ind))
        # convert data from string to float
        str_data = read_well_known_data(data_name)
        data = np.array(str_data, dtype = float)
        # zero filter
        for i in range(data_width):
            data_z, tracer_outp = no_observation_filter(data_name, data, i)
            data_n_z = normalize(data_z)
            # save tracer
            print ("MaxLoss = {0}, number of data = {1}".format(i, len(data_n_z)))
            data_n_z_ul = upperlower(data_n_z)
            label_z = np.array([ind for x in range(len(data_n_z_ul)) ])
            label_z_f = [[0 for k in range(3)] for j in range(len(label_z))]
            for u in range(len(label_z_f)):
                label_z_f[u][int(label_z[u])] = 1
            # stack them
            sum_data[i] = np.append(sum_data[i], data_n_z_ul)
            sum_label[i] = np.append(sum_label[i], label_z_f)
            sum_tracer[i] = np.append(sum_tracer[i], tracer_outp)
    # save data
    print ("###############################")
    print ("save data, label, and tracer")
    for i in range(data_width):
        sum_data[i] = np.reshape(sum_data[i], (-1, data_width))
        sum_label[i] = np.reshape(sum_label[i], (-1, 3))
        print ("number of data with MaxLoss {0} = {1}".format(i, len(sum_data[i])))
        np.save("source_sed_MaxLoss{0}.npy".format(i), sum_data[i])
        np.savetxt("source_sed_MaxLoss{0}.txt".format(i), sum_data[i])
        np.save("source_id_MaxLoss{0}.npy".format(i), sum_label[i])
        np.savetxt("source_id_MaxLoss{0}.txt".format(i), sum_label[i])
        np.savetxt("source_tracer_MaxLoss{0}.txt".format(i), sum_tracer[i])
        np.save("source_tracer_MaxLoss{0}.npy".format(i), sum_tracer[i])
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
