#!/usr/bin/python
'''
Abstract:
    This is a program to convert .dat files to npy files
    there are two options
    One is source, please uncomment the upper script to use
    Another is labels, please uncomment the lower scrpt to use
Usage:
    dat2npy.py [option] [file name]
    [option]:
        -l: the code will process the file as a label.
        -d: the code will process the file as a data set.

Example:

    You have a data file "Toy.dat" with contents below.

[[[9.11e-01,1.12e+00,1.02e+00,2.07e-01,1.26e-01,7.32e-02,0.00e+00,0.00e+00],
[0.00e+00,0.00e+00,0.00e+00,1.16e-01,9.51e-02,4.95e-02,7.71e-02,2.35e-01],
[170.,310.,320.,83.,160.,140.,110.,90.]]

    Then, do this cmd.
    $ dat2npy -d Toy.dat
    you get two file, Toy.txt and Toy.npy
    .npy file is binary, can be read by numpy.load()
    .txt file is readable, can be read by numpy.loadtxt()
    you can use "$vim Toy.txt", you should see the content below 

Editor:
    Jacob975

20170123
####################################
update log
20180123 version alpha 1
    Now it works, the code can convert both source and label into tensorflow readable
20180124 version alpha 2
    1. Now feature, you can choose processing label or data by argv.
    2. Now the data will be normalized.
'''
import tensorflow as tf
import time
import re           # this is used to apply multiple spliting
import numpy as np
from sys import argv

# the def is used to read a list of data with the same class.
def read_well_known_data(data_name):
    f = open(data_name, 'r')
    data = np.array([])
    for line in f.readlines():
        # skip if no data or it's a hint.
        if not len(line) or line.startswith('#'):
            continue
        row = np.array(re.split('[,\n\[\]]+', line))
        # clean the empty element
        row_c = np.array([x for x in row if x != ""])
        if len(row_c) != 8:
            print row_c
            print "the row is wrong."
            break
        data = np.append(data, row_c)
    w = 8
    h = len(data)/8
    data = data.reshape((h, w))
    f.close()
    return data

def read_well_known_label(label_name):
    f = open(label_name, 'r')
    content = f.read()
    label = np.array(re.split('[,\n\[\]]+', content))
    label_c =  np.array([x for x in label if x != ""])
    # convert max list to full list
    label_f = [[0 for i in range(3)] for j in range(len(label_c))]
    for i in xrange(len(label_f)):
        label_f[i][int(label_c[i])] = 1
    return label_f

def normalize(inp):
    h = len(inp)
    norm = np.abs(inp).sum(axis=1)
    outp = inp / norm.reshape(h,1)
    return outp

def str2num(inp):
    w = len(inp[0])
    outp = np.array(inp,dtype = float)
    l_outp = len(outp)
    h = l_outp
    outp = outp.reshape((h, w))
    return outp

#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #----------------------------------
    # read argv
    do_label = False
    do_data = False
    for word in argv:
        if word == "-l":
            do_label = True
        if word == "-d":
            do_data = True
    #-----------------------------------
    if do_data:
        # Load data
        str_data = read_well_known_data(argv[-1])
        data = str2num(str_data)
        n_data = normalize(data)
        np.save("{0}.npy".format(argv[-1][:-4]), n_data)
        np.savetxt("{0}.txt".format(argv[-1][:-4]), n_data)
    if do_label:
        # Load label
        str_label = read_well_known_label(argv[-1])
        label = str2num(str_label)
        np.save("{0}.npy".format(argv[-1][:-4]), label)
        np.savetxt("{0}.txt".format(argv[-1][:-4]), label)
    
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print "Exiting Main Program, spending ", elapsed_time, "seconds."
