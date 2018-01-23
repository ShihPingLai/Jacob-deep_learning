#!/usr/bin/python
'''
Abstract:
    This is a program to convert .dat files to npy files
Usage:
    dat2npu.py
Editor:
    Jacob975

20170123
####################################
update log
    20180123 version alpha 1
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
    #-----------------------------------
    # Load data
    str_data = read_well_known_data(argv[-1])
    data = str2num(str_data)
    np.save("{0}.npy".format(argv[-1][:-4]), data)
    np.savetxt("{0}.txt".format(argv[-1][:-4]), data)
    '''
    # Load label
    str_label = read_well_known_label(argv[-1])
    label = str2num(str_label)
    np.save("{0}.npy".format(argv[-1][:-4]), label)
    np.savetxt("{0}.txt".format(argv[-1][:-4]), label)
    '''
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print "Exiting Main Program, spending ", elapsed_time, "seconds."
