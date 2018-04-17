#!/usr/bin/python3
'''
Program:
This is a program to setup DL python3 code. 
For Linux and macOS only
Usage:
    1. DL_setup.py
editor:
    Jacob975

##################################
#   Python3                      #
#   This code is made in python3 #
##################################

20170809
#################################
update log
20180403 version alpha 1:
    1. the code to modify python3 path.

'''
import time
import glob
import os
import fnmatch
import DL_conf
from sys import platform, exit

def readfile(filename):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        # skip if no data or it's a hint.
        if line == "\n" or line.startswith('#'):
            continue
        data.append(line[:-1])
    f.close
    return data

# This method is used to set the path in the first line of programm
def set_path_linux(py_path, path = ""):
    py_list = glob.glob("{0}/*.py".format(path))
    for name in py_list:
        temp = 'sed -i "1s?.*?{0}?" {1}'.format(py_path, name)
        if VERBOSE>0:print (temp)
        os.system(temp)

# This method is used to set the path in the first line of each DL_python programm file
def set_path_mac(py_path, path = ""):
    py_list = glob.glob("{0}/*.py".format(path))
    for name in py_list:
        temp = 'sed -i '' -e "1s?.*?{0}?" {1}'.format(py_path, name)
        if VERBOSE>0:print (temp)
        os.system(temp)
    # remove by-product
    temp = "rm *-e"
    os.system(temp)

#--------------------------------------------
# main code
VERBOSE = 1
# measure times
start_time = time.time()
# get the path of python from DL_conf.py
py_path = DL_conf.path_of_python
py_path = "#!{0}".format(py_path)
if VERBOSE>0:
    print ("path of python3 writen in DL_conf.py: {0}".format(py_path))
# get path of DL code from DL_conf
code_path = DL_conf.path_of_source_code
# set path of python into all DL_python program file
if platform == "linux" or platform == "linux2":
    set_path_linux(py_path, code_path)
    # process all code in nest folder
    obj_list = glob.glob("*")
    for obj in obj_list:
        if os.path.isdir(obj):
            os.chdir(obj)
            temp_path = "{0}/{1}".format(code_path, obj)
            set_path_linux(py_path, temp_path)
            os.chdir(code_path)

elif platform == "darwin":
    set_path_mac(py_path, code_path)
    # process all code in nest folder
    obj_list = glob.glob("*")
    for obj in obj_list:
        if os.path.isdir(obj):
            os.chdir(obj)
            temp_path = "{0}/{1}".format(code_path, obj)
            set_path_mac(py_path, temp_path)
            os.chdir(code_path)

else:
    print ("you system is not fit the requirement of DL_python")
    print ("Please use Linux of macOS")
    exit()

# back to path of code
os.chdir(code_path)
# measuring time
elapsed_time = time.time() - start_time
print ("Exiting Setup Program, spending ", elapsed_time, "seconds.")
