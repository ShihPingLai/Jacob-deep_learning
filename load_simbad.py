#!/usr/bin/python3
'''
Abstract:
    This is a program to convert simbad data which is just download from internet into table.
Usage:
    load_simbad.py [filename]
    filename: the name of input file
Input:
    simbad like mess data
Output:
    A .tbl file contain a table split by "|"
Editor:
    Jacob975

20170204
####################################
update log
    20180204 version alpha 1
        The code work well
'''
import time
import os
from sys import argv
#--------------------------------------------
# main code
if __name__ == "__main__":
    VERBOSE = 0
    # measure times
    start_time = time.time()
    #-----------------------------------
    # get the name of your simbad file
    simbad_file_name = argv[-1]
    print (simbad_file_name)
    # choose the line with numerical header, and remove the numarical index then add to a new file
    simbad_table_name = simbad_file_name[:-7]+".tbl"
    print (" simbad_table_name : {0}".format(simbad_table_name ))
    os.system('awk \'/^[0-9]/{{$1=""; print $0}}\' {0} > {1}'.format(simbad_file_name, simbad_table_name))
    # remove duplicated lines
    simbad_uniq_table_name = simbad_table_name[:-4]+"_u.tbl"
    print (" simbad_uniq_table_name : {0}".format(simbad_uniq_table_name ))
    os.system("sort {0} | uniq > {1}".format(simbad_table_name, simbad_uniq_table_name))
    # add column name
    column_name = "dist(asec)|        identifier        |typ| coord1 (ICRS,J2000/2000)  |Mag U |Mag B |Mag V |Mag R |Mag I |Mag G |Mag J |Mag H |Mag K |Mag u |Mag g |Mag r |Mag i |Mag z |  spec. type   |#bib|#not"
    os.system("sed  -i '1i {0}' {1}".format(column_name, simbad_uniq_table_name))
    # modify the format
    os.system("sed 's/^\ |//g' {0} > {1}".format(simbad_uniq_table_name, simbad_table_name))
    os.system("rm {0}".format(simbad_uniq_table_name))
    #-----------------------------------
    # measuring time
    elapsed_time = time.time() - start_time
    print ("Exiting Main Program, spending ", elapsed_time, "seconds.")
