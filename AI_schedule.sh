#!/bin/bash

# 20180412 version alpha 1
# The code work

iter=0
while [ $iter -le 5 ]
do
        # record when the program start
        # time stamp is used as identification
        timestamp=`date --rfc-3339=seconds`
        mkdir "${timestamp}"
        sed_04_64_8.py source_sed_MaxLoss16.npy source_id_MaxLoss16.npy "${timestamp}" > "${timestamp}/Iters_${iter}"
        (( iter++ ))
done
