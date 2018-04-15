#!/bin/bash

# Usage: AI_schedule.sh [key word] [number of iterations]

# 20180412 version alpha 1
# The code work

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage:    ${0##*/} [key word] [number of iterations]"
    echo "Example : ${0##*/} MaxLoss15 5"
    exit 1
fi

keyword=${1}

iter=1
while [ $iter -le ${2} ]
do
        # record when the program start
        # time stamp is used as identification
        timestamp=`date --rfc-3339=seconds`
        mkdir "${timestamp}"
        sed_04_64_8.py source_sed_${keyword}.npy source_id_${keyword}.npy "${timestamp}" > "${timestamp}/Iters_${iter}"
        (( iter++ ))
done
exit 0
