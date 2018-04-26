#!/bin/bash

# Usage: AI_schedule_MaxLoss.sh

# 20180412 version alpha 1
# The code work

iter=0
while [ $iter -le 15 ]
do
        # record when the program start
        # time stamp is used as identification
        timestamp=`date --rfc-3339=seconds`
        mkdir "${timestamp}_trained_by_MaxLoss${iter}"
        sed_04_64_8.py source_sed_MaxLoss${iter}.npy source_id_MaxLoss${iter}.npy \
                        "${timestamp}_trained_by_MaxLoss${iter}" > "${timestamp}_trained_by_MaxLoss${iter}/Iter_log"
        (( iter++ ))
done
exit 0
