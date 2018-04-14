#!/bin/bash

# code for test AI

# Usage: test_schedule.sh [AI name]

# 20180413 version alpha 1
# The code work

# check arguments
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ${0} [the folder name saved AI]"
    exit 1
fi

# Array of AI name
array=(${2}*/)

echo "AI saved directory going to test:"
for each in "${array[@]}";
do
    echo "${each}"
    mkdir "AI_test_${each}"
    sed_test_AI_64_8.py source_sed_MaxLoss16.npy source_id_MaxLoss16.npy "AI_test_${each}"\
                        "${each}checkpoint_AI_64_8_source_sed_MaxLoss16" \
                        > "AI_test_${each}/result_of_AI_test"
done
exit 0
