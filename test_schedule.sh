#!/bin/bash

# code for test AI

# Usage: test_schedule.sh [AI name]

# 20180413 version alpha 1
# The code work

# check arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ${0} [DIR where save AI] [key word]"
    exit 1
fi

# Array of AI name
AI_POOL=${1}
keyword=${2}
echo "AI saved directory going to test:"
for each in ${AI_POOL}/2018*/;
do
    # ${each##*/} means only take the last word of $each
    # ${each::-1} means take $each but the last latter. 
    FULL_AI_NAME=${each::-1}
    AI_NAME=${FULL_AI_NAME##*/}
    echo "##############"
    echo "AI under test: ${AI_NAME}"

    # create a directory to save result of testing
    mkdir -p "AI_${AI_NAME}_test_on_${keyword}"
    sed_test_AI_64_8.py source_sed_${keyword}.npy source_id_${keyword}.npy "AI_${AI_NAME}_test_on_${keyword}"\
                        "${each}checkpoint_AI_64_8_source_sed_${keyword}" \
                        > "AI_${AI_NAME}_test_on_${keyword}/result_of_AI_test"
done
exit 0
