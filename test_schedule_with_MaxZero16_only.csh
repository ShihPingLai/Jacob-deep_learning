#!/bin/csh

set i=0
while($i <= 16)
        sed_test_AI_64_8.py source_sed_MaxLoss16.npy source_id_MaxLoss16.npy /mazu/users/Jacob975/data_Evan_et_al/witherror/AI_64_8_source_sed_MaxLoss$i/checkpoints_500000.0 > MaxLoss16_tested_by_Evan_et_al_MaxLoss$i
        @ i++
end
