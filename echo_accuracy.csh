#!/bin/csh

set i=0
while($i <= 16)
	echo $i
        head -8 MaxZero16_tested_by_Evan_et_al_MaxZero$i | tail -1
	@ i++
end
