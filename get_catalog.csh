#/bin/csh -f

set cat = catalog-SWIRE-v3.tbl

all:
  echo Usage: $0 option
  echo options : star star_tracer galaxy galaxy_tracer yso_tracer

goto $1

star: 
  awk '$17~/star/ && $17!~/dust/ {print "["$34"," $55"," $76"," $97"," $118"," $139"," $160"," $181"," $35"," $56"," $77"," $98"," $119"," $140"," $161"," $182"],"}' $cat > star_sed.dat
  exit 0
star_tracer:
  awk '$17~/star/ && $17!~/dust/ {print FNR }' $cat > star_tracer.dat
  exit 0
galaxy:
  awk '$17~/Galc/	{print "["$34"," $55"," $76"," $97"," $118"," $139"," $160"," $181"," $35"," $56"," $77"," $98"," $119"," $140"," $161"," $182"],"}' $cat > galaxy_sed.dat
  exit 0
galaxy_tracer:
  awk '$17~/Galc/	{print FNR }' $cat > galaxy_tracer.dat
  exit 0
yso_tracer:
  awk '{print NR}' yso_sed.dat > yso_tracer.dat
  exit 0
