#/bin/csh -f

# remove the line with 0
sed -i -e "/0\.,/d" $1
sed -i -e "/,0\./d" $1
sed -i -e "/0\.00e\+00/d" $1
