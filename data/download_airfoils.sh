#!/bin/bash

echo downloading...
wget https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip

echo unpacking...
unzip ./coord_seligFmt.zip

mkdir ./airfoil_database_test

cd ./coord_seligFmt/

# move only selected files to make sure we have the right sets
echo moving...

# test set
mv ag09.dat ah63k127.dat ah94156.dat bw3.dat clarym18.dat e221.dat e342.dat e473.dat e59.dat e598.dat e864.dat fx66h80.dat fx75141.dat fx84w097.dat goe07k.dat goe147.dat goe265.dat goe331.dat goe398.dat goe439.dat goe501.dat goe566.dat goe626.dat goe775.dat hq1511.dat kc135d.dat m17.dat mh49.dat mue139.dat n64012a.dat ../airfoil_database_test/ 

cd ..

rm ./coord_seligFmt.zip
mv coord_seligFmt airfoil_database

echo done!
exit 1

