#!/bin/bash
## Based on: /mnt/mimer/sjan/qsubCnt/test/test.sh for inspiration
## And see email from Mike McWilliams on 10-11th of July, 2017

LIST=$(pwd)/list
echo $LIST
if [ -e $LIST ]
then
	rm $LIST
fi

#for F in NS003*/*script
#do
#  echo $F
#	/home/MET/qsubCnt $F 50 $LIST
#done


H=$(pwd) ; 
for D in LES* 
do 
  cd $D
  cp ../base/IEA* .
  cp ../base/Fund* .
  cp ../base/Infile* .
  #/home/MET/qsubCnt jobscript 50 $LIST
  ../Flex5
  cd $H
done
