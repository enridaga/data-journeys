#!/bin/bash


[ -f kernels.csv ] && rm kernels.csv

for i in {1..11}
do
 kaggle kernels list --language python --page-size 100 -p $i --kernel-type notebook -v >> kernels.csv
 echo $i
done
