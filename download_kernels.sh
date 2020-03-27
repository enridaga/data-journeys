#!/bin/bash


#rm -rf kernels
#rm -rf download_kernes.log
#for kernel in $(cat kernels.csv |cut -f1 -d',')
for kernel in $(tail kernels.csv |cut -f1 -d',')
do
	echo 'Downloading '$kernel
	kaggle kernels pull $kernel --path kernels
done