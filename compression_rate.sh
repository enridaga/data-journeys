#!/bin/bash

while read f;
do
	o=$(echo $f| sed 's/_DN\.digraph$//')"_DJ.digraph"
	if [[ -f "$o" ]]; then
		fl=$(cat $f|wc -l)
		ol=$(cat $o| wc -l)
		echo $(echo "($fl-$ol)/$fl"|bc -l) $o $ol $f $fl  
	fi
done < <(ls ./datajourneys/*_DN.digraph|sort)
