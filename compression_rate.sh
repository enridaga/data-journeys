#!/bin/bash

w=true
o=""
while read f;
do
  if [ "$w" = true ]; then
    o="$f"
    w=false
  else
    w=true
    fl=$(cat $f|wc -l)
    ol=$(cat $o|wc -l)
    echo $(echo "($fl-$ol)/$fl"|bc -l) $o $ol $f $fl
  fi
  
done < <(ls ./datajourneys/*.digraph|sort)
