#!/bin/bash

exitcode=0
for fname in `find . -name "*.py"`; 
do
  diff -q <(head -14 $fname) <(cat license-header.txt) > /dev/null
  if [ $? -ne 0 ]
  then
    echo "$fname does not have license header. Please copy and paste ./license-header.txt"
    exitcode=1
  fi
done
exit ${exitcode}
