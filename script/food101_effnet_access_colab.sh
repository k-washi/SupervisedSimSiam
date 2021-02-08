#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/1N_aXT_8YxamsoYlAzUEcIhtVa94KeBZ9
  sleep 3600
done
