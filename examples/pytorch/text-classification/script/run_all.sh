#!/bin/bash
for i in 5 6 7 8
do
    sh run_dc.sh $i
    if [ "$?" != 0 ]
    then
        break
    fi
    # sh run_dc_static.sh $i
done
# sh run.sh
# sh run_sublinear.sh
# sh run_gc.sh
