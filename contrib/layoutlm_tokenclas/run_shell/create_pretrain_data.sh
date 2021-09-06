#!/bin/bash 
export PYTHONPATH=../../:$PYTHONPATH
total_num=12
for cur_no in $(seq 1 $total_num)  
do
    python3.7 create_pretrain_data.py gen $total_num $cur_no &
done
