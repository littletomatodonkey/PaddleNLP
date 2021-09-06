#!/bin/bash 
export PYTHONPATH=../../:$PYTHONPATH
python3.7 create_pretrain_data.py merge
