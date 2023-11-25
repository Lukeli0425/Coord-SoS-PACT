#!/bin/zsh

for n in {0..269}; do
   # echo $n
   python generate_data_mice.py --n_start $n
done