#!/bin/zsh

for n in {108..109}; do
   # echo $n
   python generate_data_mice.py --n_start $n
done