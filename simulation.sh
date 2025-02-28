gpu=2

for sample in {5,6,7,8,9}; do
    python simulation.py --sample $sample --gpu $gpu
done
