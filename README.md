# PACT


## Running the Project

The dataset used in this project can be downlaoded the dataset [here](https://figshare.com/articles/dataset/Data/9250784).
To clone this project, run:

```zsh
git clone https://github.com/Lukeli0425/PACT.git
```

Create a virtual environment and download the required packages:

```zsh
pip install -r requirements.txt
```

## Train

```zsh
nohup python3 train.py --model WienerNet --loss MSE --lr 1e-3 --nc 16 > out/wiener_16_mse.out 2>&1 &
nohup python3 train.py --model WienerNet --loss MultiScale --lr 1e-3 --nc 16 > out/wiener_16_mul.out 2>&1 &
```
