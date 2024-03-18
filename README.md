# Efficient Speed-of-sound Aberration Correction for Photoacoustic Computed Tomography

<b>[Tianao Li](https://lukeli0425.github.io)</b><sup>1</sup>, <b>Manxiu Cui</b><sup>2</sup>, <b>[Cheng Ma](https://rachmaninov-ma.wixsite.com/mysite)</b><sup>3</sup>, <b>[Emma Alexander](https://www.alexander.vision/emma)</b><sup>1</sup><br>
<sup>1</sup>Northwestern University, <sup>2</sup>Caltech, <sup>3</sup>Tsinghua University<br>
__In Submission__

Official code for [_Efficient Speed-of-sound Aberration Correction for Photoacoustic Computed Tomography_]().


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
nohup python3 train.py --model WienerNet --loss MSE --lr 1e-3 --nc 8 > out/wiener_8_mse.out 2>&1 &
nohup python3 train.py --model WienerNet --loss MultiScale --lr 1e-3 --nc 8 > out/wiener_8_mul.out 2>&1 &

nohup python3 train.py --model WienerNet --loss MSE --lr 1e-3 --nc 16 > out/wiener_16_mse.out 2>&1 &
nohup python3 train.py --model WienerNet --loss MultiScale --lr 1e-3 --nc 16 > out/wiener_16_mul.out 2>&1 &

nohup python3 train.py --model WienerNet --loss MSE --lr 1e-3 --nc 32 > out/wiener_32_mse.out 2>&1 &
nohup python3 train.py --model WienerNet --loss MultiScale --lr 1e-3 --nc 32 > out/wiener_32_mul.out 2>&1 &
```
