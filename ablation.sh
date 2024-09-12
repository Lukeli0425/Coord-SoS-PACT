
gpu=0

# # Number of delays.
# for n_delays in {64,32,16,8,4}; do
#   python reconstruction.py --task numerical --n_delays $n_delays --lam_tv 5e-5 --n_epochs 10 --gpu $gpu
# done

# for n_delays in {64,32,16,8,4}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays --lam_tv 2e-5 --n_epochs 10 --gpu $gpu
# done

# for n_delays in {64,32,16,8,4}; do
#   python reconstruction.py --task phantom --n_delays $n_delays --lam_tv 5e-5 --n_epochs 10 --gpu $gpu
# done


# # Network structure.
# for hidden_fts in {32,64,128,256}; do
#   for hidden_lyrs in {1,2}; do
#     python reconstruction.py --task numerical --hidden_fts $hidden_fts --hidden_lyrs $hidden_lyrs --lam_tv 5e-5 --n_epochs 10 --gpu $gpu
#   done
# done

# for hidden_fts in {32,64,128,256}; do
#   for hidden_lyrs in {1,2}; do
#     python reconstruction.py --task in_vivo --hidden_fts $hidden_fts --hidden_lyrs $hidden_lyrs --lam_tv 2e-5 --n_epochs 10 --gpu $gpu
#   done
# done

# for hidden_fts in {32,64,128,256}; do
#   for hidden_lyrs in {1,2}; do
#     python reconstruction.py --task phantom --hidden_fts $hidden_fts --hidden_lyrs $hidden_lyrs --lam_tv 5e-5 --n_epochs 10 --gpu $gpu
#   done
# done


# # Learning rate.
# for lr in {5e-3,2e-3,1e-3,5e-4,2e-4,1e-4}; do
#   python reconstruction.py --task numerical --lam_tv 0e-5 --n_epochs 30 --lr $lr --gpu $gpu
# done

# for lr in {5e-3,2e-3,1e-3,5e-4,2e-4,1e-4}; do
#   python reconstruction.py --task in_vivo --lam_tv 0e-5 --n_epochs 30 --lr $lr --gpu $gpu
# done

# for lr in {5e-3,2e-3,1e-3,5e-4,2e-4,1e-4}; do
#   python reconstruction.py --task phantom --lam_tv 0e-5 --n_epochs 30 --lr $lr --gpu $gpu
# done


# # TV Regularization.
# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task numerical --lam_tv $lam_tv --n_epochs 15 --lr 1e-3 --gpu $gpu
# done

# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task in_vivo --lam_tv $lam_tv --n_epochs 10 --lr 1e-3 --gpu $gpu
# done

# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task phantom --lam_tv $lam_tv --n_epochs 15 --lr 1e-3 --gpu $gpu
# done


# Multi-channel Deconvolution.
for n_delays in {64,32,16,8,4}; do
  python reconstruction.py --task numerical --method Deconv --n_delays $n_delays --gpu $gpu
done