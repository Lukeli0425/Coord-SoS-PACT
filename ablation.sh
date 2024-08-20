
gpu=0

# # Number of delays.
# for n_delays in {64,32,16,8,4}; do
#   python reconstruction.py --task numerical --n_delays $n_delays --lam_tv 5e-5 --n_iters 30 --gpu $gpu
# done

# for n_delays in {64,32,16,8,4}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays --lam_tv 1e-5 --n_iters 30 --gpu $gpu
# done

# for n_delays in {64,32,16,8,4}; do
#   python reconstruction.py --task phantom --n_delays $n_delays --lam_tv 5e-5 --n_iters 30 --gpu $gpu
# done


# Network structure.
for hidden_fts in {32,64,128,256}; do
  for hidden_lyrs in {1,2}; do
    python reconstruction.py --task numerical --hidden_fts $hidden_fts --hidden_lyrs $hidden_lyrs --lam_tv 5e-5
  done
done

for hidden_fts in {32,64,128,256}; do
  for hidden_lyrs in {1,2}; do
    python reconstruction.py --task in_vivo --hidden_fts $hidden_fts --hidden_lyrs $hidden_lyrs --lam_tv 1e-5
  done
done

for hidden_fts in {32,64,128,256}; do
  for hidden_lyrs in {1,2}; do
    python reconstruction.py --task phantom --hidden_fts $hidden_fts --hidden_lyrs $hidden_lyrs --lam_tv 5e-5
  done
done

Learning rate.
for lr in {5e-4,2e-4,1e-4,5e-5,2e-5}; do
  python reconstruction.py --task numerical--lr $lr --n_iters 50 --gpu $gpu
done

# for lr in {5e-4,2e-4,1e-4,5e-5,2e-5}; do
#   python reconstruction.py --task in_vivo --n_iters 50 --lr $lr --gpu $gpu
# done

# for lr in {5e-4,2e-4,1e-4,5e-5,2e-5}; do
#   python reconstruction.py --task phantom --lr $lr --n_iters 50 --gpu $gpu
# done


# TV Regularization.
# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task numerical --lam_tv $lam_tv --n_iters 30 --gpu $gpu
# done

# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task phantom --lr 1e-4 --lam_tv $lam_tv --n_iters 30 --gpu $gpu
# done

# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task in_vivo --lam_tv $lam_tv --n_iters 30 --gpu $gpu
# done