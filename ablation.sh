
gpu=0
n_delays_df=16
n_epochs_df=10
lr_df=5e-3
bs_df=64


################### Delay-and-sum ####################
for sample_id in {5,6,7,8,9}; do
  for v_das in {1500..1520}; do
    python reconstruction.py --task numerical --sample_id $sample_id --method DAS --v_das $v_das --gpu $gpu
  done
done


################### Dual-SOS DAS ####################
for sample_id in {5,6,7,8,9}; do
  for v_body in {1535..1570}; do
    python reconstruction.py --task numerical --sample_id $sample_id --method Dual-SOS_DAS --v_body $v_body --gpu $gpu
  done
done


################### Multi-channel Deconvolution ####################
for sample_id in {5,6,7,8,9}; do
  for n_delays in {64,32,16,8,4,2,1}; do
    python reconstruction.py --task numerical --sample_id $sample_id --n_delays $n_delays --method Deconv --batch_size $bs_df --gpu $gpu
  done
done


#################### NF-APACT ####################
hfs_df=128
hls_df=0

# # Learning rate.
# for lr in {1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4}; do
#   python reconstruction.py --task numerical --sample_id 0 --n_delays $n_delays_df --hls $hls_df --hfs $hfs_df --n_epochs 30 --lr $lr --batch_size $bs_df --gpu $gpu
# done
# for lr in {1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays_df --hls $hls_df --hfs $hfs_df --n_epochs 30 --lr $lr --batch_size $bs_df --gpu $gpu
# done
# for lr in {1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4}; do
#   python reconstruction.py --task phantom --n_delays $n_delays_df --hls $hls_df --hfs $hfs_df --n_epochs 30 --lr $lr --batch_size $bs_df --gpu $gpu
# done


# Number of delays.
for sample_id in {5,6,7,8,9}; do
  for n_delays in {64,32,16,8,4,2,1}; do
    python reconstruction.py --task numerical --sample_id $sample_id --n_delays $n_delays --hls $hls_df --hfs $hfs_df --n_epochs $n_epochs_df --lr $lr_df --batch_size 64 --gpu $gpu
  done
done

# for n_delays in {64,32,16,8,4,2,1}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays --hls $hls_df --hfs $hfs_df --n_epochs $n_epochs_df --lr $lr_df --batch_size 64 --gpu $gpu
# done

# for n_delays in {64,32,16,8,4,2,1}; do
#   python reconstruction.py --task phantom --n_delays $n_delays --hls $hls_df --hfs $hfs_df --n_epochs $n_epochs_df --lr $lr_df --batch_size 64 --gpu $gpu
# done


# Network structure.
for sample_id in {5,6,7,8,9}; do
  for hfs in {16,32,64,128,256}; do
    for hls in {0,1}; do
      python reconstruction.py --task numerical --sample_id $sample_id --n_delays $n_delays_df --hfs $hfs --hls $hls --n_epochs $n_epochs_df --batch_size $bs_df --lr $lr_df --gpu $gpu
    done
  done
done

# for hfs in {16,32,64,128,256}; do
#   for hls in {0,1}; do
#     python reconstruction.py --task in_vivo --n_delays $n_delays_df --hfs $hfs --hls $hls --n_epochs $n_epochs_df --lr $lr_df --batch_size $bs_df --gpu $gpu
#   done
# done

# for hfs in {16,32,64,128,256}; do
#   for hls in {0,1}; do
#     python reconstruction.py --task phantom --n_delays $n_delays_df --hfs $hfs --hls $hls --n_epochs $n_epochs_df --lr $lr_df --batch_size $bs_df --gpu $gpu
#   done
# done


# # TV Regularization.
# for sample_id in {5,6,7,8,9}; do
#   for lam_tv in {1e-6,1e-5,1e-4,1e-3,1e-2}; do
#     python reconstruction.py --task numerical --sample_id $sample_id --lam_tv $lam_tv --n_epochs 20 --lr 1e-3 --gpu $gpu
#   done
# done

# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task in_vivo --lam_tv $lam_tv --n_epochs 10 --lr 1e-3 --gpu $gpu
# done

# for lam_tv in {0,1e-5,2e-5,5e-5,1e-4}; do
#   python reconstruction.py --task phantom --lam_tv $lam_tv --n_epochs 15 --lr 1e-3 --gpu $gpu
# done


################### Pixel Grid ####################
lr_pg_df=0.1
lam_tv_df=0 # 1e-4
# # Learning rate.
# for lr in {2e-1,1e-1,5e-2,2e-2,1e-2,5e-3}; do
#   python reconstruction.py --task numerical --sample_id 0 --n_delays $n_delays_df --method PG --lam_tv $lam_tv_df --n_epochs 30 --lr $lr --batch_size $bs_df --gpu $gpu
# done

# for lr in {2e-1,1e-1,5e-2,2e-2,1e-2,5e-3}; do
#   python reconstruction.py --task phantom --n_delays $n_delays_df --method PG --lam_tv $lam_tv_df --n_epochs 30 --lr $lr --batch_size $bs_df --gpu $gpu
# done

# for lr in {2e-1,1e-1,5e-2,2e-2,1e-2,5e-3}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays_df --method PG --lam_tv $lam_tv_df --n_epochs 30 --lr $lr --batch_size $bs_df --gpu $gpu
# done

# Number of delays.
for sample_id in {5,6,7,8,9}; do
  for n_delays in {64,32,16,8,4,2,1}; do
    python reconstruction.py --task numerical --sample_id $sample_id --n_delays $n_delays_df --method PG --n_delays $n_delays --lam_tv $lam_tv_df --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size 64 --gpu $gpu
  done
done

# Number of delays (w/o TV).
for sample_id in {5,6,7,8,9}; do
  for n_delays in {64,32,16,8,4,2,1}; do
    python reconstruction.py --task numerical --sample_id $sample_id --n_delays $n_delays_df --method PG --n_delays $n_delays --lam_tv 0 --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size 64 --gpu $gpu
  done
done

# for n_delays in {64,32,16,8,4,2,1}; do
#   python reconstruction.py --task phantom --n_delays $n_delays_df --method PG --n_delays $n_delays --lam_tv $lam_tv_df --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size 64 --gpu $gpu
# done

# for n_delays in {64,32,16,8,4,2,1}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays_df --method PG --n_delays $n_delays --lam_tv $lam_tv_df --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size 64 --gpu $gpu
# done

# TV Regularization.
for sample_id in {5,6,7,8,9}; do
  for lam_tv in {0,1e-6,1e-5,1e-4,1e-3,1e-2}; do
    python reconstruction.py --task numerical --sample_id $sample_id --n_delays $n_delays_df --method PG --lam_tv $lam_tv --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size $bs_df --gpu $gpu
  done
done

# for lam_tv in {5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3}; do
#   python reconstruction.py --task phantom --n_delays $n_delays_df --method PG --lam_tv $lam_tv --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size $bs_df --gpu $gpu
# done

# for lam_tv in {5e-6,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3}; do
#   python reconstruction.py --task in_vivo --n_delays $n_delays_df --method PG --lam_tv $lam_tv --n_epochs $n_epochs_df --lr $lr_pg_df --batch_size $bs_df --gpu $gpu
# done




#################### Delay-and-sum ####################
# for v_das in {1508,1509,1510,1511,1512}; do
#   python reconstruction.py --task numerical --sample_id 0 --method DAS --v_das $v_das --gpu $gpu
# done


#################### Dual-SOS DAS ####################
# for v_body in {1557,1558,1559,1560,1561,1562,1563,1564}; do
#   python reconstruction.py --task numerical --sample_id 0 --method Dual-SOS_DAS --v_body $v_body --gpu $gpu
# done


# ################### APACT ####################

# Number of delays.
for sample_id in {5,6,7,8,9}; do
  for n_delays in {64,32,16,8,4,2,1}; do
    python reconstruction.py --task numerical --sample_id $sample_id --method APACT --n_delays $n_delays --lam_tsv 5e-15 --n_iters 10 --lr 50 --gpu $gpu
  done
done

# for n_delays in {64,32,16,8,4,2,1}; do
#   python reconstruction.py --task phantom --method APACT --n_delays $n_delays --lam_tsv 5e-15 --n_iters 10 --lr 50 --gpu $gpu
# done


# for n_delays in {64,32,16,8,4,2,1}; do
#   python reconstruction.py --task in_vivo --method APACT --n_delays $n_delays --lam_tsv 5e-15 --n_iters 10 --lr 50 --gpu $gpu
# done


