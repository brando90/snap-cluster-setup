#!/bin/bash
# - snap: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers and support il-action@cs.stanford.edu
# - live server stats: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-gpu-servers-stats

# source $AFS/.bashrc.lfs
source $AFS/.bash_profile
conda activate snap_cluster_setup
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

# -- Run
python ~/snap-cluster-setup/src/train/simple_train.py