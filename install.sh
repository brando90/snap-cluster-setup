#!/bin/bash

# check .bashrc
ln -s /afs/cs.stanford.edu/u/brando9/.bashrc ~/.bashrc

#- check envs
conda info -e

# - activate conda
conda update
pip install --upgrade pip
conda create -n snap_cluster_setup python=3.9
# 3.9 due to vllm
conda activate snap_cluster_setup
# conda deactivate
#conda remove --name snap_cluster_setup --all

#~/miniconda/envs/snap_cluster_setup

# - install this library
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/evals-for-autoformalization.git
echo "---> Warning: Assumes LFS was set to $HOME" 
ln -s /afs/cs.stanford.edu/u/brando9/evals-for-autoformalization $HOME/snap-cluster-setup
ln -s /afs/cs.stanford.edu/u/brando9/snap-cluster-setup $HOME/snap-cluster-setup
pip install -e ~/snap-cluster-setup
cd ~/evals-for-autoformalization

# -- Test pytorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(f'{torch.cuda.device_count()=}'); print(f'Device: {torch.cuda.get_device_name(0)=}')"

# - wandb
pip install --upgrade pip
pip install wandb
pip install wandb --upgrade
wandb login
#wandb login --relogin
cat ~/.netrc

# -- mistral 4.33.4 required
#pip install --upgrade transformers
# pip install git+https://github.com/huggingface/transformers
