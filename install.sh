#!/bin/bash

#- check envs
conda info -e

# - activate conda
conda update
pip install --upgrade pip
conda create -n evals_af python=3.10
conda activate evals_af
# conda deactivate
#conda remove --name maf --all

# - install this library
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/evals-for-autoformalization.git
echo "---> Warning: Assumes LFS was set to $HOME" 
ln -s /afs/cs.stanford.edu/u/brando9/evals-for-autoformalization $HOME/evals-for-autoformalization
pip install -e ~/evals-for-autoformalization
#pip uninstall ~/massive-autoformalization-maf
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
