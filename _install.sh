#!/bin/bash

# check .bashrc
ln -s /afs/cs.stanford.edu/u/brando9/.bashrc ~/.bashrc

#- check envs
conda info -e

# - activate conda
conda update
pip install --upgrade pip

# conda create -n snap_cluster_setup python=3.9
# conda activate snap_cluster_setup_mercury
# 3.9 due to vllm
conda create -n snap_cluster_setup_py311 python=3.11
conda activate snap_cluster_setup_py311

# conda create -n snap_cluster_setup_mercury python=3.11
# conda activate snap_cluster_setup_mercury

# conda deactivate
#conda remove --name snap_cluster_setup --all

#~/miniconda/envs/snap_cluster_setup

# - install this library
cd /afs/cs.stanford.edu/u/brando9/
# git clone git@github.com:brando90/snap-cluster-setup.git
ln -s /afs/cs.stanford.edu/u/brando9/gold-ai-olympiad $HOME/gold-ai-olympiad
ln -s /afs/cs.stanford.edu/u/brando9/putnam-math $HOME/putnam-math
ln -s /afs/cs.stanford.edu/u/brando9/PyPantograph $HOME/PyPantograph
ln -s /afs/cs.stanford.edu/u/brando9/ultimate-fm4math $HOME/ultimate-fm4math
pip install -e ~/snap-cluster-setup

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


# Set the AFS environment variable if it is not already set
export AFS="/afs/cs.stanford.edu/u/brando9"
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
export HOME=$LOCAL_MACHINE_PWD

# Move and rename the VS Code workspaces to the AFS directory, and create symbolic links in the home directory
mv "$HOME/vscode.beyond-scale-language-data-diversity.ampere1.code-workspace" "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace"
ln -s "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace" "$HOME/vscode.beyond-scale-language-data-diversity.code-workspace"

mv "$HOME/vscode.gold-ai-olympiad.afs_snap.code-workspace" "$AFS/vscode.gold-ai-olympiad.code-workspace"
ln -s "$AFS/vscode.gold-ai-olympiad.code-workspace" "$HOME/vscode.gold-ai-olympiad.code-workspace"

mv "$HOME/vscode.evaporate.skampere1.code-workspace" "$AFS/vscode.evaporate.code-workspace"
ln -s "$AFS/vscode.evaporate.code-workspace" "$HOME/vscode.evaporate.code-workspace"

mv "$HOME/vscode.lean4ai.skampere1.code-workspace" "$AFS/vscode.lean4ai.code-workspace"
ln -s "$AFS/vscode.lean4ai.code-workspace" "$HOME/vscode.lean4ai.code-workspace"

mv "$HOME/vscode.snap-cluster-setup.code-workspace" "$AFS/vscode.snap-cluster-setup.code-workspace"
ln -s "$AFS/vscode.snap-cluster-setup.code-workspace" "$HOME/vscode.snap-cluster-setup.code-workspace"

mv "$HOME/vscode.maf_data.creating_data_math_training.skamapere1.code-workspace" "$AFS/vscode.maf_data.creating_data_math_training.code-workspace"
ln -s "$AFS/vscode.maf_data.creating_data_math_training.code-workspace" "$HOME/vscode.maf_data.creating_data_math_training.code-workspace"

mv "$HOME/vscode.maf_data.training_af_model.skampere1.code-workspace" "$AFS/vscode.maf_data.training_af_model.code-workspace"
ln -s "$AFS/vscode.maf_data.training_af_model.code-workspace" "$HOME/vscode.maf_data.training_af_model.code-workspace"

mv "$HOME/vscode.math_evaporate.skampere1.code-workspace" "$AFS/vscode.math_evaporate.code-workspace"
ln -s "$AFS/vscode.math_evaporate.code-workspace" "$HOME/vscode.math_evaporate.code-workspace"

mv "$HOME/vscode.beyond-scale-language-data-diversity.skampere1.code-workspace" "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace"
ln -s "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace" "$HOME/vscode.beyond-scale-language-data-diversity.code-workspace"

mv "$HOME/KoyejoLab-Predictable-LLM-Evals.skampere1.code-workspace" "$AFS/vscode.KoyejoLab-Predictable-LLM-Evals.code-workspace"
ln -s "$AFS/vscode.KoyejoLab-Predictable-LLM-Evals.code-workspace" "$HOME/vscode.KoyejoLab-Predictable-LLM-Evals.code-workspace"
