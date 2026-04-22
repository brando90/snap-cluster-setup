#!/bin/bash
echo "---> WARNING: DO NOT RUN THIS FILE DIRECTLY. RUN EACH COMMAND MANUALLY AND FIX IT IF IT DOES NOT WORK"

# make .bashrc be in your local lfs
ln -s $AFS/.bashrc $HOME/.bashrc

# -- Install miniconda
# get conda from the web
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# source conda with the bash command and put the installation at $HOME/miniconda
bash $HOME/miniconda.sh -b -p $HOME/miniconda
# activate conda
source $HOME/miniconda/bin/activate
# Set up conda
conda init
# conda init zsh
conda init bash
conda install conda-build
conda update -n bas

# - Update conda
conda update
pip install --upgrade pip
conda info -e

# - activate conda
conda create -n snap_cluster_setup python=3.11 -y
conda activate snap_cluster_setup
# conda remove --name snap_cluster_setup --all
pip install -e ~/gold-ai-snap_cluster_setup
pip install -e .

conda create -n snap_cluster_setup python=3.11 -y
conda activate snap_cluster_setup
# # conda remove --name snap_cluster_setup --all
pip install -e ~/snap-cluster-setup
pip install -e .

# conda create -n pred_llm_evals_env python=3.11 -y
conda env create --file environment.yml -y
conda activate pred_llm_evals_env
# # conda remove --name snap_cluster_setup --all
# pip install -e ~/pred_llm_evals_env
# pip install -e .

#  old
# conda create -n gold_ai_olympiad python=3.9 -y
# conda activate gold_ai_olympiad
# # # needed in this order for vllm to work: ref: https://github.com/vllm-project/vllm/issues/2747
# # pip install vllm 
# # # conda deactivate
# # # conda remove --name gold_ai_olympiad --all
# pip install -e ~/gold-ai-olympiad

# - install this library
cd /afs/cs.stanford.edu/u/brando9/
# git clone git@github.com:brando90/gold-ai-olympiad.git
# Set the AFS environment variable if it is not already set
export AFS="/afs/cs.stanford.edu/u/brando9"
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
export HOME=$LOCAL_MACHINE_PWD
ln -s /afs/cs.stanford.edu/u/brando9/snap-cluster-setup $HOME/snap-cluster-setu
ln -s /afs/cs.stanford.edu/u/brando9/gold-ai-olympiad $HOME/gold-ai-olympiad
ln -s /afs/cs.stanford.edu/u/brando9/putnam-math $HOME/putnam-math
ln -s /afs/cs.stanford.edu/u/brando9/PyPantograph $HOME/PyPantograph
ln -s /afs/cs.stanford.edu/u/brando9/ultimate-fm4math $HOME/ultimate-fm4math
# ln -s /afs/cs.stanford.edu/u/brando9/KoyejoLab-Predictable-LLM-Evals $HOME/KoyejoLab-Predictable-LLM-Evals  # needs to be git cloned at lfs due to large files (not afs)
cd ~; git clone git@github.com:RylanSchaeffer/KoyejoLab-Predictable-LLM-Evals.git
ln -s $AFS/MetaMath $HOME/MetaMath

pip install -e ~/gold-ai-olympiad
#pip uninstall ~/gold-ai-olympiad
cd ~/gold-ai-olympiad

# - lean4ai
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/lea
ln -s /afs/cs.stanford.edu/u/brando9/lean4ai $HOME/lean4ai
#pip install -e ~/lean4ai

# - get my mathematics_in_lean repo via git submodels
cd $HOME/gold-ai-olympiad
git fetch
# adds the repo to the .gitmodule & clones the repo
git submodule add -f --name mathematics_in_lean git@github.com:brando90/mathematics_in_lean.git mathematics_in_lean/
# git submodule init initializes your local configuration file to track the submodules your repository uses, it just sets up the configuration so that you can use the git submodule update command to clone and update the submodules.
git submodule init
# - The --remote option tells Git to update the submodule to the commit specified in the upstream repository, rather than the commit specified in the superproject's repository. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
git submodule update --init --recursive --remote
# # for each submodule pull from the right branch according to .gitmodule file. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
# git submodule foreach -q --recursive 'git switch $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master || echo main )'
# - check it's in specified branch. ref: https://stackoverflow.com/questions/74998463/why-does-git-submodule-status-not-match-the-output-of-git-branch-of-my-submodule
git submodule status
cd mathematics_in_lean

# -- make vscode workspace: https://gist.github.com/brando90/57c119c621da19fc804fd91bc18eb03c
cd /afs/cs.stanford.edu/u/brando9/
vscode_file="vscode.gold-ai-olympiad.afs_snap.code-workspace"
echo $vscode_file
touch $vscode_file
echo '{
    "folders": [
        {
            "path": "gold-ai-olympiad"
        }
    ],
    "settings": {}

' > $vscode_file
ln -s /afs/cs.stanford.edu/u/brando9/$vscode_file $HOME/$vscode_file

# -- Test pytorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(f'{torch.cuda.device_count()=}'); print(f'Device: {torch.cuda.get_device_name(0)=}')"

# -- Setup our lean_src_proj project for the first time (ref: https://leanprover-community.github.io/install/project.html#lean-projects)
# - Go to a folder where you want to create a project in a subfolder lean_src, and type lake +leanprover/lean4:nightly-2023-02-04 new lean_src math
# go to root of project (not the lean project)
cd $HOME/gold-ai-olympiad
# Option1: set up the main lean_src_proj project for the first time:
# lake +leanprover/lean4:nightly-2023-02-04 new lean_src_proj math
# Option 2: set up main lean_src_proj project if you did a git clone of root gold-ai-olympiad (so depedencies, e.g., matblib need to be installed)
# mkdir lean_src_proj
# cd lean_src_proj
# lake update
# lake exe cache get
# if you get an error message saying lake is an unknown command and you have not logged in since you installed Lean, then you may need to first type 
# source ~/.profile or source ~/.bash_profile
# -- Go inside the clean_src folder and type `lake update``, then `lake exe cache get`` and then `mkdir MyProject``
cd lean_src_proj
# optionally: rm -rf lean_src_proj/.git
lake update 
lake exe cache get
# -- Your Lean code should now be put inside files with extension .lean living in lean_src/MyProject/ or a subfolder thereof.
mkdir MyProject
cd MyProject
echo 'import Mathlib.Topology.Basic\n\n#check TopologicalSpace' > Test.lean

# - wandb
pip install --upgrade pip
pip install wandb
pip install wandb --upgrade
wandb login
#wandb login --relogin
cat ~/.netrc

# Set the AFS environment variable if it is not already set
export AFS="/afs/cs.stanford.edu/u/brando9"
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
export HOME=$LOCAL_MACHINE_PWD

# Move and rename the VS Code workspaces to the AFS directory, and create symbolic links in the home directory
# mv "$HOME/vscode.beyond-scale-language-data-diversity.ampere1.code-workspace" "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace"
ln -s "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace" "$HOME/vscode.beyond-scale-language-data-diversity.code-workspace"

# mv "$HOME/vscode.gold-ai-olympiad.afs_snap.code-workspace" "$AFS/vscode.gold-ai-olympiad.code-workspace"
ln -s "$AFS/vscode.gold-ai-olympiad.code-workspace" "$HOME/vscode.gold-ai-olympiad.code-workspace"

# mv "$HOME/vscode.evaporate.skampere1.code-workspace" "$AFS/vscode.evaporate.code-workspace"
ln -s "$AFS/vscode.evaporate.code-workspace" "$HOME/vscode.evaporate.code-workspace"

# mv "$HOME/vscode.lean4ai.skampere1.code-workspace" "$AFS/vscode.lean4ai.code-workspace"
ln -s "$AFS/vscode.lean4ai.code-workspace" "$HOME/vscode.lean4ai.code-workspace"

# mv "$HOME/vscode.snap-cluster-setup.code-workspace" "$AFS/vscode.snap-cluster-setup.code-workspace"
ln -s "$AFS/vscode.snap-cluster-setup.code-workspace" "$HOME/vscode.snap-cluster-setup.code-workspace"

# mv "$HOME/vscode.maf_data.creating_data_math_training.skamapere1.code-workspace" "$AFS/vscode.maf_data.creating_data_math_training.code-workspace"
ln -s "$AFS/vscode.maf_data.creating_data_math_training.code-workspace" "$HOME/vscode.maf_data.creating_data_math_training.code-workspace"

# mv "$HOME/vscode.maf_data.training_af_model.skampere1.code-workspace" "$AFS/vscode.maf_data.training_af_model.code-workspace"
ln -s "$AFS/vscode.maf_data.training_af_model.code-workspace" "$HOME/vscode.maf_data.training_af_model.code-workspace"

# mv "$HOME/vscode.math_evaporate.skampere1.code-workspace" "$AFS/vscode.math_evaporate.code-workspace"
ln -s "$AFS/vscode.math_evaporate.code-workspace" "$HOME/vscode.math_evaporate.code-workspace"

# mv "$HOME/vscode.beyond-scale-language-data-diversity.skampere1.code-workspace" "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace"
ln -s "$AFS/vscode.beyond-scale-language-data-diversity.code-workspace" "$HOME/vscode.beyond-scale-language-data-diversity.code-workspace"

# mv "$HOME/KoyejoLab-Predictable-LLM-Evals.skampere1.code-workspace" "$AFS/vscode.KoyejoLab-Predictable-LLM-Evals.code-workspace"
ln -s "$AFS/vscode.KoyejoLab-Predictable-LLM-Evals.code-workspace" "$HOME/vscode.KoyejoLab-Predictable-LLM-Evals.code-workspace"

# mv ...
ln -s "$AFS/vscode.pypantograph.code-workspace" "$HOME/vscode.pypantograph.code-workspace"
