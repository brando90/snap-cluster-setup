#!/bin/bash
# When you log in to snap it runs .bash_profile
# Since it doesn't let you edit that file to source YOUR real .bashrc configurations, you need to inspect it and observe it is sourcing ~/.bashrc
# Since it is sourcing `. ~/.bashrc  you need to create that file at ~ manually every time you want to use a new server
# (this is because the first time you use a server that file doesn't exist yet at ~ (i.e. $HOME), which points to the root of the lfs server your setting up)
# My trick is to create a single .bashrc at /afs/cs.stanford.edu/u/brand9/.bashrc
# Then manually create a soft link to point to the file every time I set up a new server e.g.,
# if the working version is on afs
# ln -s /afs/cs.stanford.edu/u/brando9/.bashrc $HOME/.bashrc
# or somehwere else:
# ln -s /lfs/mercury1/0/brando9/evals-for-autoformalization/.bashrc $HOME/.bashrc
# then do cd $HOME; ls -lah to confirm the soft link point to where you expect:
# e.g., .bashrc -> /afs/cs.stanford.edu/u/brando9/.bashrc
# confirm your at the server you expect
# hostname e.g., output mercury1.stanford.edu
# now you can put whatever you want in your .bashrc (located at afs and soft linked at ~ for this server)!
# e.g., you can activate the conda env and install the cuda versions you need want
# note: the cuda and conda versions you need might depend on the specific server since the specific cuda version might only work for the gpu that server has

# ---- Brando's Real .bashrc file ----
# since snap is set up badly and it needs the reauth command to re-authenticate randomly somtimes, you need to make sure reauth cmd is available
export PATH="/afs/cs/software/bin:$PATH"
# 
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Loads all readable shell scripts from /afs/cs/etc/profile.d/ if the current shell is not a login shell.
#if [ "x$SHLVL" != "x1" ]; then # We're not a login shell
	for i in /afs/cs/etc/profile.d/*.sh; do
		if [ -r "$i" ]; then
			. $i
		fi
	done
#fi

# Sets the maximum number of open file descriptors for the current process to 120,000.
ulimit -n 120000
# commented out so sourcing .bashrc doesn't output things to screen.
#ulimit -Sn # to check it's 120000
#ulimit -Hn # to check it's 120000

# - The defaul $HOME is /afs/cs.stanford.edu/u/brando9 but since you want to work in a specific server due to otherwise conda being so hard you need to reset what home is, see lfs: https://ilwiki.stanford.edu/doku.php?id=hints:storefiles#lfs_local_server_storage  
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD

# - set up afs short cuts
# since you are loged in to afs this moves you to your local computer
cd $HOME
export TEMP=$HOME

# -- Conda needs to be set up first before you can test the gpu & the right pytorch version has to be installed e.g., see: https://github.com/brando90/ultimate-utils/blob/45d8b2f47ace4d09ea63fe7cca7a7822b3af2961/sh_files_repo/download_and_install_conda.sh#L1C1-L31C32
# conda magic, unsure if needed but leaving here
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('$LOCAL_MACHINE_PWD/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "$LOCAL_MACHINE_PWD/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "$LOCAL_MACHINE_PWD/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="$LOCAL_MACHINE_PWD/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# put conda in PATH env where to find executable commands (so conda coommand works)
export PATH="$HOME/miniconda/bin:$PATH"
# activates base to test things
source $HOME/miniconda/bin/activate

# Note: since each server has a specific GPU you might have to set up a seperate conda env and cuda driver, but I think this works for all snap servers (tested on mercercy1, mercuery2, hyperturning1, skampere1)
# Check if the hostname is X, Y, Z to activate the right conda env with the right pytorch version for the current cuda driver
if [[ $(hostname) == "mercury1.stanford.edu" ]]; then
    # 
    # install the right version of pytorch compatible with cuda 11.7
    # pip3 install torch torchvision torchaudio
    # pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
    # export CUDA_VISIBLE_DEVICES=0
    # export PATH=/usr/local/cuda-11.7/bin:$PATH
    # export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
    conda activate align_4_af
else
    # install the right version of pytorch compatible with cuda 11.7
    # pip3 install torch torchvision torchaudio
    # pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
    # export CUDA_VISIBLE_DEVICES=0 
    # export PATH=/usr/local/cuda-11.7/bin:$PATH
    # export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
    # conda activate my_env
fi

# -- Optionally test pytorch with a gpu
# nvcc -V
# python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"

# -- If you use wandb you might need this:
export WANDB_API_KEY=TODO

# - Start this linux env with the gpu with most memory available
#export CUDA_VISIBLE_DEVICES=5
export LEAST_GPU_ID=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
export CUDA_VISIBLE_DEVICES=$LEAST_GPU_ID