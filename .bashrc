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

# ---- Brando's scaffold .bashrc file ----

# - Approximately source the sys admin's .bashrc (located at /afs/cs/etc/skel/.bashrc)
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
export AFS=/afs/cs.stanford.edu/u/brando9
export DFS=/dfs/scratch0/brando9/

# - prompt colours
BLACK='\e[0;30m'
RED='\e[0;31m'
GREEN='\e[0;32m'
BROWN='\e[0;33m'
BLUE='\e[0;34m'
PURPLE='\e[0;35m'
CYAN='\e[0;36m'
LIGHT_GREY='\e[0;37m'
DARK_GREY='\e[1;30m'
LIGHT_RED='\e[1;31m'
LIGHT_GREEN='\e[1;32m'
YELLOW='\e[1;33m'
LIGHT_BLUE='\e[1;34m'
LIGHT_PURPLE='\e[1;35m'
LIGHT_CYAN='\e[1;36m'
WHITE='\e[1;37m'

BACK_DEFAULT_COLOR='\e[m'

HOST_PART='$(hostname | cut -d. -f1)'
export PS1="\[$LIGHT_GREY\]\u@$HOST_PART\[$LIGHT_GREEN\]\w\[$LIGHT_GREY\] \$ \[$LIGHT_CYAN\]"

# -- If you use wandb you might need this:
export WANDB_API_KEY=TODO
export HF_TOKEN='your_token_here'
export OPENAI_KEY='your_openai_key_here'

# - Start this linux env with the gpu with most memory available
export CUDA_VISIBLE_DEVICES=5
export LEAST_GPU_ID=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | awk '{print NR-1 " " $1}' | sort -nk2 | head -n1 | cut -d' ' -f1)
export CUDA_VISIBLE_DEVICES=$LEAST_GPU_ID

# -- Lean, ref: https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#lean-in-snap
export PATH="$HOME/.elan/bin:$PATH"

# # not needed, if issues see: https://ilwiki.stanford.edu/doku.php?id=hints:gpu
# export PATH=/usr/local/cuda-11.7/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

# -- poetry, we are setting up poetry before conda so that we don't accidentally use poetry's python version, see: https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#poetry
# assumes mkdir $HOME/.virtualenvs has been ran
export VENV_PATH=$HOME/.virtualenvs/venv_for_poetry
# assume poetry has been installed as explained here: https://github.com/brando90/snap-cluster-setup?tab=readme-ov-file#poetry
export PATH="$VENV_PATH/bin:$PATH"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$HOME/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# very similar to the conda init code above, will leave it out since snap seems to be working e.g., it can find conda command just fine with the above code, note above code **does** have to be at the end
# export PATH="$HOME/miniconda/bin:$PATH"

# activates the conda base env
source $HOME/miniconda/bin/activate

# activate snap_cluster_setup default conda env
conda activate snap_cluster_setup_py311
