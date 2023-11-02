# evals-for-autoformalization

## Get Compute for your Research Project

### Snap Cluster Important References
Always use the original documentation or wiki for each cluster: https://ilwiki.stanford.edu/doku.php?id=start (your **snap bible**).
Other useful resources:
- Support IT for snap: il-action@cs.stanford.edu
- compute instructions from Professor Koyejo's (Sanmi's) lab (STAIR Lab): https://docs.google.com/document/d/1PSTLJdtG3AymDGKPO-bHtzSnDyPmJPpJWXLmnJKzdfU/edit?usp=sharing
- advanced video from Rylan and Brando (made for the STAIR/Koyejo Lab): https://www.youtube.com/watch?v=XEB79C1yfgE&feature=youtu.be
- our CS 197 section channel
- join the snap slack & ask questions there too: https://join.slack.com/t/snap-group/shared_invite/zt-1lokufgys-g6NOiK3gQi84NjIK_2dUMQ

### Set up Project in the Snap Cluster - assuming a single Sever
Goal: Set up Project in the Snap Cluster - **assuming a single sever** (as if you were using a single computer, since that is most simple + good enough for essential goal (i.e., doing research)).
The details of how to use different storage systems are complicated and you don't need that version right now.
Note snap currently has no slurm (HPC workload manager).

## Get access to snap
Send an e-mail to professor Koyejo's (Brando's advisor) administrator Eric pined requesting for Snap access **for the quarter you are working with me**. 
CC my advisor (Sanmi) and me (Brando). 
The title email should be
> Compute access request Snap Cluster -- Working With Brando Miranda CS197 for this quarter only
the emails are:
- Eric Pineda: eric.pineda@stanford.edu
- Professor Sanmi Koyejo: sanmi@stanford.edu
- Brando Miranda: brando9@stanford.edu
We already have an advanced video for snap: https://youtu.be/XEB79C1yfgE .

## Setting up your compute environment in Snap
note: compute clusters usually have their official docs/wiki https://ilwiki.stanford.edu/doku.php?id=start always check those and if they don't work then the docs are wrong and you should e-mail the it for them to fix the docs il-action@cs.stanford.edu 

### SSH into cluster
First SSH in the cluster to use the server computer you were assigned:
```bash
#10 Quadro RTX 8000 48GB
ssh brando9@hyperturing1.stanford.edu
ssh brando9@hyperturing2.stanford.edu
#10 RTX A4000 16GB
ssh brando9@mercury1.stanford.edu
# ssh brando9@mercury2.stanford.edu
```
you run these commands from you computer.
You should see a bash shell in the cluster.
Sample output:
```bash
(data_quality) brandomiranda~ ❯ ssh brando9@mercury2.stanford.edu
Last login: Fri Oct 27 15:34:59 2023 from 172.24.69.154
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 mercury2.stanford.edu
 Ubuntu 20.04 (5.4.0-135-generic)
 96 x Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz, 503.55 GiB RAM, 2.00 GiB swap

 eno1: 172.24.75.55
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
   -> For help or to report problems, go to http://support.cs.stanford.edu

brando9@mercury2:~$
```
This might look different if you already set up your .bashrc file. This is what we will go set up next.

Tip, since the server's have weird security persmission you might need to reauth often (or your programs are killed by the admins/IT's):
```bash
reauth
```
type password. If the reauth command doesn't work do:
```bash
export PATH="/afs/cs/software/bin:$PATH"
```
Consider adding that to your `.bashrc`, but my suggested `.bashrc` should already have it.

### Setting up your bashrc file in Snap
tip: anything you don't understand discuss with GPT4/Claude! Highly recommend it. I do it often and save the conv links or convs in my evernote. e.g., ask it what an env variable is. Or what vim is. Or what git clone is. etc.

Every time you login to server or open a linux terminal, you need to configure your unix/linux environment 
(i.e., set up environment variables that your bash shell uses to figure out where things are, like commands etc.)

Usually the linux terminal runs `.bash_profile` to set up your linux environment. 
In this case if you inspect it with cat `.bash_profile` you can see it runs ("sources") another file called `.bashrc` i.e.,
```bash
# note ~ is at AFS path before we changed it to the LFS for your server
brando9@mercury2:~ $ pwd .bash_profile
/afs/cs.stanford.edu/u/brando9
brando9@mercury2:~ $ realpath .
/afs/cs.stanford.edu/u/brando9
brando9@mercury2:~$ cat .bash_profile
# DO NOT MODIFY THIS FILE! Your changes will be overwritten!
# For user-customization, please edit ~/.bashrc.user
#
# $ROOT$
#
# CSD-CF miles 2003-01-24

# Don't do anything in this file -- do it in .bashrc

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi
```
So it means we need to put our personal linix configurations in ~/.bashrc

First echo `$HOME` to figure out what home path you're using (since the .bashrc is located at `~` which is `$HOME`):
```bash
brando9@mercury2:~$ echo $HOME
/afs/cs.stanford.edu/u/brando9
```
So this means we need our bash configurations at `~/.bashrc` i.e., to be at `$HOME/.bashrc` (`~` means `$HOME`).
So first let's create that file with vim (see basic Vim bellow in this tutorial to know the basics):
```bash
# note I used the absolute path because we will have $HOME (i.e., tilde) point to the local (lfs) home directory.
cd /afs/cs.stanford.edu/u/brando9
vim .bashrc
```
Now go to this https://github.com/brando90/evals-for-autoformalization/blob/33e1bf825f7de5d49557256332355fa2872bbbd7/.bashrc#L1 and copy paste it to your clip board e.g., with command/control + c. 
Now press `i` to go in vim's insert mode and paste the file as you'd normally e.g., with control/command + v. 
You can read through the `.bashrc` to see why it's set up the way it is.
Then press `esc` to `:w` enter to save the file. Then press `:q` enter to exist (or `:x` enter for for save & exit).
Note, this is (most likely) correct even though the wiki/docs for snap say to update `.bash.user` (but .bash.user is never sourced, I asked the it and I strongly recommend you ask too, see wrong/confusing docs if you want https://ilwiki.stanford.edu/doku.php?id=hints:enviroment but that's not what `.bash_profile` is sourcing!?).

The goal will be to have all your files live at the storage space for your local server (LFS, stands for local file server).
Therefore, let's move this .bashrc, $HOME (~) and all your git close at the home of lfs for your assinged server.

#### Move .bashrc to your LFS
First let's create a soft link to your .bashrc at your lfs $HOME.
Creete that environment variable
```bash
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
export HOME=$LOCAL_MACHINE_PWD
echo $HOME
cd ~

# the full output
brando9@mercury2:~$ pwd
/afs/cs.stanford.edu/u/brando9
brando9@mercury2:~$ export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
brando9@mercury2:~$ export HOME=$LOCAL_MACHINE_PWD
brando9@mercury2:/afs/cs.stanford.edu/u/brando9$ echo $HOME
/lfs/mercury2/0/brando9
brando9@mercury2:/afs/cs.stanford.edu/u/brando9$ cd ~
brando9@mercury2:~$ pwd
/lfs/mercury2/0/brando9
```
the last line confirms we are at the local servers storage (called lfs).

If you look at the .bashrc there is this line:
```bash
# - The defaul $HOME is /afs/cs.stanford.edu/u/brando9 but since you want to work in a specific server due to otherwise conda being so hard you need to reset what home is, see lfs: https://ilwiki.stanford.edu/doku.php?id=hints:storefiles#lfs_local_server_storage  
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD

# - set up afs short cuts
# since you are logged in to afs this moves you to your local computer
cd $HOME
```
meaning that every time you login to the server assigned you got to your lfs directory instead of the afs home directory (since `$HOME` was changed).

### Git clone
Now that you have a sensible `.bashrc` file that cd's you to your local server's lfs storage, it's time to git clone your project, conda install all of the project's depedencies and pip install the project.

First you need set up an SSH keys in your lfs server individually.
For that see the bellow instructions for SSH with sample outputs of the termianl.

After that works first check you are in your server's lfs home:
```bash
(data_quality) brandomiranda~ ❯ ssh brando9@mercury2.stanford.edu
Last login: Fri Oct 27 18:34:13 2023 from 172.24.69.154
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 mercury2.stanford.edu
 Ubuntu 20.04 (5.4.0-135-generic)
 96 x Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz, 503.55 GiB RAM, 2.00 GiB swap

 eno1: 172.24.75.55
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
   -> For help or to report problems, go to http://support.cs.stanford.edu

brando9@mercury2~ $ pwd
/lfs/mercury2/0/brando9
brando9@mercury2~ $ realpath .
/lfs/mercury2/0/brando9
```
we are at lfs! Good. 
Now make sure you have a **team** fork of your project's repo e.g., project links:
- https://github.com/brando90/evals-for-autoformalization/tree/main
- https://github.com/brando90/beyond-scale-language-data-diversity/tree/main
click the git forke and then go to settings, collaborators and choose one of the forks to be the team's fork where everyone pushes the git changes to the project.
Assuming each person's SSH keys for their server and your github correct as the instructions bellow, create a **local copy in your server's lfs (git clone) of the tema github fork** i.e.,: 
```bash
cd $HOME
git clone git@github.com:brando90/beyond-scale-language-data-diversity.git
# or
git clone git@github.com:brando90/evals-for-autoformalization.git
# sample output
brando9@mercury2~ $ cd $HOME
brando9@mercury2~ $ realpath .
/lfs/mercury2/0/brando9
brando9@mercury2~ $ git clone git@github.com:brando90/evals-for-autoformalization.git
Cloning into 'evals-for-autoformalization'...
remote: Enumerating objects: 431, done.
remote: Counting objects: 100% (288/288), done.
remote: Compressing objects: 100% (164/164), done.
remote: Total 431 (delta 166), reused 220 (delta 118), pack-reused 143
Receiving objects: 100% (431/431), 15.77 MiB | 14.41 MiB/s, done.
Resolving deltas: 100% (230/230), done.
```
always make sure to read carefully the output of the commands you run in your terminal/cli.
Now that you have cloned your team's fork, we can install conda and test the GPUs!

### Install conda
Now that you have a team's fork and git cloned your project's repo let's install it with `pip install -e .` with a conda environment!
First check everything looks good.
```bash
brando9@mercury2~ $ realpath .
/lfs/mercury2/0/brando9
brando9@mercury2~ $ ls -lah
total 40K
drwxr-xr-x  6 brando9 users 4.0K Oct 27 18:40 .
drwxrwxrwt 14 root    root  4.0K Oct 27 15:10 ..
-rw-------  1 brando9 users   55 Oct 27 16:03 .bash_history
lrwxrwxrwx  1 brando9 users   38 Oct 27 15:10 .bashrc -> /afs/cs.stanford.edu/u/brando9/.bashrc
lrwxrwxrwx  1 brando9 users   67 Oct 27 15:10 beyond-scale-language-data-diversity -> /afs/cs.stanford.edu/u/brando9/beyond-scale-language-data-diversity
drwxr-xr-x  3 brando9 users 4.0K Oct 27 15:35 .cache
drwxr-xr-x  3 brando9 users 4.0K Oct 27 15:10 data
lrwxrwxrwx  1 brando9 users   80 Oct 27 15:10 diversity-for-predictive-success-of-meta-learning -> /afs/cs.stanford.edu/u/brando9/diversity-for-predictive-success-of-meta-learning
drwxr-xr-x  6 brando9 users 4.0K Oct 27 18:40 evals-for-autoformalization
lrwxrwxrwx  1 brando9 users   49 Oct 27 15:10 iit-term-synthesis -> /afs/cs.stanford.edu/u/brando9/iit-term-synthesis
lrwxrwxrwx  1 brando9 users   35 Oct 27 15:10 keys -> /afs/cs.stanford.edu/u/brando9/keys
drwx------  5 brando9 users 4.0K Oct 27 15:35 .local
lrwxrwxrwx  1 brando9 users   60 Oct 27 15:10 massive-autoformalization-maf -> /afs/cs.stanford.edu/u/brando9/massive-autoformalization-maf
lrwxrwxrwx  1 brando9 users   36 Oct 27 15:10 pycoq -> /afs/cs.stanford.edu/u/brando9/pycoq
lrwxrwxrwx  1 brando9 users   45 Oct 27 15:10 ultimate-pycoq -> /afs/cs.stanford.edu/u/brando9/ultimate-pycoq
lrwxrwxrwx  1 brando9 users   45 Oct 27 15:10 ultimate-utils -> /afs/cs.stanford.edu/u/brando9/ultimate-utils
brando9@mercury2~ $ cd evals-for-autoformalization/
brando9@mercury2~/evals-for-autoformalization $ pwd
/lfs/mercury2/0/brando9/evals-for-autoformalization
```
now we need to install the conda package manager using this cript https://github.com/brando90/ultimate-utils/blob/master/sh_files_repo/download_and_install_conda.sh. 
Check it's not there:
```bash
brando9@mercury1:/lfs/mercury1/0/brando9/evals-for-autoformalization$ conda
conda: command not found
```
Conda manages different python environments for different projects.
This is so that different depdencies between different python projects don't clash.
It also helps you be more organized.
For that execute one line at a time of the file above https://github.com/brando90/ultimate-utils/blob/master/sh_files_repo/download_and_install_conda.sh
But here are the main commands:
```bash
# RUN ALL THE INSTRUCTIONS! PLEASE!
echo $HOME
cd $HOME
# -- Install miniconda
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
# - Set up conda
conda init
# conda init zsh
conda init bash
conda install conda-build
conda update -n base -c defaults conda
conda update conda

# - Create conda env
conda create -n my_env python=3.10
conda activate my_env
## conda remove --name my_env --all

# - Make sure pip is up to date
which python
pip install --upgrade pip
pip3 install --upgrade pip
which pip
which pip3
``` 
at the end you should be able to run the conda command after putting this somewhere in your `$HOME/.bashrc`:
```bash
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
```
Since I already put it in a good place (hopefully) for us, you don't need to do anything except restart your bash shell
(or re-ssh into the server):
```bash
brando9@mercury1:/lfs/mercury1/0/brando9/evals-for-autoformalization$ conda
conda: command not found
brando9@mercury1:/lfs/mercury1/0/brando9/evals-for-autoformalization$ bash
(base) brando9@mercury1~ $ which conda
/lfs/mercury1/0/brando9/miniconda/bin/conda
```
We can already see something is different with the prefix `(base)`, but `which conda` confirms which conda we are using (it tells us the location of the binary for the command conda).

Now we can create a conda env for our project:
```bash
(base) brando9@mercury1~ $ conda create -n my_env python=3.10
(base) brando9@mercury1~ $ conda activate my_env
(my_env) brando9@mercury1~ $  
```
you can confirm you create and are in the right conda env with (my_env) prefix but you can also see all the conda envs you have set up so far:
```bash
(my_env) brando9@mercury1~ $ conda info -e
# conda environments:
#
base                     /lfs/mercury1/0/brando9/miniconda
align_4_af               /lfs/mercury1/0/brando9/miniconda/envs/align_4_af
my_env                *  /lfs/mercury1/0/brando9/miniconda/envs/my_env
```
### Now that you have conda working let's test the GPU
I am putting this first before installing your projects depedencies because it guarantees we have the right version of pytorch that works in gpu.
But sometimes some other depedency could mean you need to change your pytorch cuda compatible version and the cuda driver. 
But you will learn what those are here and fix them if you need by returning here.

So install pytorch with gpu https://pytorch.org/get-started/locally/. That is the official way to do it. 
But all these options worked for me:
```bash
pip3 install torch torchvision torchaudio
# these last too are explicit on the torch and cuda version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```
then tell your linux environment where the installation of the cuda driver is:
```bash
# - https://ilwiki.stanford.edu/doku.php?id=hints:gpu
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```
Now let's run a bunch of pytorch commands the require gpu here in the terminal to test pytorch + cuda/gpu:
```bash
(my_env) brando9@mercury1~ $ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
(my_env) brando9@mercury1~ $ python -c "import torch; print(torch.cuda.get_device_capability())"
(8, 6)
(my_env) brando9@mercury1~ $ python -c "import torch; print(torch.bfloat16);"
torch.bfloat16
(my_env) brando9@mercury1~ $ python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"
tensor([[3.7820],
        [0.1884]], device='cuda:0')
```
success! It worked :) Gpu and pytorch versions are working well together.

### Install your project!
Now that you have conda let's install the depedencies needed for the project.
Cd to your project
```bash
(my_env) brando9@mercury1~ $ ls -l
total 117972
lrwxrwxrwx  1 brando9 users        67 Oct 19 14:26 beyond-scale-language-data-diversity -> /afs/cs.stanford.edu/u/brando9/beyond-scale-language-data-diversity
drwxr-xr-x  3 brando9 users      4096 Jun 22 18:23 data
lrwxrwxrwx  1 brando9 users        80 Jun 22 18:23 diversity-for-predictive-success-of-meta-learning -> /afs/cs.stanford.edu/u/brando9/diversity-for-predictive-success-of-meta-learning
drwxr-xr-x  6 brando9 users      4096 Oct 27 15:36 evals-for-autoformalization
-rw-r--r--  1 brando9 users       133 Oct 27 12:40 evals-for-autoformalization.mercurcy1-code-workspace.code-workspace
lrwxrwxrwx  1 brando9 users        49 Jun 22 18:23 iit-term-synthesis -> /afs/cs.stanford.edu/u/brando9/iit-term-synthesis
lrwxrwxrwx  1 brando9 users        35 Oct 19 14:26 keys -> /afs/cs.stanford.edu/u/brando9/keys
lrwxrwxrwx  1 brando9 users        60 Oct 19 14:26 massive-autoformalization-maf -> /afs/cs.stanford.edu/u/brando9/massive-autoformalization-maf
drwxr-xr-x 19 brando9 users      4096 Oct 23 19:21 miniconda
-rw-r--r--  1 brando9 users 120771089 Oct 19 08:49 miniconda.sh
lrwxrwxrwx  1 brando9 users        44 Jun 22 18:23 proverbot9001 -> /afs/cs.stanford.edu/u/brando9/proverbot9001
lrwxrwxrwx  1 brando9 users        36 Jun 22 18:23 pycoq -> /afs/cs.stanford.edu/u/brando9/pycoq
lrwxrwxrwx  1 brando9 users        45 Jun 22 18:23 ultimate-pycoq -> /afs/cs.stanford.edu/u/brando9/ultimate-pycoq
lrwxrwxrwx  1 brando9 users        45 Jun 22 18:23 ultimate-utils -> /afs/cs.stanford.edu/u/brando9/ultimate-utils

# cd
cd evals-for-autoformalization
# or
cd beyond-scale-language-data-diversity
# to make it concrete
(my_env) brando9@mercury1~ $ cd evals-for-autoformalization
(my_env) brando9@mercury1~/evals-for-autoformalization $ 
```
then do a pip editable install with `pip install -e .`.
Note `.` is the location of your project's root where the `setup.py` is. 
This is needed I becuase if you do only `pip install .` then when you edit the project's python files, it will not be reflected in conda and you won't be running/using your new code:
```bash
(my_env) brando9@mercury1~ $ cd evals-for-autoformalization
(my_env) brando9@mercury1~/evals-for-autoformalization $ pip install -e .
Obtaining file:///lfs/mercury1/0/brando9/evals-for-autoformalization
  Preparing metadata (setup.py) ... done
Requirement already satisfied: dill in /lfs/mercury1/0/brando9/miniconda/envs/my_env/lib/python3.10/site-packages (from evals-for-autoformalization==0.0.1) (0.3.7)
... 
Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /lfs/mercury1/0/brando9/miniconda/envs/my_env/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard->evals-for-autoformalization==0.0.1) (0.5.0)
Requirement already satisfied: oauthlib>=3.0.0 in /lfs/mercury1/0/brando9/miniconda/envs/my_env/lib/python3.10/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard->evals-for-autoformalization==0.0.1) (3.2.2)
Installing collected packages: evals-for-autoformalization
  Attempting uninstall: evals-for-autoformalization
    Found existing installation: evals-for-autoformalization 0.0.1
    Uninstalling evals-for-autoformalization-0.0.1:
      Successfully uninstalled evals-for-autoformalization-0.0.1
  Running setup.py develop for evals-for-autoformalization
Successfully installed evals-for-autoformalization-0.0.1
```
then this should mean your project is correct installed! 

### Testing the main script you are developing
Now you should install vscode in your computer.
Download the SSH extension in vscode.
Then as we did in seciton (see video https://youtu.be/dzMAtTGplGg or https://youtu.be/8PBL-qcJWZw I think the first one has the vscode part with the SSH extension) create a host to conect to the server you are using for snap. 
After that your vscode should editing files from the server directly. 
You can do `File > Add Folder to Workspace` so that it appears on your explorer window on vscode.
Then after adding the root of your project e.g.,
```
# added via vscode 
/lfs/mercury1/0/brando9/evals-for-autoformalization
# added via vscode
/lfs/mercury1/0/brando9/beyond-scale-language-data-diversity
```
Now, open your main projects file and edit it in vscode. 
Then save it.
Now let's run the hellopy and your main project's file it to test the changes:
```
# test hello.py
python /lfs/mercury1/0/brando9/evals-for-autoformalization/src/hello.py
python ~/evals-for-autoformalization/src/hello.py
(my_env) brando9@mercury1~/evals-for-autoformalization $ python ~/evals-for-autoformalization/src/hello.py
hello (Test your python pip install -e . by printing hello)

(8, 6)
torch.bfloat16
tensor([[0.6177],
        [2.2084]], device='cuda:0')
```
then test the main project's file
```bash
# - prover based eval project (runs the ppl eval)
# sanity check full pipeline of eval works using ppl
python ~/evals-for-autoformalization/src/nlp_eval/af_ppl_eval.py
# sanity check full pipeline of eval works for your prover based model
python ~/evals-for-autoformalization/src/prover_based_evals/af_lean_tactic_eval.py
python ~/evals-for-autoformalization/src/prover_based_evals/af_re_prover_eval.py

# - diversity project
# sample scrip to train a model, goal is to reproduce the results form beyond scale and train for very long or for a deeper network e.g., falcon-1B is good (but initialize it from scratch)
python ~/beyond-scale-language-data-diversity/src/train/train.py

# - alignment af-fine-tuning
# test that the alignment code outputs sensible values according to the sanity check
python ~/beyond-scale-language-data-diversity/src/alignment/align.py
# fine tuning script for AF, test if fine tuning in aligned or not aligned improves test ppl most assuming same trained tokens (fair comparison)
python ~/beyond-scale-language-data-diversity/src/alignment/fine_tuning_with_aligned_data.py
```

## Running long running jobs
Due to very unstandard set up of snap, your long running processes will be killed if they do not (kerberos) reauthenticate (that it's you).
So the practice to run long processes/jobs in snap is:
1. Run a tmux session (or kerberos tmux `krbtmux`) (read about regular (tmux)[https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/]).
```bash
# note this is not the standard tmux command tmux
krbtmux
```
2. Then run the reauthentication daemon/backgrou process command `reuath` and type your password, so that your tmux session is not killed
```bash
reauth
```
3. Then you can run whatever long process you want e.g., your experiment script:
```bash
python ~/beyond-scale-language-data-diversity/src/alignment/fine_tuning_with_aligned_data.py
```
once inside your temux sessions.

Usually, you will want to "go out" of your tmux session.
This is done with keyboard scroke "tmuyx prefix + d" which ends up being `contro b` follewed by `d`. 
This returns you to your normal bash terminal.

Then you can see which tmux sessions you have with:
```bash
tmux ls
```
and you can create a new tmux session with:
```bash
tmux new -t <tmux_session_name>
# e.g.,
tmux new -s 0
```
and you can return to see how your script is doing with (assuming you login from scratch to your server again)
```bash
tmux attach -t <tmux_session_name>
# e.g.
tmux attach -t 0
```
- ref: snap's tutorial on long running prcesses https://ilwiki.stanford.edu/doku.php?id=hints:long-jobs
- ref: read about tmux https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/

# Vscode

## SSH remote extension

## Python Environment (e.g., venv, conda, poetry)
If you want to be able to run the script or the debugger from within python you need to tell vscode the path to the python environment you are using (you need python envs to manage different dependencies between different projects to avoid conflict. If you are unsure why you need python envs, ask ChatGPT, likely focusing on conda & python).
To tell vscode first figure out where the python interpreter is
```bash
(evals_af) brando9@skampere1~ $ which python
/lfs/skampere1/0/brando9/miniconda/envs/evals_af/bin/python
```
then press Command + Shift + p to get vscode's command pallete.
Then you should have a `>` and start typing 
```
> Python: Select Interpreter
```
press enter. Then type in vscode's command window:
```
select at workspace level
```
then type
```
select interpreter path
```
then copy paste the path you got above:
```
/lfs/skampere1/0/brando9/miniconda/envs/evals_af/bin/python
```
your python env should be working now when you run the file or run the debugger from vscode (even from the ssh extension! which means you can debug your code using GPUs! The real env for your experiments which likely decreases iterations for coding!).
You can also visually check the bottm left corner with the name of your env, in this case `evals_af`.

## Debugger

# Other

TODO: write a nice readme with commands demoing how to use snap.

Need to know let's decide later where to put this in the intructions:
- .bashrc + .bashrc details of snap: https://github.com/brando90/.dotfiles 

Bonus:
- kinit for avoiding passwords
- request an optional pull rquest to the original repo
- ampere arch fp32 vs fp 16 and bf16. The goods for ML are bf16 and fp32.

TODO: carlos, why isn't conda available in snap?

note: you should understand (roughly) what everything means in here to be effective. Google, gpt4/claude it etc. 
Tips:
- use `man` to understand bash command or if you want to chat with it use LLMs/GPT4/Claude and `--help` or `-h`.

List of thinks to know about:
- git, ssh, bash,
- basic unix commands, ls, ls -lah, cd, pwd, which,
- vim
- nvidia-smi
- module load (common in HPC's)

## SSH
Goal: add the public key you created on sherlock's login node to your github so you can clone your fork. For that follow the instructions here https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account or the outline from bellow that was inspired from the official github link in this sentence.

First create ssh public key on sherlock
```bash
ssh your_sunetid@login.sherlock.stanford.edu
[brando9@sh03-ln06 login ~/.ssh]$ ssh-keygen -t ed25519 -C "brandojazz@gmail.com"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/users/brando9/.ssh/id_ed25519):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/users/brando9/.ssh/id_ed25519.
Your public key has been saved in /home/users/brando9/.ssh/id_ed25519.pub.
The key fingerprint is:
...
The key's randomart image is:
+--[ED25519 256]--+
...
+----[SHA256]-----+
# press the enter key to not change file name
# press the enter key or a passphase to use this key
```
Now run ssh agent in sherlock
```
[brando9@sh03-ln06 login ~/.ssh]$ eval "$(ssh-agent -s)"
Agent pid 50895
```
Now configure your .ssh if you've never done it on this server.
Concretely, if ~.ssh/config doesn't exist create it with (or vim): 
```
touch ~/.ssh/config
# or
[brando9@sh03-ln06 login ~/.ssh]$ vim .config
```
put the contets of for hithub (i.e., copy the bellow into your clip board, read it) with the vim:
```
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```
i.e. use vim editor in sherlock(read about vim, it's just an edit) in the server i.e.
do
```
[brando9@sh03-ln06 login ~/.ssh]$ cat ~/.ssh/config
cat: /home/users/brando9/.ssh/config: No such file or directory
vim ~/.ssh/config
# press i in the new black window,
#copy paste the contents above after pressing i,
#then press escape esc
# then safe the file and exist with :x or :w followed by :q
# then do 
cat .config
# confirms you copied it correctly
[brando9@sh03-ln06 login ~/.ssh]$ cat .config
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```
Then add the key to your github using https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account . For a summary of what I did do:
```
# in the sherlock login/head node do:
[brando9@sh03-ln06 login ~/.ssh]$ cat ~/.ssh/id_ed25519.pub
# then copy paste the output, very carefully, do not share this stuff publicly wide web
```
Then go to setting in your github e.g., https://github.com/settings/keys and create a new key by copy pasting the contents of the previous cat command.

Then git clone on your fork should work, e.g.,:
```
[brando9@sh03-ln06 login ~/.ssh]$ git clone git@github.com:brando90/evals-for-autoformalization.git
Cloning into 'evals-for-autoformalization'...
remote: Enumerating objects: 270, done.
remote: Counting objects: 100% (264/264), done.
remote: Compressing objects: 100% (163/163), done.
remote: Total 270 (delta 150), reused 175 (delta 90), pack-reused 6
Receiving objects: 100% (270/270), 78.74 KiB | 0 bytes/s, done.
Resolving deltas: 100% (151/151), done.
```

## Tutorial on setting up a python project
1. create the `setup.py` file
2. Make sure your setup.py file has the following
```python
    package_dir={'': 'src'},
    packages=find_packages('src'),
```
so that `pip install -e .` knows were the python modules are when installing the python library. 
Anything outside `src`` won't be found for this libraries pip -e install.
Read the comments for those lines in `setup.py`` to understand it if you wish and refs.
3. Now you can do `pip install -e .` or `pip install -e $HOME/evals-for-autoformalization` (assuming you have your python env/conda env activated).

Now you should be able to import statements for this library in the expected way!

## Python Envs with conda & using pip instlal -e <path>

```bash
# This script demonstrates the use of conda for managing Python environments and pip for installing Python packages.
# Conda is an open-source package management and environment management system.

# 1. List all the available conda environments on your system.
# The command 'conda info -e' will list all the environments available.
conda info -e

# 2. Update conda to the latest version.
# It's good practice to keep conda updated to the latest version to avoid any compatibility issues.
conda update --all

# 3. Upgrade pip to the latest version.
# Pip is the package installer for Python. Upgrading it ensures that you can install packages without issues.
pip install --upgrade pip

# 4. Create a new conda environment.
# 'conda create -n maf python=3.10' creates a new environment named 'maf' with Python version 3.10 installed.
# '-n maf' specifies the name of the environment, and 'python=3.10' specifies the Python version.
conda create -n af_evals python=3.10

# 5. Activate the newly created conda environment.
# 'conda activate maf' activates the 'maf' environment. Once activated, any Python packages installed will be specific to this environment.
conda activate af_evals

# To deactivate the current environment and return to the base environment, you can use:
# conda deactivate

# If you want to remove the 'maf' environment completely, you can use:
# conda remove --name af_evals --all

# 6. Install Python packages using pip in editable mode.
# 'pip install -e <path>' installs a package in 'editable' mode, meaning changes to the source files will immediately affect the installed package without needing a reinstall.
# Replace '<path>' with the path to the directory containing the 'setup.py' file of the package you want to install.
# pip install -e $HOME/evals-for-autoformalization
pip install -e .

# Test pytorch install with GPU
python -c "import torch; print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"

# Note: It's crucial to activate the correct conda environment before using pip install to avoid installing packages in the wrong environment.
```
ref: https://chat.openai.com/c/375d5d26-7602-4888-9ef5-9f92359330dc

## Basic Git
```bash
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/massive-autoformalization-maf.git
ln -s /afs/cs.stanford.edu/u/brando9/massive-autoformalization-maf $HOME/massive-autoformalization-maf
pip install -e ~/massive-autoformalization-maf
#pip uninstall ~/massive-autoformalization-maf
cd ~/massive-autoformalization-maf
```

## Basic vim
Vim is for editing file in the terminal (or cli). These are the commands you might need https://coderwall.com/p/adv71w/basic-vim-commands-for-getting-started
Mostly knowing that:
-> pressing `i` puts you in insert mode
-> pressing `w` writes/save the files
-> pressing `q` quits
that's all you need.
Note: ask GPT4/Claude for help if your stuck.

## Lean4
Recommend watching:
- https://www.youtube.com/watch?v=yZo6k48L0VY
- https://www.youtube.com/watch?v=_0QZXHoyZlA 
