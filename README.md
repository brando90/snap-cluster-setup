# evals-for-autoformalization

## Get Compute for your Research Project

### Snap Cluster Important References & Help/Support
Always use the original documentation or wiki for each cluster: https://ilwiki.stanford.edu/doku.php?id=start -- your **snap bible**.
Other useful resources:
- Support IT for snap: il-action@cs.stanford.edu (don't be shy to ask them question or help for SNAP.)
- compute instructions from Professor Koyejo's (Sanmi's) lab (STAIR Lab): https://docs.google.com/document/d/1PSTLJdtG3AymDGKPO-bHtzSnDyPmJPpJWXLmnJKzdfU/edit?usp=sharing
- advanced video from Rylan and Brando (made for the STAIR/Koyejo Lab): https://www.youtube.com/watch?v=XEB79C1yfgE&feature=youtu.be
- our CS 197 section channel
- join the snap slack & ask questions there too: https://join.slack.com/t/snap-group/shared_invite/zt-1lokufgys-g6NOiK3gQi84NjIK_2dUMQ

## Get access to snap and requesting CSID
First create a CSID here and  please make your CSID the same as your Stanford SUNET id. 
Request it here:  https://webdb.cs.stanford.edu/csid

To get access to snap write an e-mail with this subject:

> Access Request Snap Cluster Working With Brando Miranda CS197 for <full_name> <CSID>  <SUNET>

For example: 

> Access Request Snap Cluster Working With Brando Miranda CS197 for Brando Miranda brando9 brando9$ 

and sent it to
- Eric Pineda: eric.pineda@stanford.edu
- Brando Miranda: brando9@stanford.edu
- [Snap cluster IT](https://ilwiki.stanford.edu/doku.php?id=start): il-action@cs.stanford.edu
- Sanmi Koyejo: sanmi@stanford.edu

Once you get access to it you will be able to login to the cluster via the ssh command in the terminal (cli) and vscode.

## SSH into a Sanmi owned node/server
SSH into a SNAP server/node owned by Sanmi's Lab (`skampere1`, `mercury1`, `mercury2`) directly with your csid/sunet e.g.,:
```bash
# Example
ssh brando9@skampere1.stanford.edu

# Yours
ssh CSID@skampere1.stanford.edu
```
type your password and login. 
Then use the `ls` command and `pwd` and `realpath ~` and `realpath $HOME` and `realpath .` commands to see the contents of your root `~` directory.
Do the `cd ..` and `pwd` to be familiar with the file system/directory structure in snap.
Then run the `reauth` command, this command is needed due to the (kerberos) security settings in SNAP (hote: Brando strongly disagrees with how IT set up this part and apologizes for the complexity for you).
[For more info on reauth read the wiki here](https://ilwiki.stanford.edu/doku.php?id=hints:long-jobs).

**Tip**: see the wiki for other nodes that exist on snap!

Sample output if it worked:
```bash
(base) brandomiranda~ ❯ ssh brando9@skampere1.stanford.edu
brando9@skampere1.stanford.edu's password:
Last login: Wed Apr  3 12:07:20 2024 from 172.24.69.154
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 skampere1.stanford.edu
 Ubuntu 20.04 (5.4.0-162-generic)
 128 x AMD EPYC 7543 32-Core Processor, 1.96 TiB RAM, 8.00 GiB swap

 enp44s0f0: 172.24.75.194
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
   -> For help or to report problems, go to http://support.cs.stanford.edu

ln: failed to create symbolic link '/lfs/skampere1/0/brando9/iit-term-synthesis': File exists
(evals_af) brando9@skampere1~ $
(evals_af) brando9@skampere1~ $ realpath ~
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ realpath $HOME
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ realpath .
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ ls
beyond-scale-language-data-diversity			  massive-autoformalization-maf       ultimate-anatome
data							  miniconda			      ultimate-pycoq
diversity-for-predictive-success-of-meta-learning	  miniconda.sh			      ultimate-utils
evals-for-autoformalization				  nltk_data			      vscode.beyond-scale-language-data-diversity.skampere1.code-workspace
evaporate						  proverbot9001			      vscode.evaporate.skampere1.code-workspace
iit-term-synthesis					  putnam-math			      vscode.lean4ai.skampere1.code-workspace
keys							  pycoq				      vscode.maf_data.creating_data_math_training.skamapere1.code-workspace
_KoyejoLab-Predictable-LLM-Evals			  snap-cluster-setup		      vscode.maf_data.training_af_model.skampere1.code-workspace
KoyejoLab-Predictable-LLM-Evals				  tempt.txt			      vscode.snap-cluster-setup.code-workspace
KoyejoLab-Predictable-LLM-Evals.skampere1.code-workspace  test_folder			      wandb
lean4ai							  the-data-quality-edge-for-top-llms
(evals_af) brando9@skampere1~ $ pwd
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ cd ..
(evals_af) brando9@skampere1/lfs/skampere1/0 $ pwd
/lfs/skampere1/0
(evals_af) brando9@skampere1/lfs/skampere1/0 $ cd ~
(evals_af) brando9@skampere1~ $
(evals_af) brando9@skampere1~ $ reauth
Password for brando9:
Background process pid is: 4145964
```
Note: this might look slightly different if you already set up your `.bashrc` file. 

**Tip**: If the `reauth `command doesn't work do or/and e-mail the [Snap cluster IT](https://ilwiki.stanford.edu/doku.php?id=start) il-action@cs.stanford.edu:
```bash
export PATH="/afs/cs/software/bin:$PATH"
```
**TIP**: Ask ChatGPT what `export PATH="/afs/cs/software/bin:$PATH"` does. ChatGPT is excellent at the terminal and bash commands. Note consider adding this to your `.bashrc` for example [see this](https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc#L24). 
**TIP**: you might have to ssh into your node again outside of vscode for this to work if vscode is giving you permission issues or e-mail snap IT. 

## Setup your .bashrc and redirect the $HOME (~) envi variable from afs to lfs 
### lfs, dfs, afs
Rationale: [Snap has 3 file systems afs, lfs, dfs](https://ilwiki.stanford.edu/doku.php?id=hints:storefiles) (folders where your files, data and code could potentially be stored). 
We will only be using `afs` and `lfs`. 
`dsf` stands for distributed file system and it makes your files available with all nodes/servers/computers in the snap cluster but it's too slow to be usable (IT should have set `dfs` up properly but they did not and this is why this tutorial is long).
So what we will do is put your code `afs` and create a soft link to it in `lfs`.
`lfs` stands for local file system and it's where your actual data (trainning data, models, python conda environment will reside).
We will soft link (with `ln -s <src> <target>`) your code/github repos from `afs` (src) to `lfs` (target) later. 

**TIP**: anything you don't understand we encourage you to discuss it with GPT4/Claude/your favorite LLM + paste the entire tutorial to in your conversation with it!  
e.g., ask it what an environment variable is, what a git command does, nvidia-smi command does, what `vim` is or what `git clone` is, `pip install`, `pip instlal -e .` , what conda is and python envs, what `$PATH` or `$HOME` is, tmux, basically bash related things it's great for!.

Now that you understand the rationale, so background on how starting a terminal (cli) works. 
Every time one logins into a snap node/server (or creates a new linux terminal with the `bash` cli command), the terminal (cli) needs to configure your unix/linux environment (the set of variables that hold important paths so that the terminal/cli "works as expected" e.g., standard commands/execuatable binaries are found like `python`, `ls`, `cd`, `echo`, `pwd`, `cat`, `vim` etc. or environment variables like `$HOME`, `$PATH`, etc. have the right content)

Usually the linux terminal (e.g. `bash`) runs ("sources") `.bash_profile` to set up your linux environment before providing the bash/cli/terminal session (where you write commands). 
This command usually sets up your environment variables for your terminal (e.g., current `bash` session) by running commands you specify in that file. 
In particular, it sets up [environment variables](https://www.google.com/search?q=environment+variables&rlz=1C5CHFA_enUS741US741&oq=environment+variab&gs_lcrp=EgZjaHJvbWUqDQgAEAAYgwEYsQMYgAQyDQgAEAAYgwEYsQMYgAQyBwgBEAAYgAQyBwgCEAAYgAQyBwgDEAAYgAQyBggEEEUYOTIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIHCAgQABiABDIHCAkQABiABNIBCDQ0MjVqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8), which are terminal/cli variables that point to important locations in the node/computer so that everything works! 
Now, let's inspect your `.bash_profile`. 
For that we need to go to first go to `/afs/cs.stanford.edu/u/brando9` with the `cd` command and then we can display the contents of the file with `cat`:
```bash
# ... AFTER sshing to a SNAP node
(evals_af) brando9@skampere1~ $
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ cd /afs/cs.stanford.edu/u/brando9
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ cat .bash_profile
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
In this specific `.bash_profile` runs ("sources") another file called `.bashrc`. 
Since `.bash_profile` runs/sources your `.bashrc` file each time you ssh/login to snap we will put our personal configurations for SNAP in our `.bashrc` located at `~/.bashrc` (note: `~` is the same as `$HOME` and points to your local path) i.e., that is the meaning of `. ~/.bashrc`. 

The plan is to:
  1. put `.bashrc` in `afs` since `afs` is accessible via all nodes (to manage a single version for all SNAP nodes)
  2. have the environment variable `$HOME` (`~`) point from the root of `afs` to `lfs` automatically everytime you login to any node.
  3. create a soft link in your node's lfs home root `$HOME` (`~`) pointing to the path at `afs` for the `.bashrc` file (otherwise when `.bash_profile` is ran/sourced your terminal won't be set up as you expect/want)
This culminates in you putting your `.bashrc` file in exactly this path:
```bash
/afs/cs.stanford.edu/u/<YOUR_CSID>/.bashrc
```

Let's do setps 1, 2, 3. 

Now we are doing 1 and 2 in one shot in `.bashrc`. 
First create a file `.bashrc` at `/afs/cs.stanford.edu/u/<YOUR_CSID>/` with (if it doesn't exist already, if it does exit remove it's contents, you will change it):
```bash
touch /afs/cs.stanford.edu/u/<YOUR_CSID>/.bashrc
```
open it with a terminal text editor. 
I suggest `vim` since that is what I use:
```bash
vim /afs/cs.stanford.edu/u/<YOUR_CSID>/.bashrc
```
then in `vim` press `i` to get into insert mode. 
Then [copy paste the contents of our base `.bashrc`](https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc#L24) file **but change everywhere the string `brando9` appears and put your `CSID`** (so read the file carefully before copy pasting). 
In particular, [note this command in your `.bashrc` file](https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc#L43C1-L47C31):
```bash
# - The defaul $HOME is /afs/cs.stanford.edu/u/brando9 but since you want to work in a specific server due to otherwise conda being so hard you need to reset what home is, see lfs: https://ilwiki.stanford.edu/doku.php?id=hints:storefiles#lfs_local_server_storage  
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD
```
which changes where your `$HOME` directory points to every time you login to a node in SNAP (since `.bashrc` is sourced via `.bash_profile` everytime you login or run the `bash` terminal command). 
It changes your default home/root location from `afs` (some weird Stanford file system with limited disk space) to the local node's `lfs` file system/directories. 
We do this because `afs` does not have a lot of disk compared to `lfs` (so that you don't run into disk space issues, if you do however, you will need to clean your disk or e-mail snap's IT). 
However, `afs` is fast but has little space (so it can only hold small text file e.g., usually git hub repos with code only, so not with data or gitlfs). 
In addition, `lfs` is also quicker (since it is "a local computer" i.e., you are logged in to **a compute node directly**, which is an unusual set up for a cluster). 
I repeat, we won't be using `dfs`. 
The eventual goal is that your code will be at `afs` and that soft link to your code will be at `lfs`. 
Note: the python conda environment, data and big files or data that needs fast access will be at `lfs`.

Now let's do 3 manually and needs to be done manually **every time you want to use a new node in SNAP**. 
This is only SNAP specific. 
Note: if you think you know how to not set up this manually each time, open a gitissue and propose a solution!
So let's make sure there is a pointer/reference to the your `.bashrc` file so `.bash_profile` can actually find it and run it when you create a terminal/cli or run `bash`.
For that create the soft link but change `brando9` to your `CSID`:
```bash
ln -s /afs/cs.stanford.edu/u/brando9/.bashrc ~/.bashrc
```
check that this worked by inspecting if the link exists at your root in lfs. 
Run these commands: 
```bash
cd /lfs/skampere1/0/<CSID>
ls -lah
```
Sample output:
```bash
(evals_af) brando9@skampere1~ $ cd /lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ ls -lah
total 99M
drwxr-sr-x  22 brando9 root 4.0K Apr  3 17:07 .
drwxrwsrwt  23 root    root 4.0K Mar 27 19:21 ..
-rw-------   1 brando9 root  48K Apr  3 11:20 .bash_history
lrwxrwxrwx   1 brando9 root   38 Oct 27 12:57 .bashrc -> /afs/cs.stanford.edu/u/brando9/.bashrc
...
lrwxrwxrwx   1 brando9 root   49 Apr  3 11:02 snap-cluster-setup -> /afs/cs.stanford.edu/u/brando9/snap-cluster-setup
...
```
Notice how we have `.bashrc -> /afs/cs.stanford.edu/u/brando9/.bashrc`, which menas the name `.bashrc` actually points to ` /afs/cs.stanford.edu/u/brando9/.bashrc`.  

Now that your `.bashrc` is in the `afs` location and `$HOME` (`~`) points to your node's `lfs` home path, we should restart your bash terminal in SNAP to test that your changes take effect. 
Recall, your terminal is now set up as in described in 1,2, 3 because `.bash_profile` runs `.bashrc`. 
So let's test it. 
Run in your terminal, one at a time and read the output of each command 
(never run any command blindly, always read/understand the command you're running and it's output): 
```bash
bash
echo $HOME
realpath ~/.bashrc
pwd ~/.bashrc
```
Sample ouput:
```bash
# Or log out and relogin to snap with your preferred node, so bash isn't needed anymore
(evals_af) brando9@skampere1~ $ bash
ln: failed to create symbolic link '/lfs/skampere1/0/brando9/iit-term-synthesis': File exists
(evals_af) brando9@skampere1~ $ echo $HOME
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ realpath ~/.bashrc
/afs/cs.stanford.edu/u/brando9/.bashrc
(evals_af) brando9@skampere1~ $ pwd ~/.bashrc
/lfs/skampere1/0/brando9
```
this demonstrates `$HOME` (`~`) points to your node's `lfs` and that the real path of `.bashrc` is actually in `afs`. 

Now repeat the above (without the `bash` command) but log out the snap node you are into and re login via ssh, and check your `.bashrc` is in `afs` and `$HOME` points to `lfs` (never run any command blindly, always read/understand the command you're running and it's output)::
```bash
sh brando9@skampere1.stanford.edu
echo $HOME
realpath ~/.bashrc
pwd ~/.bashrc
```
sample output:
```bash
(evals_af) brando9@skampere1~ $ echo $HOME
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ realpath ~/.bashrc
/afs/cs.stanford.edu/u/brando9/.bashrc
(evals_af) brando9@skampere1~ $ pwd ~/.bashrc
/lfs/skampere1/0/brando9
```
make sure you understand the difference between `realpath ~/.bashrc` and `pwd ~/.bashrc`.

### Setting up your bashrc file in Snap
Therefore, the goal is create your `.bashrc` at afs (some weird Stanford where your stuff might live with limited space) and then move it to your node's lfs and have everything live at your node's lfs permentantly (or optionally soft link it from you're node's lfs).

First echo `$HOME` (i.e., `~`) to figure out where your current home path is pointing too (most likely your `.bashrc` is located at `$HOME` or doesn't exist). 
Sample output: 
```bash
# where is $HOME or ~ pointing too?
brando9@mercury2:~$ echo $HOME
/afs/cs.stanford.edu/u/brando9
```
So this means we need our bash configurations at `~/.bashrc` i.e., to be at `$HOME/.bashrc` (`~` means `$HOME`).
So first let's create that file with `vi`m (see basic Vim bellow in this tutorial to know the basics or use ChatGPT):
```bash
# note I used the absolute path because we will have $HOME (i.e., ~) point to the local (lfs) home directory.
cd /afs/cs.stanford.edu/u/brando9
vim .bashrc
```
Now go to this https://github.com/brando90/evals-for-autoformalization/blob/main/.bashrc and copy paste it to your clip board 
e.g., with command/control + c. 
Now press `i` to go in vim's insert mode and paste the file as you'd normally 
e.g., with control/command + v. 
Read/sim through the `.bashrc` to see why it's set up the way it is.
Then press `esc` to `:w` enter to save the file. Then press `:q` enter to exist (or `:x` enter for for save & exit).
Note: this is (most likely) correct even though the wiki/docs for snap say to update `.bash.user` (but .bash.user is never sourced, I asked the it and I strongly recommend you ask too, see wrong/confusing docs if you want https://ilwiki.stanford.edu/doku.php?id=hints:enviroment but that's not what `.bash_profile` is sourcing!?).

Then the goal will be to have all your files live at the storage space for your local server (LFS, stands for local file server).
Therefore, let's move this `.bashrc` to your lfs username by permentantly changing `$HOME` (`~`) and all your git clones at the home of lfs for your assinged server.









#### Copy your .bashrc to your LFS
Recall we want everything including your `.bashrc` to live at your home's username at lfs (Optional TIP: you can also have a single `.bashrc` at AFS and every node's lfs you use point to it at afs, to save time & have a single `.bashrc`).

So likely `$HOME` is not pointing to your lfs username home. 
So let's first let's make sure `$HOME` (`~`) to your permanent location of your username home at lfs.
```bash
# run these commands in your terminal, this will make sure $HOME (and ~) point to your lfs location for only the current bash session (to have it permanent change it has to be in the bash file .bash_profile is sourcing/running each time you start a bash session
# -- Set up the lfs home for this bash session
brando9@mercury2:~$ export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
brando9@mercury2:~$ export HOME=$LOCAL_MACHINE_PWD

# -- Confirm $HOME and that you're in lfs
brando9@mercury2:/afs/cs.stanford.edu/u/brando9$ echo $HOME
/lfs/mercury2/0/brando9
brando9@mercury2:/afs/cs.stanford.edu/u/brando9$ cd ~
brando9@mercury2:~$ pwd
/lfs/mercury2/0/brando9
```
the last line confirms we are at the local servers storage (called lfs).

Now the goal will be have `.bash_profile` run the right bash file you have set up (in this case it's running `~/.bashrc` so we need to set that up).
For that we can copy your `.bashrc` file in afs to your lfs location.
Or copy paste the right contents to a new `~/.bashrc` file to your lfs user location (e.g., either from the .bashrc file I set up for you or the contents of your afs one).
(note a 3rd option exists to soft link to the afs .bashrc, which is the one I usually use).
To do that do this:
```bash
# confirm your home points to the right place
brando9@mercury2:~$ echo $HOME
/lfs/mercury2/0/brando9

brando9@mercury2:~$ cp /afs/cs.stanford.edu/u/brando9/.bashrc ~/.bashrc

# Run this to confirm you moved it!
brando9@mercury2:~$ cat ~/.bashrc
```
This should copy your .bashrc file to your lfs location (and always confirm and read what your running!).

But why is it that now that we login to the cluster we are in the lfs location and not afs?
Well look at look at the .bashrc there is this line:
```
```bash
# - The defaul $HOME is /afs/cs.stanford.edu/u/brando9 but since you want to work in a specific server due to otherwise conda being so hard you need to reset what home is, see lfs: https://ilwiki.stanford.edu/doku.php?id=hints:storefiles#lfs_local_server_storage  
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD

# - set up afs short cuts
# since you are logged in to afs this moves you to your local computer
cd $HOME

...more code...
```
meaning that every time you login to the server assigned you got to your lfs directory instead of the afs home directory (since `$HOME` was changed).
