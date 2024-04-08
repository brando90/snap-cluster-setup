# SNAP Cluster Setup

If you want to do this tutorial with active reflections in the style of a assignment see this: https://docs.google.com/document/d/1k3d3_AOp-Y22L-GbgHivDqKe9r9LC5nscaIzbm3LWYc/edit?usp=sharing

## Get Compute for your Research Project

### SNAP Cluster Important References & Help/Support
Always use the original documentation or wiki for each cluster: https://ilwiki.stanford.edu/doku.php?id=start -- your **snap bible**.
Other useful resources:
- Support IT for snap: il-action@cs.stanford.edu (don't be shy to ask them question or help for SNAP.)
- compute instructions from Professor Koyejo's (Sanmi's) lab (STAIR Lab): https://docs.google.com/document/d/1PSTLJdtG3AymDGKPO-bHtzSnDyPmJPpJWXLmnJKzdfU/edit?usp=sharing
- advanced video from Rylan and Brando (made for the STAIR/Koyejo Lab): https://www.youtube.com/watch?v=XEB79C1yfgE&feature=youtu.be
- our CS 197 section channel
- join the snap slack & ask questions there too: https://join.slack.com/t/snap-group/shared_invite/zt-1lokufgys-g6NOiK3gQi84NjIK_2dUMQ

## Get access to SNAP with a CSID

### First request CSID with Michael Bersntein as sponsor 
First create a CSID here and  please make your CSID the same as your Stanford SUNET id. 
Request it here:  https://webdb.cs.stanford.edu/csid and put Michael Bernstein as your CSID sponsor/supervisor 
Note: this is different from SNAP cluster sponsor. 

### Second get acces to SNAP

To get access to snap write an e-mail with this subject:

> Access Request SNAP Cluster Working With Brando Miranda CS197 for <full_name> <CSID>  <SUNET>

For example: 

> Access Request SNAP Cluster Working With Brando Miranda CS197 for Brando Miranda brando9 brando9$ 

and sent it to
- Eric Pineda: eric.pineda@stanford.edu
- Brando Miranda: brando9@stanford.edu
- [SNAP cluster IT](https://ilwiki.stanford.edu/doku.php?id=start): il-action@cs.stanford.edu
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

**Tip**: If the `reauth `command doesn't work do or/and e-mail the [SNAP cluster IT](https://ilwiki.stanford.edu/doku.php?id=start) il-action@cs.stanford.edu:
```bash
export PATH="/afs/cs/software/bin:$PATH"
```
**TIP**: Ask ChatGPT what `export PATH="/afs/cs/software/bin:$PATH"` does. ChatGPT is excellent at the terminal and bash commands. Note consider adding this to your `.bashrc` for example [see this](https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc#L24). 
**TIP**: you might have to ssh into your node again outside of vscode for this to work if vscode is giving you permission issues or e-mail snap IT. 

## Setup your .bashrc, redirect the $HOME (~) envi variable from afs to lfs and create a soft link for .bashrc (afs/.bashrc -> lfs/.bashrc)
Rationale: [SNAP has 3 file systems afs, lfs, dfs](https://ilwiki.stanford.edu/doku.php?id=hints:storefiles) (folders where your files, data and code could potentially be stored). 
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
I suggest `vim` (read [this for a quick ref for vim](https://coderwall.com/p/adv71w/basic-vim-commands-for-getting-started)) since that is what I use:
```bash
vim /afs/cs.stanford.edu/u/<YOUR_CSID>/.bashrc
```
then in `vim` press `i` to get into insert mode. 
Then [copy paste the contents of our base `.bashrc`](https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc#L24) file **but change everywhere the string `brando9` appears and put your `CSID`** (so read the file carefully before copy pasting). 
Then press `esc` to `:w` enter to save the file. Then press `:q` enter to exist (or `:x` enter for for save & exit). 
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

Bonus Note: this is (most likely) correct even though the wiki/docs for snap say to update `.bash.user` (but .bash.user is never sourced, I asked the it and I strongly recommend you ask too, see wrong/confusing docs if you want https://ilwiki.stanford.edu/doku.php?id=hints:enviroment but that's not what `.bash_profile` is sourcing!?).

### Basic editing in the terminal with vim
You already created a `.bashrc`
Vim is for editing file in the terminal (cli). 
These are the commands you might need https://coderwall.com/p/adv71w/basic-vim-commands-for-getting-started. 
Mostly knowing that:
-> pressing `i` puts you in insert mode
-> pressing `w` writes/save the files
-> pressing `q` quits
that's all you need.
If you want more experience with vim do this:

Certainly! Below is a small, self-contained markdown tutorial focusing on editing files in the terminal using Vim. This tutorial will cover the basics of opening a file, entering insert mode for editing, exiting insert mode, and saving your changes.

```markdown
#### Vim Editing Basics Tutorial
Vim is a highly efficient text editor that operates in various modes. The most commonly used modes are "Normal" and "Insert" modes. This tutorial will guide you through opening a file in Vim, editing it, and saving your changes.
## Opening a File in Vim
To open a file with Vim, use the following command in your terminal:
```bash
vim filename
```
Replace `filename` with the name of the file you wish to edit. If the file does not exist, Vim will create it for you.
Entering Insert Mode: 
Once you have your file opened in Vim, you'll start in Normal mode. To start editing your file, you need to enter Insert mode.
```vim
i
```
Pressing `i` will switch you from Normal mode to Insert mode. Now you can start typing and editing the file as you would in a conventional text editor.
Exiting Insert Mode: 
To stop editing and return to Normal mode, press:
```vim
Esc
```
The `Esc` key brings you back to Normal mode, where you can execute Vim commands.

Saving Changes: To save your changes in Normal mode, type:
```vim
:w
```
This command writes your modifications to the file but keeps it open in Vim.

Exiting Vim: If you want to exit Vim *after* saving your changes, type:
```vim
:q
```
However, to both save your changes and exit Vim in one command, you can use:
```vim
:x
```
or
```vim
:wq
```
Both commands save any changes and then close the file.

Conclusion: 
This tutorial covered the basics of file editing with Vim in the terminal, including opening files, switching between Insert and Normal modes, saving changes, and exiting Vim. With these commands, you can start editing files efficiently in the terminal. 
[Couresty of ChatGPT (GPT4)](https://chat.openai.com/c/9fbefbf9-f7c7-4e68-84de-6bbd23853a77).

## Using git and conda environments in SNAP
At this point, you know what the environment variable `$HOME` is and that yours should be pointing to your node’s `lfs` home directory -- as suggested by [this .bahrc file](https://github.com/brando90/snap-cluster-setup/blob/main/.bashrc#L43C1-L47C31) (but using your CSID).

Now the goal will be to:
1. show you how to `git clone` (optionally git fork this repo) a small sized git project (so not using `gitlfs`, that would most likely create issues with AFS's storage limit) like [this one (`snap-cluster-setup`)](https://github.com/brando90/snap-cluster-setup)
2. learn what a [python environment](<https://csguide.cs.princeton.edu/software/virtualenv#:~:text=A%20Python%20virtual%20environment%20(venv,installed%20in%20the%20specific%20venv.>) is and create one using [conda](https://docs.conda.io/en/latest/)

For step 1 we will first go to your `afs` home root path. 
We do this because `afs` is the only "distributed" file system that isn't very slow and that **is** accessible at all [SNAP nodes/servers](https://ilwiki.stanford.edu/doku.php?id=start). 
It will be similar to how we created our `.bashrc` file previously at your `afs` root directory and then soft link it at your `lfs` directory (which should be your `$HOME` path). 
The main difference is that instead of soft linking a file, we will soft linke a directory. 

So `cd` to your `afs` root directory. 
If you correctly edited your `.bashrc` then you should be able to do to move to your `afs` root: 
```bash
cd $AFS
```
tip: do `echo $AFS` or to any environment variable to see the contents of them!).
If that didn't work, fix your `.bashrc` and do the previous command (e.g., make sure `.bashrc` exports `$AFS`, change all `brando9` instances with your `CSID`, and potentially see the original `.bashrc` if something doesn't work or even start with a new `.bashrc` again.
Note, `cd $AFS` should be equivalent to doing (with your `CSID`):
```bash
cd /afs/cs.stanford.edu/u/brando9
```
Since AFS should be point to your afs due to this command `export AFS=/afs/cs.stanford.edu/u/brando9` in the `.bashrc`. 
Sample output (understand the commands): 
```bash
# after sshing into a snap node or running bash on your terminal (cli)
(evals_af) brando9@skampere1~ $
(evals_af) brando9@skampere1~ $ realpath .
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ pwd
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ cd $AFS
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ pwd
/afs/cs.stanford.edu/u/brando9
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ realpath .
/afs/cs.stanford.edu/u/brando9
```

Now that we are at our `afs` home root directory, let's git clone this repo (or ideally your git fork). 
This step is really showing you to git clone any github repo to your `afs` (npte: later you will git clone your actual project's repo and repeat cloning it to your afs every time you have a new repo you want to use for snap.
Note: if you're following this setup, since we are using `lfs` due to storage issues with `afs` you will need to set up a new conda env each time use a new node, although the repo might be available already).

Git clone this repo with using ssh path from the github repo option and check that it was downloaded with `ls -lah` and `realpath`
e.g.,: 
```bash
cd $AFS
git clone git@github.com:brando90/snap-cluster-setup.git
ls -lah
realpath snap-cluster-setup
```
Sample output: 
```bash
(evals_af) brando9@skampere1~ $ cd $AFS
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ 
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $  git clone git@github.com:brando90/snap-cluster-setup.git
Cloning into 'snap-cluster-setup'...
remote: Enumerating objects: 906, done.
remote: Counting objects: 100% (360/360), done.
remote: Compressing objects: 100% (134/134), done.
remote: Total 906 (delta 284), reused 291 (delta 225), pack-reused 546
Receiving objects: 100% (906/906), 74.63 MiB | 7.03 MiB/s, done.
Resolving deltas: 100% (503/503), done.
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ ls -lah
total 74M
drwxrwxrwx 29 brando9 users 4.0K Apr  3 13:41 .
drwxr-xr-x  2 root    users 520K Apr  1 11:10 ..
...
-rw-r--r--  1 brando9 users 8.8K Jan 29 16:33 .bashrc
...
drwxr-xr-x  5 brando9 users 2.0K Apr  3 16:50 snap-cluster-setup
...
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ realpath snap-cluster-setup
/afs/cs.stanford.edu/u/brando9/snap-cluster-setup
```
Note that since you are at your root `afs`, the `.bashrc` file is not a soft link. 
Your `snap-cluster-setup` should also **not** be a soft link at your root `afs` directory. 
Verify that with `realpath` (and not `pwd`):
```bash
pwd .bashrc
realpath .bashrc
pwd snap-cluster-setup/
realpath snap-cluster-setup/
```
Sample output:
```bash
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ pwd .bashrc
/afs/cs.stanford.edu/u/brando9
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ realpath .bashrc
/afs/cs.stanford.edu/u/brando9/.bashrc
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ pwd snap-cluster-setup/
/afs/cs.stanford.edu/u/brando9
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ realpath snap-cluster-setup/
/afs/cs.stanford.edu/u/brando9/snap-cluster-setup
```

### Create a soft link for your cloned github project (lfs -> afs)
We will create a soft link as we previously did in your home `lfs` path for the github project you just cloned, and santiy check it is a soft link (similar to your `.bashrc` file).
Run one command at a time and see output (we strongly suggest against running command blindly, e.g., if one doesn't work, then how will you fix it if you don't know what you've run so far?):
```bash
echo $HOME
# cd ~ 
cd $HOME
# ln -s /afs/cs.stanford.edu/u/brando9/snap-cluster-setup $HOME/snap-cluster-setup
ln -s $AFS/snap-cluster-setup $HOME/snap-cluster-setup
ls -lah
pwd $HOME/snap-cluster-setup
realpath $HOME/snap-cluster-setup
```
(Tip: to learn about the equivalence of `~` and `$HOME`, re-run the `pwd` command with `~` instead of `$HOME`. 
What happens?)
Sample output:
```bash
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ echo $HOME
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1/afs/cs.stanford.edu/u/brando9 $ cd $HOME
(evals_af) brando9@skampere1~ $ ln -s $AFS/snap-cluster-setup $HOME/snap-cluster-setup
(evals_af) brando9@skampere1~ $ ln -s $AFS/snap-cluster-setup $HOME/snap-cluster-setup
(evals_af) brando9@skampere1~ $ ls -lah
total 99M
drwxr-sr-x  22 brando9 root 4.0K Apr  3 19:39 .
drwxrwsrwt  24 root    root 4.0K Apr  3 18:33 ..
...
lrwxrwxrwx   1 brando9 root   38 Oct 27 12:57 .bashrc -> /afs/cs.stanford.edu/u/brando9/.bashrc
...
lrwxrwxrwx   1 brando9 root   49 Apr  3 11:02 snap-cluster-setup -> /afs/cs.stanford.edu/u/brando9/snap-cluster-setup
...
(evals_af) brando9@skampere1~ $ pwd $HOME/snap-cluster-setup
/lfs/skampere1/0/brando9
(evals_af) brando9@skampere1~ $ realpath $HOME/snap-cluster-setup
/afs/cs.stanford.edu/u/brando9/snap-cluster-setup
```
Tip: `ls` is a very useful command. What is is? Try `ls` vs `ls -l` vs `ls -la` vs `ls -lah`. 
What is the difference? Use the `man` command or `help` flag to learn more about it.

This sanity checks that your `snap-cluster-setup` indeed is locacted at your root `afs` but a soft link is at your root `lfs` path. 
This should convince you by looking where `.bashrc` points too and where your github repo called `snap-cluster-setup` points too. 

Note: you have to repeat the above ach time you set up a new github repo in SNAP. 
This is because you have not cloned the repo to `afs` to be shared accross all the SNAP nodes. 
In addition `afs` is small (but fast) and `dfs` is useless in this cluster sadly, so we put the code in `afs` so it's shared accross all nodes and not use `dfs`. 

### Python envs with Conda in lfs (not afs!)
When you have python projects, they usually have libraries they depend on. 
These are called depedencies (Google it to learn more!). 
So that these depedency installations for different projects in the same machine do not conflict we have different "<https://csguide.cs.princeton.edu/software/virtualenv#:~:text=A%20Python%20virtual%20environment%20(venv,installed%20in%20the%20specific%20venv.>) is and create one using [conda](https://docs.conda.io/en/latest/)" (also called conda environments or [virtual environments](https://ilwiki.stanford.edu/doku.php?id=hints:virtualenv)). 
Approximately, these are different locations in your system will your the installations for each project and setting up usually "secret" enviornment variable depending what depedency management system you are using e.g., conda or virtual env or poetry etc. 
This is usually specific for each programming langauge you use (e.g., pip, conda for python, opam for coq, etc.) so it's often hard to remember the details and often people learn one setup and re-use it all the time. 

For snap we will use conda. 
First check if you have conda in your cli by doing
```bash
which conda 
```
Pro Tip: if you are using vscode make sure your vscode is pointing to the python or conda env you want for your project, usually you set it up at the bottom right in vscode. 
Sample output:
```bash
(base) brando9@skampere1~ $ which conda
/lfs/skampere1/0/brando9/miniconda/bin/conda
```
Tip: a nice sanity check is check which binary the default/system is using. Usually by doing conda deactivate and then which python in the cli. 

If the previous command doesn't work then we will have to **locally** install conda (i.e., you will install conda in lfs by putting it's binary in a location you have permission to use).  
When you have a cluster usually the system admins install software you need and usually you do not have sudo priviledges (google that or do `man sudo` t learn!). 
Otherwise a trick to go around it is to install locally as we are about to do (but it's not usually recommended). 
Install conda (and update pip) if it's missing by running the following commands (**do not run them blindly, see the output before proceeding to the next one**):
```bash
echo $HOME
cd $HOME

# -- Install miniconda
# get conda from the web
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# source conda with the bash command and put the installation at $HOME/miniconda
bash $HOME/miniconda.sh -b -p $HOME/miniconda
# activate conda
source $HOME/miniconda/bin/activate

# - Set up conda
conda init
conda init bash
conda install conda-build
conda update -n base -c defaults conda
conda update conda

# - Make sure pip is up to date
pip install --upgrade pip
pip3 install --upgrade pip
which pip
which pip3

# - Restart your bash terminal
bash
```
The previous commands take a while so instead I will only demonstrate two commands to check if conda was installed right. 
Make sure you have a fresh `bash` terminal (understand the commands):
```bash
# which conda binary are we using?
which conda
# which python binary are we using?
which python
# which conda envs do we have?
conda info -e
```
sample output:
```bash
(base) brando9@skampere1~ $ which conda
/lfs/skampere1/0/brando9/miniconda/bin/conda
(base) brando9@skampere1~ $ which python
/lfs/skampere1/0/brando9/miniconda/bin/python
(base) brando9@skampere1~ $ conda info -e
# conda environments:
#
base                  *  /lfs/skampere1/0/brando9/miniconda
beyond_scale             /lfs/skampere1/0/brando9/miniconda/envs/beyond_scale
evaporate                /lfs/skampere1/0/brando9/miniconda/envs/evaporate
lean4ai                  /lfs/skampere1/0/brando9/miniconda/envs/lean4ai
maf                      /lfs/skampere1/0/brando9/miniconda/envs/maf
my_env                   /lfs/skampere1/0/brando9/miniconda/envs/my_env
olympus                  /lfs/skampere1/0/brando9/miniconda/envs/olympus
pred_llm_evals_env       /lfs/skampere1/0/brando9/miniconda/envs/pred_llm_evals_env
putnam_math              /lfs/skampere1/0/brando9/miniconda/envs/putnam_math
snap_cluster_setup       /lfs/skampere1/0/brando9/miniconda/envs/snap_cluster_setup
```

It would be ideal to create a single conda env for each project and put it in afs so that all nodes can use all the code for your github project **and** it's depedencies too. 
Unfortuantely, installing depedencies usually takes a lot of space. 
Thus we will install conda and create a conda env for each github project with it's installed depedencies in each node's lfs. 
Thus, create a new conda env for this `snap-cluster-setup` tutorial:
```bash
# - check envs
conda info -e

# - activate conda
conda update
pip install --upgrade pip

# - create conda env (note: vllm has issues with 3.10 so we are using 3.9)
conda create -n snap_cluster_setup python=3.9

# - activate your conda env
conda activate snap_cluster_setup

# - wandb
pip install --upgrade pip
pip install wandb
pip install wandb --upgrade
```
Note: I try to use pip as much as I can instead of conda to install my packages (or be consistent with which one I use, but sometimes you will have to combine each one since package management and installation can be tricky and hacky. Welcome to the real world!).
Sample output:
```bash
(base) brando9@skampere1~ $ conda create -n snap_cluster_setup python=3.9
Retrieving notices: ...working... done
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /lfs/skampere1/0/brando9/miniconda/envs/snap_cluster_setup

  added / updated specs:
    - python=3.9


The following NEW packages will be INSTALLED:

  _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main 
  _openmp_mutex      pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu 
  ca-certificates    pkgs/main/linux-64::ca-certificates-2024.3.11-h06a4308_0 
  ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.38-h1181459_1 
  libffi             pkgs/main/linux-64::libffi-3.4.4-h6a678d5_0 
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-11.2.0-h1234567_1 
  libgomp            pkgs/main/linux-64::libgomp-11.2.0-h1234567_1 
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-11.2.0-h1234567_1 
  ncurses            pkgs/main/linux-64::ncurses-6.4-h6a678d5_0 
  openssl            pkgs/main/linux-64::openssl-3.0.13-h7f8727e_0 
  pip                pkgs/main/linux-64::pip-23.3.1-py39h06a4308_0 
  python             pkgs/main/linux-64::python-3.9.19-h955ad1f_0 
  readline           pkgs/main/linux-64::readline-8.2-h5eee18b_0 
  setuptools         pkgs/main/linux-64::setuptools-68.2.2-py39h06a4308_0 
  sqlite             pkgs/main/linux-64::sqlite-3.41.2-h5eee18b_0 
  tk                 pkgs/main/linux-64::tk-8.6.12-h1ccaba5_0 
  tzdata             pkgs/main/noarch::tzdata-2024a-h04d1e81_0 
  wheel              pkgs/main/linux-64::wheel-0.41.2-py39h06a4308_0 
  xz                 pkgs/main/linux-64::xz-5.4.6-h5eee18b_0 
  zlib               pkgs/main/linux-64::zlib-1.2.13-h5eee18b_0 


Proceed ([y]/n)? y


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate snap_cluster_setup
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(base) brando9@skampere1~ $ conda activate snap_cluster_setup
(snap_cluster_setup) brando9@skampere1~ $ which python
/lfs/skampere1/0/brando9/miniconda/envs/snap_cluster_setup/bin/python
(snap_cluster_setup) brando9@skampere1~ $ pip install --upgrade pip
Requirement already satisfied: pip in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (24.0)

(snap_cluster_setup) brando9@skampere1~ $ pip install wandb
Requirement already satisfied: wandb in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (0.16.6)
Requirement already satisfied: Click!=8.0.0,>=7.1 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (8.1.7)
Requirement already satisfied: GitPython!=3.1.29,>=1.0.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (3.1.43)
Requirement already satisfied: requests<3,>=2.0.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (2.31.0)
Requirement already satisfied: psutil>=5.0.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (5.9.8)
Requirement already satisfied: sentry-sdk>=1.0.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (1.44.1)
Requirement already satisfied: docker-pycreds>=0.4.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (0.4.0)
Requirement already satisfied: PyYAML in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (6.0.1)
Requirement already satisfied: setproctitle in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (1.3.3)
Requirement already satisfied: setuptools in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (68.2.2)
Requirement already satisfied: appdirs>=1.4.3 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (1.4.4)
Requirement already satisfied: typing-extensions in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (4.11.0)
Requirement already satisfied: protobuf!=4.21.0,<5,>=3.15.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from wandb) (4.25.3)
Requirement already satisfied: six>=1.4.0 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from docker-pycreds>=0.4.0->wandb) (1.16.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from GitPython!=3.1.29,>=1.0.0->wandb) (4.0.11)
Requirement already satisfied: charset-normalizer<4,>=2 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from requests<3,>=2.0.0->wandb) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from requests<3,>=2.0.0->wandb) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from requests<3,>=2.0.0->wandb) (2.2.1)
Requirement already satisfied: certifi>=2017.4.17 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from requests<3,>=2.0.0->wandb) (2024.2.2)
Requirement already satisfied: smmap<6,>=3.0.1 in ./miniconda/envs/snap_cluster_setup/lib/python3.9/site-packages (from gitdb<5,>=4.0.1->GitPython!=3.1.29,>=1.0.0->wandb) (5.0.1)
```
Note: to remove a conda env do `conda remove --name snap_cluster_setup --all`, careful!
Note: sometimes you might have to do `pip3`, not sure why pip is inconsistent. 

## Setting up a python project and using an editable pip installation (pip install -e .)
At this point, you know what the environment variable $HOME is, and it is pointing to your/any node’s lfs path as suggested by this .bahrc file (but using your CSID instead of Brando’s). In addition, you have created a fork of the snap-cluster-setup github repo in your afs root, and made sure you have a soft link in your lfs pointing to the afs path of the github repo. You also created a conda environment in your lfs (NOT your afs). You also know why we put large storage things (like conda installations and data or models) in your working node’s lfs and not lfs or afs (if not review it here).

