# evals-for-autoformalization

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

Now you should be able to import statements for this library in the expected way! e.g.,
```python
# TODO
```

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
TODO
```bash
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/massive-autoformalization-maf.git
ln -s /afs/cs.stanford.edu/u/brando9/massive-autoformalization-maf $HOME/massive-autoformalization-maf
pip install -e ~/massive-autoformalization-maf
#pip uninstall ~/massive-autoformalization-maf
cd ~/massive-autoformalization-maf
```

## Basic Docker

## Basic Cluster use (Snap)
