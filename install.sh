# set up paths
export LOCAL_MACHINE_PWD=$(python3 -c "import socket;hostname=socket.gethostname().split('.')[0];print('/lfs/'+str(hostname)+'/0/brando9');")
mkdir -p $LOCAL_MACHINE_PWD
export WANDB_DIR=$LOCAL_MACHINE_PWD
export HOME=$LOCAL_MACHINE_PWD
# Conda needs to be set up first before you can test the gpu
export PATH="$HOME/miniconda/bin:$PATH"

# - activates base to test things
source $HOME/miniconda/bin/activate
conda activate my_env

# first make sure you install the right version of pytorch 
# pip3 install torch torchvision torchaudio
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
#pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
#pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# - set up cuda driver for your server: https://ilwiki.stanford.edu/doku.php?id=hints:gpu
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

# - test gpu: print cuda version and run a pytorch gpu op
nvcc -V
python -c "import torch; print(torch.cuda.get_device_capability())"
python -c "import torch; print(torch.bfloat16);"
python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"