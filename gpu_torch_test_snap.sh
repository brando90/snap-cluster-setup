# -- Conda needs to be set up first before you can test the gpu & the right pytorch version has to be installed
echo "Assumes you have my_env with python >=3.10"
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
export PATH="$HOME/miniconda/bin:$PATH"

# - activates base to test things
source $HOME/miniconda/bin/activate
# conda activate my_env

# first make sure you install the right version of pytorch 
# pip3 install torch torchvision torchaudio
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# - https://ilwiki.stanford.edu/doku.php?id=hints:gpu
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

# test gpu: print cuda version and run a pytorch gpu op
nvcc -V
python -c "import torch; print(torch.cuda.get_device_capability())"
python -c "import torch; print(torch.bfloat16);"
python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"

