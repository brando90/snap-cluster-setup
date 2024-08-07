# ref: https://chatgpt.com/c/f58b4f97-ca57-4b82-9cce-f094a4dbbd6a

# Check CUDA version
nvcc --version

# Check cuDNN version
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Check NVIDIA driver version
nvidia-smi

# Check PyTorch version and CUDA version
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda); print('cuDNN version:', torch.backends.cudnn.version())"
