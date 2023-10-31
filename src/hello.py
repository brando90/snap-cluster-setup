# - Test your python pip install -e . by printing hello
print('hello (Test your python pip install -e . by printing hello)')

# - Imports PyTorch and prints the CUDA compute capability of the current GPU, indicating its specifications and supported features. 
# # python -c "import torch; print(torch.cuda.get_device_capability())"
import torch; print(torch.cuda.get_device_capability())

# - Prints in torch bfloat16 is implemented in the current GPU (the usually better float for ML)
# # python -c "import torch; print(torch.bfloat16);"
import torch; print(torch.bfloat16)

# - Test the GPU by checking if it can actually do GPU computations
# # python -c "import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'));"
import torch; print(torch.randn(2, 4).to('cuda') @ torch.randn(4, 1).to('cuda'))