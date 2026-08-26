import torch

# Print the installed PyTorch version
print(torch.__version__)

# Check if your GPU acceleration is working (Returns True or False)
print(torch.cuda.is_available())
