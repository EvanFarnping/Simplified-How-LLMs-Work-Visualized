import torch

# Check Cuda exists to use GPU.
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)