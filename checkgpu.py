import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA detected, running on CPU")


#lmao you can use this to test whether your computer has a GPU or not, apparently mine didn't. 

