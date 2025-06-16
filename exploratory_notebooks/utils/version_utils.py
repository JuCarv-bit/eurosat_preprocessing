import sys
import torch
import numpy as np
import random

try:
    import torch
except ImportError:
    torch = None

try:
    import torchvision
except ImportError:
    torchvision = None


def print_versions():
    def check_python_version():
        return sys.version.split()[0]


    def check_torch_version():
        return torch.__version__ if torch else None


    def check_cuda_available():
        return torch.cuda.is_available() if torch else False


    def get_cuda_device_count():
        return torch.cuda.device_count() if torch else 0


    def check_torchvision_version():
        return torchvision.__version__ if torchvision else None

    print(f"Python version: {check_python_version()}")
    print(f"PyTorch version: {check_torch_version() or 'Not installed'}")
    print(f"CUDA available: {check_cuda_available()}")
    if check_cuda_available():
        print(f"CUDA device count: {get_cuda_device_count()}")
    print(f"Torchvision version: {check_torchvision_version() or 'Not installed'}")



def configure_gpu_device(TARGET_GPU_INDEX):
    if torch.cuda.is_available():
        if TARGET_GPU_INDEX < torch.cuda.device_count():
            DEVICE = torch.device(f"cuda:{TARGET_GPU_INDEX}")
            print(f"Successfully set to use GPU: {TARGET_GPU_INDEX} ({torch.cuda.get_device_name(TARGET_GPU_INDEX)})")
        else:
            print(f"Error: Physical GPU {TARGET_GPU_INDEX} is not available. There are only {torch.cuda.device_count()} GPUs (0 to {torch.cuda.device_count() - 1}).")
            print("Falling back to CPU.")
            DEVICE = torch.device("CPU")
    else:
        print("CUDA is not available. Falling back to CPU.")
        DEVICE = torch.device("CPU")

    print(f"Final DEVICE variable is set to: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"Current PyTorch default device: {torch.cuda.current_device()}")
        torch.cuda.set_device(TARGET_GPU_INDEX)
        print(f"Current PyTorch default device (after set_device): {torch.cuda.current_device()}")


    dummy_tensor = torch.randn(2, 2)
    dummy_tensor_on_gpu = dummy_tensor.to(DEVICE)
    print(f"Dummy tensor is on device: {dummy_tensor_on_gpu.device}")
    return DEVICE

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

