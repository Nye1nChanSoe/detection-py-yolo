import torch

def main():
    print(f"PyTorch version: {torch.__version__}")

    # check CUDA gpu is available if not it will use CPU
    gpu_available = torch.cuda.is_available()
    print(f"CUDA is available: {gpu_available}")

    if gpu_available:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("Tensor:")
    print(tensor)

    result = tensor + tensor
    print("Tensor after addition:")
    print(result)

if __name__ == "__main__":
    main()
