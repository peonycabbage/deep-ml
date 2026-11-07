import torch
import torch.nn.functional as F


torch.backends.nnpack.set_flags(False)

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)

    #Step 1: add batch channels and dimensions to the input_matrix and the kernel
    input_4d = input_matrix.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

    output = F.conv2d(input_4d, kernel_4d, padding=padding, stride=stride)
    return output.squeeze(0).squeeze(0)





    # # Step1: add batch and channel dimensions
    # input_4d = input_matrix.unsqueeze(0).unsqueeze(0)
    # # Step2 : add output and input channel dimensions
    # kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    # # Step3: Perform convolution
    # output = F.conv2d(input_4d, kernel_4d, padding=padding, stride=stride)

    # # Step4 : Remove extra dimensions
    # return output.squeeze(0).squeeze(0)
