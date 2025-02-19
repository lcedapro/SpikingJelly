# 打印模型权重文件结构，输出到csv文件

import torch

checkpoint = torch.load(r"D:\Users\oilgi\source\Python\PyTorch\SpikingJelly\16-davispku\logs_gconv\T_4_b_16_c_2_SGD_lr_0.2_CosALR_48_amp_cupy\checkpoint_max_bn2conv.pth", map_location='cpu', weights_only=False)
import csv
with open('model_parameters_conv2int.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Layer', 'Size', 'Dtype', 'Requires Grad'])
    for name, param in checkpoint['net'].items():
        size = param.size()
        dtype = param.dtype
        requires_grad = param.requires_grad
        writer.writerow([name, size, dtype, requires_grad])