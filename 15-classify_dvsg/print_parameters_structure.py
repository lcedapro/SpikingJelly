# 打印模型权重文件结构，输出到csv文件

import torch

checkpoint = torch.load('./logs/T_16_b_48_c_32_Adam_lr_0.01_CosALR_16_amp_cupy_3/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=False)
import csv
with open('model_parameters_conv2int.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Layer', 'Size', 'Dtype', 'Requires Grad'])
    for name, param in checkpoint['net'].items():
        size = param.size()
        dtype = param.dtype
        requires_grad = param.requires_grad
        writer.writerow([name, size, dtype, requires_grad])