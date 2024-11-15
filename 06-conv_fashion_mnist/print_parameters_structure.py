# 打印模型权重文件结构，输出到csv文件

import torch

checkpoint = torch.load('./logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_bn2conv.pth', map_location='cpu')
import csv
with open('model_parameters_bn2conv.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Layer', 'Size', 'Dtype', 'Requires Grad'])
    for name, param in checkpoint['net'].items():
        size = param.size()
        dtype = param.dtype
        requires_grad = param.requires_grad
        writer.writerow([name, size, dtype, requires_grad])