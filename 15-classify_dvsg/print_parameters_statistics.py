# 打印模型各层参数和所有参数的统计量

import torch
import numpy as np

# 加载模型权重
checkpoint = torch.load('./logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_bn2conv.pth', map_location='cpu')
state_dict = checkpoint['net']

# 初始化统计量
total_mean = 0
total_var = 0
total_max = -float('inf')
total_min = float('inf')
total_count = 0

# 遍历OrderedDict中的所有参数
for param_key, param_tensor in state_dict.items():
    # 将参数转换为NumPy数组
    param_array = param_tensor.numpy()
    
    # 计算当前参数的统计量
    mean = np.mean(param_array)
    var = np.var(param_array)
    max_val = np.max(param_array)
    min_val = np.min(param_array)
    # print(f'Mean of {param_key}: {mean}')
    # print(f'Variance of {param_key}: {var}')
    print(f'Max value of {param_key}: {max_val}')
    print(f'Min value of {param_key}: {min_val}')
    
    # 更新统计量
    total_mean += mean * param_array.size
    total_var += var * param_array.size
    total_max = max(total_max, max_val)
    total_min = min(total_min, min_val)
    total_count += param_array.size

# 计算所有参数的统计量
total_mean /= total_count
total_var = (total_var / total_count) - (total_mean ** 2)  # 计算样本方差

# 打印统计量
# print(f'Mean of all parameters: {total_mean}')
# print(f'Variance of all parameters: {total_var}')
print(f'Max value of all parameters: {total_max}')
print(f'Min value of all parameters: {total_min}')