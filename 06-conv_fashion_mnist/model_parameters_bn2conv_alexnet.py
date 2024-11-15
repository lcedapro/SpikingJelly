# 下述代码实现将 BatchNorm 层的参数吸收到 Conv 层中，生成新的 Conv 层权重和偏置。

import torch
import torch.nn as nn
from collections import OrderedDict

def fuse_bn_to_conv(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps):
    """
    将 BatchNorm2d 层的参数吸收到 Conv2d 层中，生成新的权重和偏置。
    
    Args:
    - conv_weight: Conv2d 的权重
    - bn_weight: BatchNorm2d 的 weight (gamma)
    - bn_bias: BatchNorm2d 的 bias (beta)
    - bn_running_mean: BatchNorm2d 的 running mean (mu)
    - bn_running_var: BatchNorm2d 的 running variance (sigma^2)
    - bn_eps: BatchNorm2d 层的 epsilon 值

    返回值:
    - 融合后的 Conv2d 层权重和偏置
    """
    # 计算标准差
    std = torch.sqrt(bn_running_var + bn_eps)
    
    # 融合后的 Conv 权重
    fused_weight = bn_weight.view(-1, 1, 1, 1) / std.view(-1, 1, 1, 1) * conv_weight
    
    # 融合后的 Conv 偏置
    fused_bias = bn_weight / std * (-bn_running_mean) + bn_bias
    
    return fused_weight, fused_bias

def manual_fuse_model_bn_to_conv(state_dict: OrderedDict, conv_prefix_list: list, bn_prefix_list: list):
    """
    遍历模型的卷积层，将 BatchNorm2d 层的参数吸收到 Conv2d 层。
    输出的模型会为每个卷积层增加 bias 参数，并移除对应的 BatchNorm2d 层参数。
    """
    new_state_dict = state_dict.copy()
    for conv_prefix, bn_prefix in zip(conv_prefix_list, bn_prefix_list):
        conv_weight = new_state_dict[conv_prefix + '.weight']
        bn_weight = new_state_dict[bn_prefix + '.weight']
        bn_bias = new_state_dict[bn_prefix + '.bias']
        bn_running_mean = new_state_dict[bn_prefix + '.running_mean']
        bn_running_var = new_state_dict[bn_prefix + '.running_var']
        bn_eps = 1e-5

        # 融合 BatchNorm 参数到 Conv 层
        fused_weight, fused_bias = fuse_bn_to_conv(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps)
        
        # 将融合后的权重和偏置写入新的 state dict
        new_state_dict[conv_prefix + '.weight'] = fused_weight
        new_state_dict[conv_prefix + '.bias'] = fused_bias

        # 删除 BatchNorm 层的参数（不再需要）
        del new_state_dict[bn_prefix + '.weight']
        del new_state_dict[bn_prefix + '.bias']
        del new_state_dict[bn_prefix + '.running_mean']
        del new_state_dict[bn_prefix + '.running_var']
        del new_state_dict[bn_prefix + '.num_batches_tracked']

    return new_state_dict

def main():
    # 加载原始权重文件
    checkpoint_path = './logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 将BatchNorm层的参数吸收到Conv层中
    fused_checkpoint = checkpoint.copy()
    conv_prefix_list = ['conv_fc.0', 'conv_fc.3', 'conv_fc.6', 'conv_fc.9', 'conv_fc.12']
    bn_prefix_list = ['conv_fc.1', 'conv_fc.4', 'conv_fc.7', 'conv_fc.10', 'conv_fc.13']
    fused_checkpoint['net'] = manual_fuse_model_bn_to_conv(checkpoint['net'], conv_prefix_list, bn_prefix_list)

    # 保存新的权重文件
    fused_checkpoint_path = './logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_bn2conv.pth'
    torch.save(fused_checkpoint, fused_checkpoint_path)
    print("BatchNorm参数已吸收到Conv层，新权重文件保存为 " + fused_checkpoint_path)

if __name__ == "__main__":
    main()
