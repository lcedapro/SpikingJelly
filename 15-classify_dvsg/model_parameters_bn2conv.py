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


def fuse_model_bn_to_conv(state_dict):
    """
    遍历模型的卷积层，将 BatchNorm2d 层的参数吸收到 Conv2d 层。
    输出的模型会为每个卷积层增加 bias 参数，并移除对应的 BatchNorm2d 层参数。
    """
    new_state_dict = OrderedDict()

    # 遍历模型中的卷积层（conv.x.0.weight）
    for key in list(state_dict.keys()):
        if 'conv' in key and '0.weight' in key:  # 只匹配卷积层的权重
            # 获取卷积层和 BatchNorm 层的参数前缀
            prefix = key.rsplit('.', 2)[0]  # e.g., 'conv.0'
            conv_weight = state_dict[key]  # Conv2d 的权重

            # 获取 BatchNorm 层的参数
            bn_weight = state_dict[prefix + '.1.weight']  # BatchNorm 的 weight (gamma)
            bn_bias = state_dict[prefix + '.1.bias']  # BatchNorm 的 bias (beta)
            bn_running_mean = state_dict[prefix + '.1.running_mean']  # running mean (mu)
            bn_running_var = state_dict[prefix + '.1.running_var']  # running var (sigma^2)
            bn_eps = 1e-5  # BatchNorm 的 epsilon

            # 融合 Conv 和 BatchNorm 的参数
            fused_weight, fused_bias = fuse_bn_to_conv(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps)

            # 将融合后的权重和偏置写入新的 state dict
            new_state_dict[prefix + '.0.weight'] = fused_weight
            new_state_dict[prefix + '.0.bias'] = fused_bias

            # 删除 BatchNorm 层的参数（不再需要）
            del state_dict[prefix + '.1.weight']
            del state_dict[prefix + '.1.bias']
            del state_dict[prefix + '.1.running_mean']
            del state_dict[prefix + '.1.running_var']
            del state_dict[prefix + '.1.num_batches_tracked']

        elif not key.startswith('conv'):
            # 仅复制非 'conv' 开头的层
            new_state_dict[key] = state_dict[key]

    return new_state_dict

def main():
    # 加载原始权重文件
    checkpoint_path = './logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 将BatchNorm层的参数吸收到Conv层中
    fused_checkpoint = checkpoint.copy()
    fused_checkpoint['net'] = fuse_model_bn_to_conv(checkpoint['net'])

    # 保存新的权重文件
    fused_checkpoint_path = './logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_bn2conv.pth'
    torch.save(fused_checkpoint, fused_checkpoint_path)
    print("BatchNorm参数已吸收到Conv层，新权重文件保存为 " + fused_checkpoint_path)
    print("注意：如果要进行cpu推理验证，还需要进行模型参数键名重命名")

if __name__ == "__main__":
    main()
