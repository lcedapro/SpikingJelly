import torch
import torch.nn as nn
from collections import OrderedDict

def manual_rename(state_dict:OrderedDict, src:list, dst:list):
    new_state_dict = state_dict.copy()
    for key, value in state_dict.items():
        if key in src:
            # pop
            new_state_dict.pop(key)
            # add
            new_state_dict[dst[src.index(key)]] = value
    return new_state_dict

def main():
    # 加载原始权重文件
    checkpoint_path = './logs/T_16_b_48_c_32_Adam_lr_0.01_CosALR_16_amp_cupy_3/checkpoint_max_bn2conv.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 手动指定重命名前后键名
    src = ["conv.0.0.weight", "conv.0.0.bias", "conv.2.0.weight", "conv.2.0.bias", "conv.4.0.weight", "conv.4.0.bias", "conv.6.0.weight", "conv.6.0.bias", "conv.8.0.weight", "conv.8.0.bias", "conv.10.0.weight", "conv.10.0.bias", "conv.12.0.weight", "conv.12.0.bias", "conv.14.0.weight", "conv.14.0.bias", "conv.16.0.weight", "conv.16.0.bias", "conv.18.0.weight", "conv.18.0.bias", "fc.2.0.weight", "fc.5.0.weight"]
    dst = ["conv.0.weight", "conv.0.bias", "conv.2.weight", "conv.2.bias", "conv.4.weight", "conv.4.bias", "conv.6.weight", "conv.6.bias", "conv.8.weight", "conv.8.bias", "conv.10.weight", "conv.10.bias", "conv.12.weight", "conv.12.bias", "conv.14.weight", "conv.14.bias", "conv.16.weight", "conv.16.bias", "conv.18.weight", "conv.18.bias", "fc.2.weight", "fc.5.weight"]

    # 将模型中的指定参数重命名
    fused_checkpoint = checkpoint.copy()
    fused_checkpoint['net'] = manual_rename(checkpoint['net'], src, dst)

    # 保存新的权重文件
    fused_checkpoint_path = './logs/T_16_b_48_c_32_Adam_lr_0.01_CosALR_16_amp_cupy_3/checkpoint_max_bn2conv.pth'
    torch.save(fused_checkpoint, fused_checkpoint_path)
    print("参数重命名完成，新权重文件保存为 " + fused_checkpoint_path)

if __name__ == "__main__":
    main()
