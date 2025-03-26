# 权重量化（SpikingJelly转PAIBox）

from model_parameters_bn2conv import fuse_model_bn_to_conv
from model_parameters_rename import auto_rename
from model_parameters_conv2int import maxium_multply_model
import torch
import numpy as np
import os
state_dict_path = './logs_t1e4_simple/T_16_b_64_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_temporary_datasets'

def main():
    # 加载原始权重文件
    checkpoint_path = os.path.join(state_dict_path, 'checkpoint_max.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 1. bn2conv

    # 将BatchNorm层的参数吸收到Conv层中
    fused_checkpoint = checkpoint.copy()
    fused_checkpoint['net'] = fuse_model_bn_to_conv(checkpoint['net'])

    # 保存新的权重文件
    fused_checkpoint_path = os.path.join(state_dict_path, 'checkpoint_max_bn2conv.pth')
    torch.save(fused_checkpoint, fused_checkpoint_path)
    print("BatchNorm参数已吸收到Conv层，新权重文件保存为 " + fused_checkpoint_path)

    # 2. rename

    # 将模型中的指定参数重命名
    renamed_checkpoint = fused_checkpoint.copy()
    renamed_checkpoint['net'] = auto_rename(fused_checkpoint['net'])

    # 保存新的权重文件
    renamed_checkpoint_path = os.path.join(state_dict_path, 'checkpoint_max_bn2conv.pth')
    torch.save(renamed_checkpoint, renamed_checkpoint_path)
    print("参数重命名完成，新权重文件保存为 " + renamed_checkpoint_path)

    # 3. conv2int
    
    # 将所有参数乘以MULT后取整
    multiplied_checkpoint = renamed_checkpoint.copy()
    multiplied_checkpoint['net'], vthr_list = maxium_multply_model(renamed_checkpoint['net'], sj_vthr=1.0)

    # 保存新的权重文件
    multiplied_checkpoint_path = os.path.join(state_dict_path, 'checkpoint_max_conv2int.pth')
    torch.save(multiplied_checkpoint, multiplied_checkpoint_path)
    print("所有参数乘并转int8完成，新权重文件保存为 " + multiplied_checkpoint_path)

    # 保存vthr_list
    vthr_list_path = os.path.join(state_dict_path, 'vthr_list.npy')
    np.save(vthr_list_path, vthr_list)
    print("转换完成的int8网络 vthr_list 保存路径为 " + vthr_list_path)

if __name__ == "__main__":
    main()
