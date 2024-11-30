# 下述代码实现将原权重文件的所有参数缩放至-128到127后取整，并打印出新网络各层LIF的Vthr值。原权重文件的参数以float32格式存储，新权重文件的参数以int8格式存储。

import torch
import numpy as np

def multply_model(model, multiplier):
    for name, param in model.items():
        if isinstance(param, torch.Tensor):
            # model[name] = (param * multiplier).to(torch.int8)
            model[name] = (param * multiplier).to(torch.int8)
    return model

def manual_multply_model(model, layer_name_list: list, mult_list: list):
    for layer_name, mult in zip(layer_name_list, mult_list):
        if layer_name + '.weight' in model:
            model[layer_name + '.weight'] = (model[layer_name + '.weight'] * mult).to(torch.int8)
            print("Layer " + layer_name + " weight multiplied by " + str(mult) + " and converted to int8.")
        else:
            print("Layer " + layer_name + " does not exist or has no weight.")
        if layer_name + '.bias' in model:
            model[layer_name + '.bias'] = (model[layer_name + '.bias'] * mult).to(torch.int8)
            print("Layer " + layer_name + " bias multiplied by " + str(mult) + " and converted to int8.")
        else:
            print("Layer " + layer_name + " does not exist or has no bias.")
    return model

def maxium_multply_model(model, layer_name_list: list):
    """
    每层参数分别缩放并取整数，缩放系数由int(127.0/该层参数最大值)，该缩放系数作为新网络各层LIF的Vthr参数
    """
    for layer_name in layer_name_list:
        max_weight = 0
        max_bias = 0
        if layer_name + '.weight' in model:
            weight_array = model[layer_name + '.weight'].detach().cpu().numpy()
            max_weight = np.max(np.abs(weight_array))
        if layer_name + '.bias' in model:
            bias_array = model[layer_name + '.bias'].detach().cpu().numpy()
            max_bias = np.max(np.abs(bias_array))
        max_value = max(max_weight, max_bias)
        print("Layer " + layer_name + " max value is " + str(max_value))
        mult = int(127.0 / float(max_value))
        print("Layer " + layer_name + " multiply factor is " + str(mult))
        if layer_name + '.weight' in model:
            model[layer_name + '.weight'] = (model[layer_name + '.weight'] * mult).to(torch.int8)
        if layer_name + '.bias' in model:
            model[layer_name + '.bias'] = (model[layer_name + '.bias'] * mult).to(torch.int8)
    return model

def main():
    # 加载原始权重文件
    checkpoint_path = './logs/T_16_b_48_c_32_Adam_lr_0.01_CosALR_16_amp_cupy_3/checkpoint_max_bn2conv.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 将所有参数乘以MULT后取整
    multiplied_checkpoint = checkpoint.copy()
    layer_name_list = ['conv.0', 'conv.2', 'conv.4', 'conv.6', 'conv.8', 'conv.10', 'conv.12', 'conv.14', 'conv.16', 'conv.18', 'fc.2', 'fc.5']
    multiplied_checkpoint['net'] = maxium_multply_model(checkpoint['net'], layer_name_list)

    # 保存新的权重文件
    multiplied_checkpoint_path = './logs/T_16_b_48_c_32_Adam_lr_0.01_CosALR_16_amp_cupy_3/checkpoint_max_conv2int.pth'
    torch.save(multiplied_checkpoint, multiplied_checkpoint_path)
    print("所有参数乘并转int8完成，新权重文件保存为 " + multiplied_checkpoint_path)

if __name__ == "__main__":
    main()

# 运行结果：
# Layer conv.0 max value is 2.3826818
# Layer conv.0 multiply factor is 53
# Layer conv.2 max value is 0.8683898
# Layer conv.2 multiply factor is 146
# Layer conv.4 max value is 0.94496036
# Layer conv.4 multiply factor is 134
# Layer conv.6 max value is 1.0696411
# Layer conv.6 multiply factor is 118
# Layer conv.8 max value is 1.0184215
# Layer conv.8 multiply factor is 124
# Layer conv.10 max value is 1.2978486
# Layer conv.10 multiply factor is 97
# Layer conv.12 max value is 0.8408925
# Layer conv.12 multiply factor is 151
# Layer conv.14 max value is 1.6707848
# Layer conv.14 multiply factor is 76
# Layer conv.16 max value is 1.1237819
# Layer conv.16 multiply factor is 113
# Layer conv.18 max value is 1.2791295
# Layer conv.18 multiply factor is 99
# Layer fc.2 max value is 1.046356
# Layer fc.2 multiply factor is 121
# Layer fc.5 max value is 1.3987551
# Layer fc.5 multiply factor is 90