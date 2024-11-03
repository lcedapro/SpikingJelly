# 下述代码实现将原权重文件的所有参数缩放至-128到127后取整，并打印出新网络各层LIF的Vthr值。原权重文件的参数以float32格式存储，新权重文件的参数以int32格式存储。

import torch
import numpy as np

def multply_model(model, multiplier):
    for name, param in model.items():
        if isinstance(param, torch.Tensor):
            # model[name] = (param * multiplier).to(torch.int32)
            model[name] = (param * multiplier).to(torch.int32)
    return model

def manual_multply_model(model, layer_name_list: list, mult_list: list):
    for layer_name, mult in zip(layer_name_list, mult_list):
        if layer_name + '.weight' in model:
            model[layer_name + '.weight'] = (model[layer_name + '.weight'] * mult).to(torch.int32)
            print("Layer " + layer_name + " weight multiplied by " + str(mult) + " and converted to int32.")
        else:
            print("Layer " + layer_name + " does not exist or has no weight.")
        if layer_name + '.bias' in model:
            model[layer_name + '.bias'] = (model[layer_name + '.bias'] * mult).to(torch.int32)
            print("Layer " + layer_name + " bias multiplied by " + str(mult) + " and converted to int32.")
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
            model[layer_name + '.weight'] = (model[layer_name + '.weight'] * mult).to(torch.int32)
        if layer_name + '.bias' in model:
            model[layer_name + '.bias'] = (model[layer_name + '.bias'] * mult).to(torch.int32)
    return model

def main():
    # 加载原始权重文件
    checkpoint_path = './logs/T4_b768_sgd_lr0.1_c32_amp_cupy/checkpoint_max_bn2conv.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 将所有参数乘以MULT后取整
    multiplied_checkpoint = checkpoint.copy()
    layer_name_list = ['conv_fc.0', 'conv_fc.2', 'conv_fc.4', 'conv_fc.6', 'conv_fc.9', 'conv_fc.11']
    multiplied_checkpoint['net'] = maxium_multply_model(checkpoint['net'], layer_name_list)

    # 保存新的权重文件
    multiplied_checkpoint_path = './logs/T4_b768_sgd_lr0.1_c32_amp_cupy/checkpoint_max_conv2int.pth'
    torch.save(multiplied_checkpoint, multiplied_checkpoint_path)
    print("所有参数乘并转int32完成，新权重文件保存为 " + multiplied_checkpoint_path)

if __name__ == "__main__":
    main()

# 运行结果：
# Layer conv_fc.0 max value is 3.5678024
# Layer conv_fc.0 multiply factor is 35
# Layer conv_fc.2 max value is 1.7235727
# Layer conv_fc.2 multiply factor is 73
# Layer conv_fc.4 max value is 1.7206967
# Layer conv_fc.4 multiply factor is 73
# Layer conv_fc.6 max value is 1.924712
# Layer conv_fc.6 multiply factor is 65
# Layer conv_fc.9 max value is 0.043127324
# Layer conv_fc.9 multiply factor is 2944
# Layer conv_fc.11 max value is 0.12744989
# Layer conv_fc.11 multiply factor is 996