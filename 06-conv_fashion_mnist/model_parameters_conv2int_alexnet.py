# 下述代码实现将原权重文件的所有参数缩放至-128到127后取整，并打印出新网络各层LIF的Vthr值。原权重文件的参数以float32格式存储，新权重文件的参数以int32格式存储。

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
            print("Layer " + layer_name + " weight multiplied by " + str(mult) + " and converted to int32.")
        else:
            print("Layer " + layer_name + " does not exist or has no weight.")
        if layer_name + '.bias' in model:
            model[layer_name + '.bias'] = (model[layer_name + '.bias'] * mult).to(torch.int8)
            print("Layer " + layer_name + " bias multiplied by " + str(mult) + " and converted to int32.")
        else:
            print("Layer " + layer_name + " does not exist or has no bias.")
    return model

def maxium_multply_model(model, layer_name_list: list):
    """
    每层参数分别缩放并取整数，缩放系数由int(127.0/该层参数最大值)，该缩放系数作为新网络各层LIF的Vthr参数
    """
    multiplied_model = model.copy()
    for layer_name in layer_name_list:
        max_weight = 0
        max_bias = 0
        if layer_name + '.weight' in model:
            weight_array = model[layer_name + '.weight'].detach().cpu().numpy()
            max_weight = np.max(np.abs(weight_array))
            print("Layer " + layer_name + " max weight is " + str(max_weight))
        if layer_name + '.bias' in model:
            bias_array = model[layer_name + '.bias'].detach().cpu().numpy()
            max_bias = np.max(np.abs(bias_array))
            print("Layer " + layer_name + " max bias is " + str(max_bias))
        max_value = max(max_weight, max_bias)
        print("Layer " + layer_name + " max value is " + str(max_value))
        mult = int(127.0 / float(max_value))
        print("Layer " + layer_name + " multiply factor is " + str(mult))
        if layer_name + '.weight' in model:
            multiplied_model[layer_name + '.weight'] = (model[layer_name + '.weight'] * mult).to(torch.int8)
        if layer_name + '.bias' in model:
            multiplied_model[layer_name + '.bias'] = (model[layer_name + '.bias'] * mult).to(torch.int8)
    return multiplied_model

def main():
    # 加载原始权重文件
    checkpoint_path = './logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_bn2conv.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 将所有参数乘以MULT后取整
    multiplied_checkpoint = checkpoint.copy()
    layer_name_list = ['conv_fc.0', 'conv_fc.2', 'conv_fc.4', 'conv_fc.6', 'conv_fc.8', 'conv_fc.11', 'conv_fc.13', 'conv_fc.15']
    multiplied_checkpoint['net'] = maxium_multply_model(checkpoint['net'], layer_name_list)

    # 保存新的权重文件
    multiplied_checkpoint_path = './logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_conv2int.pth'
    torch.save(multiplied_checkpoint, multiplied_checkpoint_path)
    print("所有参数乘并转int8完成，新权重文件保存为 " + multiplied_checkpoint_path)

if __name__ == "__main__":
    main()

# 运行结果：
# Layer conv_fc.0 max weight is 2.382504
# Layer conv_fc.0 max bias is 1.0698292
# Layer conv_fc.0 max value is 2.382504
# Layer conv_fc.0 multiply factor is 53
# Layer conv_fc.2 max weight is 0.2496002
# Layer conv_fc.2 max bias is 1.7862588
# Layer conv_fc.2 max value is 1.7862588
# Layer conv_fc.2 multiply factor is 71
# Layer conv_fc.4 max weight is 0.3296965
# Layer conv_fc.4 max bias is 1.3522851
# Layer conv_fc.4 max value is 1.3522851
# Layer conv_fc.4 multiply factor is 93
# Layer conv_fc.6 max weight is 0.2619039
# Layer conv_fc.6 max bias is 0.8791547
# Layer conv_fc.6 max value is 0.8791547
# Layer conv_fc.6 multiply factor is 144
# Layer conv_fc.8 max weight is 0.30725917
# Layer conv_fc.8 max bias is 1.593527
# Layer conv_fc.8 max value is 1.593527
# Layer conv_fc.8 multiply factor is 79
# Layer conv_fc.11 max weight is 0.051310923
# Layer conv_fc.11 max value is 0.051310923
# Layer conv_fc.11 multiply factor is 2475
# Layer conv_fc.13 max weight is 0.09354615
# Layer conv_fc.13 max value is 0.09354615
# Layer conv_fc.13 multiply factor is 1357
# Layer conv_fc.15 max weight is 0.32684073
# Layer conv_fc.15 max value is 0.32684073
# Layer conv_fc.15 multiply factor is 388