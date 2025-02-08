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

def maxium_multply_model(model, layer_name_list: list, sj_vthr: float = 1.0):
    """
    每层参数分别缩放并取整数，缩放系数由int(127.0/该层参数最大值)，该缩放系数作为新网络各层LIF的Vthr参数
    在缩放之前，需要将原网络参数除以sj_vthr，以使缩放时的sj_vthr归一化到1.0
    """
    vthr_list = []
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
        normalized_max_value = max_value / sj_vthr # 归一化的最大值
        print("Layer " + layer_name + " max value is " + str(max_value) + ", normalized max value is " + str(normalized_max_value))
        normalized_mult = int(127.0 / float(normalized_max_value))
        mult = float(normalized_mult) / sj_vthr
        print("Layer " + layer_name + " multiply factor is " + str(mult) + ", normalized multiply factor is " + str(normalized_mult))
        if layer_name + '.weight' in model:
            model[layer_name + '.weight'] = torch.round(model[layer_name + '.weight'] * mult).to(torch.int8)
        if layer_name + '.bias' in model:
            model[layer_name + '.bias'] = torch.round(model[layer_name + '.bias'] * mult).to(torch.int8)
        
        # normalized_mult就是转换为int8之后新网络的vthr值
        vthr_list.append(normalized_mult)
    return model, vthr_list

def main():
    # 加载原始权重文件
    checkpoint_path = './logs_897_others/T_16_b_4_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_random_en=True/checkpoint_max_bn2conv.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 将所有参数乘以MULT后取整
    multiplied_checkpoint = checkpoint.copy()
    layer_name_list = ['conv.0', 'conv.2', 'conv.4', 'conv.6', 'fc.2', 'fc.5', 'fc.8']
    multiplied_checkpoint['net'], vthr_list = maxium_multply_model(checkpoint['net'], layer_name_list, sj_vthr=1.0)

    # 保存新的权重文件
    multiplied_checkpoint_path = './logs_897_others/T_16_b_4_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_random_en=True/checkpoint_max_conv2int.pth'
    torch.save(multiplied_checkpoint, multiplied_checkpoint_path)
    print("所有参数乘并转int8完成，新权重文件保存为 " + multiplied_checkpoint_path)

    # 保存vthr_list
    vthr_list_path = './logs_897_others/T_16_b_4_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_random_en=True/vthr_list.npy'
    np.save(vthr_list_path, vthr_list)
    print("转换完成的int8网络 vthr_list 保存路径为 " + vthr_list_path)

if __name__ == "__main__":
    main()

# 运行结果：
# Layer conv.0 max value is 0.906594, normalized max value is 0.9065939784049988
# Layer conv.0 multiply factor is 140.0, normalized multiply factor is 140
# Layer conv.2 max value is 0.90164787, normalized max value is 0.9016478657722473
# Layer conv.2 multiply factor is 140.0, normalized multiply factor is 140
# Layer conv.4 max value is 1.305011, normalized max value is 1.3050110340118408
# Layer conv.4 multiply factor is 97.0, normalized multiply factor is 97
# Layer conv.6 max value is 1.000928, normalized max value is 1.0009280443191528
# Layer conv.6 multiply factor is 126.0, normalized multiply factor is 126
# Layer conv.8 max value is 1.7552505, normalized max value is 1.7552504539489746
# Layer conv.8 multiply factor is 72.0, normalized multiply factor is 72
# Layer conv.10 max value is 0.6547526, normalized max value is 0.6547526121139526
# Layer conv.10 multiply factor is 193.0, normalized multiply factor is 193
# Layer fc.2 max value is 0.053099077, normalized max value is 0.05309907719492912
# Layer fc.2 multiply factor is 2391.0, normalized multiply factor is 2391
# Layer fc.5 max value is 0.082026765, normalized max value is 0.08202676475048065
# Layer fc.5 multiply factor is 1548.0, normalized multiply factor is 1548
# Layer fc.8 max value is 0.09124133, normalized max value is 0.09124132990837097
# Layer fc.8 multiply factor is 1391.0, normalized multiply factor is 1391