## SpikingJelly转PAIBox部分

模型参数转化流程：

1. bn层吸收到conv层

   参考公式：[ANN转换SNN — spikingjelly alpha 文档](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/ann2snn.html#id8)

   找到训练好的模型参数，有[checkpoint_max.pth](logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max.pth)文件

   使用[model_parameters_bn2conv.py](model_parameters_bn2conv.py)进行转化，将bn层吸收到conv层，输出[checkpoint_max_bn2conv.pth](logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_bn2conv.pth)

2. 模型参数键值重命名

   使用[model_parameters_rename.py](model_parameters_rename.py)将键值重命名，以将用于CUDA训练的模型（CextNet）重命名为用于CPU推理的模型（PythonNet），输出[checkpoint_max_bn2conv.pth](logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_bn2conv.pth)

   （可选）使用[infer_bn2conv_380.py](infer_bn2conv_380.py)进行推理，验证bn2conv的性能损失

3. 转为int网络

   使用[model_parameters_conv2int.py](model_parameters_conv2int.py)进行转化，将float网络转为int网络，输出[checkpoint_max_conv2int.pth](logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_conv2int.pth)，输出[vthr_list.npy](logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/vthr_list.npy)

   （可选）使用[infer_conv2int_380.py](infer_conv2int_380.py)进行推理，验证conv2int的性能损失

模型参数转化两个辅助工具：

1. [print_parameters_structure.py](print_parameters_structure.py)

   打印模型权重文件结构，输出到csv文件

2. [print_parameters_statistics.py](print_parameters_statistics.py)

   打印模型各层参数和所有参数的统计量

   在bn层吸收后，按照经验如果所有参数的Max value和Min value的绝对值越大，conv转int后的性能损失越大

## SpikingJelly训练部分

数据预处理工具：

[frames_modify.py](frames_modify.py)处理积分后的'train'和'test'文件夹下的帧文件

在训练前需要对数据进行整体的预处理

训练部分核心代码：

[classify_dvsg_duration_131072_380.py](classify_dvsg_duration_131072_380.py)

训练时加上'-cupy'超参数（使用CextNet），使用混合精度进行训练

## PAIBox推理部分

[paibox_spikingjelly_compare_main.py](paibox_spikingjelly_compare_main.py)

PAIBox网络定义是手写的，与SpikingJelly的网络一致

test_num变量控测试集长度，推理结果输出为csv文件
