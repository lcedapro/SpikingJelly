## 完整的SpikingJelly-PAIBox-PAIBoard工作流

### 1. 准备工作

#### 1.1 环境准备



|         | 1 模型训练 | 2 权重量化 | 3.1 SJ推理 | 3.2 PB推理 | 3.3 推理结果比较 |      |
| ------- | ---------- | ---------- | ---------- | ---------- | ---------------- | ---- |
| pytorch |            |            |            |            |                  |      |
|         |            |            |            |            |                  |      |
|         |            |            |            |            |                  |      |
|         |            |            |            |            |                  |      |
|         |            |            |            |            |                  |      |



### 1 模型训练（需要CUDA CPU）

核心代码：
[davispku_duration_2000_static.py](davispku_duration_2000_static.py)

依赖的自定义模块：
[CustomStaticDataset](CustomStaticDataset.py), 数据集

参考运行示例：
[temp_gpu.sh](temp_gpu.sh)

### 2 权重量化

核心代码：
[model_parameters_quantify_main.py](model_parameters_quantify_main.py)

依赖的自定义模块：
[model_parameters_bn2conv.py](model_parameters_bn2conv.py)
[model_parameters_rename.py](model_parameters_rename.py)
[model_parameters_conv2int.py](model_parameters_conv2int.py)

### 3 推理验证（可跳过）

#### 3.1 SpikingJelly推理

核心代码：
[infer_bn2conv_t1e4.py](infer_bn2conv_t1e4.py), BatchNorm2d层吸收到Conv2d层后的模型推理
[infer_conv2int_t1e4.py](infer_conv2int_t1e4.py), 量化为int后的模型推理

参考运行示例：
在上述两个代码的第一行注释上面

#### 3.2 PAIBox推理

核心代码：
[simple_pb_infer.py](simple_pb_infer.py), PAIBox简单推理

依赖的自定义模块：
[CustomStaticDataset](CustomStaticDataset.py), 数据集

#### 3.3 PB SJ PAIBoard 推理结果比较

PAIBoard可选，需要先编译导出，推理速度较慢

核心代码：
[paibox_spikingjelly_compare_main_t1e4.py](paibox_spikingjelly_compare_main_t1e4.py)

依赖的自定义模块：
[simple_pb_infer.py](simple_pb_infer.py), PB推理
[infer_conv2int_t1e4.py](infer_conv2int_t1e4.py), SJ推理
[CustomStaticDataset](CustomStaticDataset.py), 数据集

### 4 编译导出

核心代码：
[simple_pb_compile.py](simple_pb_compile.py), PAIBox编译

### 5 上板推理（需要PAIBoard_PCIe）

核心代码：
[paiboard_main.py](paiboard_main.py)

依赖的自定义模块：
[events_process.py](events_process.py)
[integrate_events_to_frame.py](integrate_events_to_frame.py)
[paiboard_process.py](paiboard_process.py)
[opencv_process.py](opencv_process.py)


