## SpikingJelly转PAIBox部分

（见[../15-classify_dvsg/README.md](../15-classify_dvsg/README.md)）

## SpikingJelly训练部分

自定义数据集：

[CustomImageDataset.py](CustomImageDataset.py)自定义DAVISPKU数据集，在__getitem__中对事件流进行T轴上的随机裁剪

训练部分核心代码：

[davispku_duration_1000_897.py](davispku_duration_1000_897.py)

由于数据集较小，使用batch_size=4

798指的是在[PB网络设计](paibox_spikingjelly_compare_main_debug_davispku.ipynb)中估算的"n_core_required"=798

## PAIBox推理及导出部分

PB和SJ推理结果比较（在int8量化下）：

[paibox_spikingjelly_compare_main_897.py](paibox_spikingjelly_compare_main_897.py)

在整个测试集上进行测试，推理结果输出为csv文件

PB简单推理仿真及网络导出：

