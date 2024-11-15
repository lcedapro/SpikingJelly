import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import paibox as pb
import torch, torchvision
import csv

from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
# from infer_conv2int import CSNN

import matplotlib.pyplot as plt

class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential( # input = 1 * 28 * 28 = 784 AlexNet Pool1 (channels=128)
        layer.Conv2d(1, channels*2, kernel_size=5, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=13515.0), # 26 * 26 * 2C AlexNet CL2 # 53*255 SpikingJelly使用float输入，范围0-1；PAIbox使用int输入，范围0-255
    
        layer.Conv2d(channels*2, channels*2, kernel_size=4, padding=1, stride=2, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=71.0), # 13 * 13 * 2C AlexNet Pool2

        layer.Conv2d(channels*2, channels*3, kernel_size=3, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*3),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=93.0), # 13 * 13 * 3C AlexNet CL3

        # layer.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*3),
        # neuron.IFNode(surrogate_function=surrogate.ATan()), # 13 * 13 * 3C AlexNet CL4

        layer.Conv2d(channels*3, channels*2, kernel_size=3, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=144.0), # 13 * 13 * 2C AlexNet Pool2

        layer.Conv2d(channels*2, channels*2, kernel_size=3, padding=0, stride=2, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=79.0), # 6 * 6 * 2C AlexNet CL5

        layer.Flatten(),
        layer.Linear(channels*2*6*6, channels*2*16, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=2475.0), # 16C AlexNet FCL6

        layer.Linear(channels*2*16, channels*2*4, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=1357.0), # 16C AlexNet FCL7

        layer.Linear(channels*2*4, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=388.0), # output = 10
        )

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3]

class Conv2d_Net(pb.Network):
    def __init__(self, channels, weight0, bias0, Vthr0, weight1, bias1, Vthr1, weight2, bias2, Vthr2, weight3, bias3, Vthr3, weight4, bias4, Vthr4, weight5, Vthr5, weight6, Vthr6, weight7, Vthr7):
        super().__init__()

        self.i0 = pb.InputProj(input=None, shape_out=(1, 28, 28))
        self.n0 = pb.LIF((channels*2, 26, 26), bias=bias0, threshold=Vthr0, reset_v=0, tick_wait_start=1) # 26 * 26 * 2C AlexNet CL2
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=weight0, padding=1, stride=1)
        
        self.n1 = pb.LIF((channels*2, 13, 13), bias=bias1, threshold=Vthr1, reset_v=0, tick_wait_start=2) # 13 * 13 * 2C AlexNet CL2
        self.conv2d_1 = pb.Conv2d(self.n0, self.n1, kernel=weight1, padding=1, stride=2)
        
        self.n2 = pb.LIF((channels*3, 13, 13), bias=bias2, threshold=Vthr2, reset_v=0, tick_wait_start=3) # 13 * 13 * 3C AlexNet CL3
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=weight2, padding=1, stride=1)
        
        self.n3 = pb.LIF((channels*2, 13, 13), bias=bias3, threshold=Vthr3, reset_v=0, tick_wait_start=4) # 13 * 13 * 2C AlexNet Pool2
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=weight3, padding=1, stride=1)
        
        self.n4 = pb.LIF((channels*2, 6, 6), bias=bias4, threshold=Vthr4, reset_v=0, tick_wait_start=5) # 6 * 6 * 2C AlexNet CL5
        self.conv2d_4 = pb.Conv2d(self.n3, self.n4, kernel=weight4, padding=0, stride=2)
        
        self.n5 = pb.LIF(channels*2*16, threshold=Vthr5, reset_v=0, tick_wait_start=6) # 16C AlexNet FCL6
        self.fc5 = pb.FullConn(self.n4, self.n5, weights=weight5, conn_type=pb.SynConnType.All2All)
        
        self.n6 = pb.LIF(channels*2*4, threshold=Vthr6, reset_v=0, tick_wait_start=7) # 16C AlexNet FCL7
        self.fc6 = pb.FullConn(self.n5, self.n6, weights=weight6, conn_type=pb.SynConnType.All2All)
        
        self.n7 = pb.LIF(10, threshold=Vthr7, reset_v=0, tick_wait_start=8) # output = 10
        self.fc7 = pb.FullConn(self.n6, self.n7, weights=weight7, conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n7, "spike")

        self.pb_n0_voltage = pb.Probe(self.n0, "voltage")
        self.pb_n0_feature_map = pb.Probe(self.n0, "feature_map")
        self.pb_n1_voltage = pb.Probe(self.n1, "voltage")
        self.pb_n1_feature_map = pb.Probe(self.n1, "feature_map")
        self.pb_n2_voltage = pb.Probe(self.n2, "voltage")
        self.pb_n2_feature_map = pb.Probe(self.n2, "feature_map")
        self.pb_n3_voltage = pb.Probe(self.n3, "voltage")
        self.pb_n3_feature_map = pb.Probe(self.n3, "feature_map")
        self.pb_n4_voltage = pb.Probe(self.n4, "voltage")
        self.pb_n4_feature_map = pb.Probe(self.n4, "feature_map")
        self.pb_n5_voltage = pb.Probe(self.n5, "voltage")
        self.pb_n5_feature_map = pb.Probe(self.n5, "feature_map")
        self.pb_n6_voltage = pb.Probe(self.n6, "voltage")
        self.pb_n6_feature_map = pb.Probe(self.n6, "feature_map")
        self.pb_n7_voltage = pb.Probe(self.n7, "voltage")
        self.pb_n7_feature_map = pb.Probe(self.n7, "feature_map")

param_dict = {}


def getNetParam():
    timestep = 8
    layer_num = 8
    delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay

    checkpoint = torch.load('./logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
    param_dict["w0"]=checkpoint['net']['conv_fc.0.weight'].numpy()
    param_dict["b0"]=checkpoint['net']['conv_fc.0.bias'].numpy()
    param_dict["vthr0"]=13515 # 53*255 SpikingJelly使用float输入，范围0-1；PAIbox使用int输入，范围0-255
    param_dict["w1"]=checkpoint['net']['conv_fc.2.weight'].numpy()
    param_dict["b1"]=checkpoint['net']['conv_fc.2.bias'].numpy()
    param_dict["vthr1"]=71
    param_dict["w2"]=checkpoint['net']['conv_fc.4.weight'].numpy()
    param_dict["b2"]=checkpoint['net']['conv_fc.4.bias'].numpy()
    param_dict["vthr2"]=93
    param_dict["w3"]=checkpoint['net']['conv_fc.6.weight'].numpy()
    param_dict["b3"]=checkpoint['net']['conv_fc.6.bias'].numpy()
    param_dict["vthr3"]=144
    param_dict["w4"]=checkpoint['net']['conv_fc.8.weight'].numpy()
    param_dict["b4"]=checkpoint['net']['conv_fc.8.bias'].numpy()
    param_dict["vthr4"]=79
    param_dict["w5"]=checkpoint['net']['conv_fc.11.weight'].numpy().T
    param_dict["vthr5"]=2475
    param_dict["w6"]=checkpoint['net']['conv_fc.13.weight'].numpy().T
    param_dict["vthr6"]=1357
    param_dict["w7"]=checkpoint['net']['conv_fc.15.weight'].numpy().T
    param_dict["vthr7"]=388

if __name__ == "__main__":

    # PAIBox 部分

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="visualize the input data", action="store_true"
    )
    args = parser.parse_args()

    getNetParam()
    pb_net = Conv2d_Net(16,
        param_dict["w0"],
        param_dict["b0"],
        param_dict["vthr0"],
        param_dict["w1"],
        param_dict["b1"],
        param_dict["vthr1"],
        param_dict["w2"],
        param_dict["b2"],
        param_dict["vthr2"],
        param_dict["w3"],
        param_dict["b3"],
        param_dict["vthr3"],
        param_dict["w4"],
        param_dict["b4"],
        param_dict["vthr4"],
        param_dict["w5"],
        param_dict["vthr5"],
        param_dict["w6"],
        param_dict["vthr6"],
        param_dict["w7"],
        param_dict["vthr7"]
    )

    # Network simulation
    image_array_path = './save_dir/4.npy'
    raw_data = np.load(image_array_path)
    input_data = raw_data.ravel()

    # Visualize
    if args.verbose:
        print(raw_data)

    # Input
    pb_net.i0.input = input_data

    # Simulation, duration=timestep + delay
    sim = pb.Simulator(pb_net)
    sim.run(param_dict["timestep"] + param_dict["delay"], reset=False)

    # Print all the results
    pb_n0_voltage = sim.data[pb_net.pb_n0_voltage]
    pb_n0_feature_map = sim.data[pb_net.pb_n0_feature_map]
    pb_n1_voltage = sim.data[pb_net.pb_n1_voltage]
    pb_n1_feature_map = sim.data[pb_net.pb_n1_feature_map]
    pb_n2_voltage = sim.data[pb_net.pb_n2_voltage]
    pb_n2_feature_map = sim.data[pb_net.pb_n2_feature_map]
    pb_n3_voltage = sim.data[pb_net.pb_n3_voltage]
    pb_n3_feature_map = sim.data[pb_net.pb_n3_feature_map]
    pb_n4_voltage = sim.data[pb_net.pb_n4_voltage]
    pb_n4_feature_map = sim.data[pb_net.pb_n4_feature_map]
    pb_n5_voltage = sim.data[pb_net.pb_n5_voltage]
    pb_n5_feature_map = sim.data[pb_net.pb_n5_feature_map]
    pb_n6_voltage = sim.data[pb_net.pb_n6_voltage]
    pb_n6_feature_map = sim.data[pb_net.pb_n6_feature_map]
    pb_n7_voltage = sim.data[pb_net.pb_n7_voltage]
    pb_n7_feature_map = sim.data[pb_net.pb_n7_feature_map]

    # shape of PB pb_n0_feature_map (tuple)
    print("PB pb_n0_feature_map shape:{}".format(np.array(pb_n0_feature_map).shape)) # (15,32,26,26)=(T,C,H,W)
    print("PB pb_n5_feature_map shape:{}".format(np.array(pb_n5_feature_map).shape)) # (15,512)=(T,N)

    # SpikingJelly 部分

    # SpikingJelly模型初始化
    T = 8
    net = CSNN(T=T, channels=16, use_cupy=False)
    checkpoint = torch.load('./logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # 监视器
    spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode)

    # 输入数据
    input_data = torch.tensor(raw_data, dtype=torch.float32)

    # 前向传播
    output = net(input_data)

    spike_seq_monitor_records_0 = spike_seq_monitor.records[0].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_1 = spike_seq_monitor.records[1].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_2 = spike_seq_monitor.records[2].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_3 = spike_seq_monitor.records[3].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_4 = spike_seq_monitor.records[4].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_5 = spike_seq_monitor.records[5].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_6 = spike_seq_monitor.records[6].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_7 = spike_seq_monitor.records[7].numpy().squeeze(1).astype(np.int8)

    # shape of spike_seq_monitor_records_0 (tuple)
    print("SP spike_seq_monitor_records_0 shape:{}".format(np.array(spike_seq_monitor_records_0).shape)) # (8,32,26,26)=(T,C,H,W)
    print("SP spike_seq_monitor_records_5 shape:{}".format(np.array(spike_seq_monitor_records_5).shape)) # (8,512)=(T,N)

    # 两个结果可视化（同一层）
    # pb_n0_feature_map和spike_seq_monitor_records_0是一个(T, C, H, W) 的numpy数组，表示T个时间步，每个时间步有C个通道，每个通道有HxW的特征图。
    # 要实现可视化，需要分别将特征图转换为灰度图像，并显示出来。可以使用matplotlib库来实现这个功能。
    # matplotlib库最多支持二维的子图可视化，将横轴定义为时间步，纵轴定义为通道，每个子图表示一个特征图。

    # 由于显示尺寸有限，32个通道只显示前16个，其余的不显示；8个时间步全部显示。则需要定义一个16x8的子图。
    # fig, axs = plt.subplots(16, 8, figsize=(10, 8))
    # fig.suptitle('pb_n0_feature_map')
    # for i in range(16):
    #     for j in range(8):
    #         axs[i, j].imshow(pb_n0_feature_map[j, i, :, :], cmap='gray')
    #         axs[i, j].axis('off')
    # fig.tight_layout()
    
    # fig, axs = plt.subplots(16, 8, figsize=(10, 8))
    # fig.suptitle('spike_seq_monitor_records_0')
    # for i in range(16):
    #     for j in range(8):
    #         axs[i, j].imshow(spike_seq_monitor_records_0[j, i, :, :], cmap='gray')
    #         axs[i, j].axis('off')
    # fig.tight_layout()
    
    # fig, axs = plt.subplots(16, 8, figsize=(10, 8))
    # fig.suptitle('pb_n0_feature_map - spike_seq_monitor_records_0')
    # for i in range(16):
    #     for j in range(8):
    #         axs[i, j].imshow(pb_n0_feature_map[j, i, :, :] - spike_seq_monitor_records_0[j, i, :, :], cmap='gray')
    #         axs[i, j].axis('off')
    # fig.tight_layout()
    # plt.show()

    # 两个结果可视化（不同层，选取第0个通道）

    # 8层网络全部显示，其中前5层为卷积层，输出的特征图直接是二维图像((T,C,H,W)中的(H,W))，后3层为全连接层，输出的特征图需要拓展为二维((T,N)中的(N,)拓展为(N,1))；8个时间步全部显示。则需要定义一个5x8的子图。
    # 其中，对于PAIBox的运行结果需要特殊处理。PAIBox每层会产生一个时钟周期的延迟，因此每层要显示的结果，其时间步要+1

    fig, axs = plt.subplots(8, 8, figsize=(10, 8))
    for j in range(8):
        axs[0, j].imshow(pb_n0_feature_map[j, 0, :, :], cmap='gray')
        axs[0, j].axis('off')
        axs[1, j].imshow(pb_n1_feature_map[j+1, 0, :, :], cmap='gray')
        axs[1, j].axis('off')
        axs[2, j].imshow(pb_n2_feature_map[j+2, 0, :, :], cmap='gray')
        axs[2, j].axis('off')
        axs[3, j].imshow(pb_n3_feature_map[j+3, 0, :, :], cmap='gray')
        axs[3, j].axis('off')
        axs[4, j].imshow(pb_n4_feature_map[j+4, 0, :, :], cmap='gray')
        axs[4, j].axis('off')
        axs[5, j].imshow(np.expand_dims(pb_n5_feature_map[j+5, :], axis=0), cmap='gray')
        axs[5, j].axis('off')
        axs[6, j].imshow(np.expand_dims(pb_n6_feature_map[j+6, :], axis=0), cmap='gray')
        axs[6, j].axis('off')
        axs[7, j].imshow(np.expand_dims(pb_n7_feature_map[j+7, :], axis=0), cmap='gray')
        axs[7, j].axis('off')
    fig.tight_layout()

    fig, axs = plt.subplots(8, 8, figsize=(10, 8))
    for j in range(8):
        axs[0, j].imshow(spike_seq_monitor_records_0[j, 0, :, :], cmap='gray')
        axs[0, j].axis('off')
        axs[1, j].imshow(spike_seq_monitor_records_1[j, 0, :, :], cmap='gray')
        axs[1, j].axis('off')
        axs[2, j].imshow(spike_seq_monitor_records_2[j, 0, :, :], cmap='gray')
        axs[2, j].axis('off')
        axs[3, j].imshow(spike_seq_monitor_records_3[j, 0, :, :], cmap='gray')
        axs[3, j].axis('off')
        axs[4, j].imshow(spike_seq_monitor_records_4[j, 0, :, :], cmap='gray')
        axs[4, j].axis('off')
        axs[5, j].imshow(np.expand_dims(spike_seq_monitor_records_5[j, :], axis=0), cmap='gray')
        axs[5, j].axis('off')
        axs[6, j].imshow(np.expand_dims(spike_seq_monitor_records_6[j, :], axis=0), cmap='gray')
        axs[6, j].axis('off')
        axs[7, j].imshow(np.expand_dims(spike_seq_monitor_records_7[j, :], axis=0), cmap='gray')
        axs[7, j].axis('off')
    fig.tight_layout()


    # 对于差分结果，如果大于0，则显示为红色；如果小于0，则显示为蓝色。
    # 先统计所有误差值中的最大值和最小值以便获得最佳显示效果
    diff_max = 0
    diff_min = 0
    for j in range(8):
        diff_max = max(diff_max, np.max(pb_n0_feature_map[j, 0, :, :] - spike_seq_monitor_records_0[j, 0, :, :]))
        diff_min = min(diff_min, np.min(pb_n0_feature_map[j, 0, :, :] - spike_seq_monitor_records_0[j, 0, :, :]))
        diff_max = max(diff_max, np.max(pb_n1_feature_map[j+1, 0, :, :] - spike_seq_monitor_records_1[j, 0, :, :]))
        diff_min = min(diff_min, np.min(pb_n1_feature_map[j+1, 0, :, :] - spike_seq_monitor_records_1[j, 0, :, :]))
        diff_max = max(diff_max, np.max(pb_n2_feature_map[j+2, 0, :, :] - spike_seq_monitor_records_2[j, 0, :, :]))
        diff_min = min(diff_min, np.min(pb_n2_feature_map[j+2, 0, :, :] - spike_seq_monitor_records_2[j, 0, :, :]))
        diff_max = max(diff_max, np.max(pb_n3_feature_map[j+3, 0, :, :] - spike_seq_monitor_records_3[j, 0, :, :]))
        diff_min = min(diff_min, np.min(pb_n3_feature_map[j+3, 0, :, :] - spike_seq_monitor_records_3[j, 0, :, :]))
        diff_max = max(diff_max, np.max(pb_n4_feature_map[j+4, 0, :, :] - spike_seq_monitor_records_4[j, 0, :, :]))
        diff_min = min(diff_min, np.min(pb_n4_feature_map[j+4, 0, :, :] - spike_seq_monitor_records_4[j, 0, :, :]))
        diff_max = max(diff_max, np.max(pb_n5_feature_map[j+5, :] - spike_seq_monitor_records_5[j, :]))
        diff_min = min(diff_min, np.min(pb_n5_feature_map[j+5, :] - spike_seq_monitor_records_5[j, :]))
        diff_max = max(diff_max, np.max(pb_n6_feature_map[j+6, :] - spike_seq_monitor_records_6[j, :]))
        diff_min = min(diff_min, np.min(pb_n6_feature_map[j+6, :] - spike_seq_monitor_records_6[j, :]))
        diff_max = max(diff_max, np.max(pb_n7_feature_map[j+7, :] - spike_seq_monitor_records_7[j, :]))
        diff_min = min(diff_min, np.min(pb_n7_feature_map[j+7, :] - spike_seq_monitor_records_7[j, :]))
    # 显示的阈值中心应该是0，因此取abs(diff_max)和abs(diff_min)中的较大值作为显示阈值
    vthr = max(abs(diff_max), abs(diff_min))
    print(f"max diff = {diff_max}, min diff = {diff_min}, vthr = {vthr}")
    # 如果vthr为0，则设置为1
    if vthr == 0:
        vthr = 1
    
    fig, axs = plt.subplots(8, 8, figsize=(10, 8))
    for j in range(8):
        axs[0, j].imshow(pb_n0_feature_map[j, 0, :, :] - spike_seq_monitor_records_0[j, 0, :, :], cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[0, j].axis('off')
        axs[1, j].imshow(pb_n1_feature_map[j+1, 0, :, :] - spike_seq_monitor_records_1[j, 0, :, :], cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[1, j].axis('off')
        axs[2, j].imshow(pb_n2_feature_map[j+2, 0, :, :] - spike_seq_monitor_records_2[j, 0, :, :], cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[2, j].axis('off')
        axs[3, j].imshow(pb_n3_feature_map[j+3, 0, :, :] - spike_seq_monitor_records_3[j, 0, :, :], cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[3, j].axis('off')
        axs[4, j].imshow(pb_n4_feature_map[j+4, 0, :, :] - spike_seq_monitor_records_4[j, 0, :, :], cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[4, j].axis('off')
        axs[5, j].imshow(np.expand_dims(pb_n5_feature_map[j+5, :] - spike_seq_monitor_records_5[j, :], axis=0), cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[5, j].axis('off')
        axs[6, j].imshow(np.expand_dims(pb_n6_feature_map[j+6, :] - spike_seq_monitor_records_6[j, :], axis=0), cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[6, j].axis('off')
        axs[7, j].imshow(np.expand_dims(pb_n7_feature_map[j+7, :] - spike_seq_monitor_records_7[j, :], axis=0), cmap='RdBu', vmin=-vthr, vmax=vthr)
        axs[7, j].axis('off')
    fig.tight_layout()


    plt.show()