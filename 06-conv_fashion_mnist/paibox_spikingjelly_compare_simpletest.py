import argparse
from pathlib import Path

import numpy as np

import paibox as pb
import torch, torchvision
import torch.nn as nn
import csv

from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor

import matplotlib.pyplot as plt

class CSNN(nn.Module):
    def __init__(self, T: int, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
        layer.Conv2d(1, 2, kernel_size=3, padding=1, stride=1, bias=True),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=20.0),
    
        layer.Conv2d(2, 2, kernel_size=3, padding=1, stride=1, bias=True),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=10.0),

        layer.Conv2d(2, 2, kernel_size=3, padding=1, stride=1, bias=True),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=8.0),

        layer.Conv2d(2, 1, kernel_size=3, padding=1, stride=1, bias=True),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=8.0),

        layer.Flatten(),
        layer.Linear(1*3*3, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=30.0),

        layer.Linear(10, 2, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=30.0)
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

class Conv2d_Net(pb.Network):
    def __init__(self, weight0, bias0, Vthr0, weight1, bias1, Vthr1, weight2, bias2, Vthr2, weight3, bias3, Vthr3, weight4, Vthr4, weight5, Vthr5):
        super().__init__()

        self.i0 = pb.InputProj(input=None, shape_out=(1, 3, 3))
        self.n0 = pb.LIF((2, 3, 3), bias=bias0, threshold=Vthr0, reset_v=0, tick_wait_start=1)
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=weight0, padding=1, stride=1)
        
        self.n1 = pb.LIF((2, 3, 3), bias=bias1, threshold=Vthr1, reset_v=0, tick_wait_start=2)
        self.conv2d_1 = pb.Conv2d(self.n0, self.n1, kernel=weight1, padding=1, stride=1)

        self.n2 = pb.LIF((2, 3, 3), bias=bias2, threshold=Vthr2, reset_v=0, tick_wait_start=3)
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=weight2, padding=1, stride=1)

        self.n3 = pb.LIF((1, 3, 3), bias=bias3, threshold=Vthr3, reset_v=0, tick_wait_start=4)
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=weight3, padding=1, stride=1)
        
        self.n4 = pb.LIF(10, threshold=Vthr4, reset_v=0, tick_wait_start=5)
        self.fc4 = pb.FullConn(self.n3, self.n4, weights=weight4, conn_type=pb.SynConnType.All2All)

        self.n5 = pb.LIF(2, threshold=Vthr5, reset_v=0, tick_wait_start=6)
        self.fc5 = pb.FullConn(self.n4, self.n5, weights=weight5, conn_type=pb.SynConnType.All2All)

        self.probe1 = pb.Probe(self.n5, "spike")

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

param_dict = {}


def getNetParam():
    timestep = 4
    layer_num = 6
    delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay


    checkpoint = torch.load('./simpletest_logs/checkpoint2.pt', map_location='cpu', weights_only=True)
    param_dict["w0"]=checkpoint['conv_fc.0.weight'].numpy()
    param_dict["b0"]=checkpoint['conv_fc.0.bias'].numpy()
    param_dict["vthr0"]=20
    param_dict["w1"]=checkpoint['conv_fc.2.weight'].numpy()
    param_dict["b1"]=checkpoint['conv_fc.2.bias'].numpy()
    param_dict["vthr1"]=10
    param_dict["w2"]=checkpoint['conv_fc.4.weight'].numpy()
    param_dict["b2"]=checkpoint['conv_fc.4.bias'].numpy()
    param_dict["vthr2"]=8
    param_dict["w3"]=checkpoint['conv_fc.6.weight'].numpy()
    param_dict["b3"]=checkpoint['conv_fc.6.bias'].numpy()
    param_dict["vthr3"]=8
    param_dict["w4"]=checkpoint['conv_fc.9.weight'].numpy().T
    param_dict["vthr4"]=30
    param_dict["w5"]=checkpoint['conv_fc.11.weight'].numpy().T
    param_dict["vthr5"]=30

if __name__ == "__main__":
    # PAIBox 部分
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="visualize the input data", action="store_true"
    )
    args = parser.parse_args()

    getNetParam()
    pb_net = Conv2d_Net(
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
    param_dict["vthr4"],
    param_dict["w5"],
    param_dict["vthr5"]
    )

    # Network simulation
    raw_data = np.array([[[4,3,2],[1,0,1],[2,3,4]]])
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

    # save output to file paibox_test_paioutput.npz
    # full mode
    # np.savez('paibox_test_paioutput.npz', pb_n0_voltage=pb_n0_voltage, pb_n0_feature_map=pb_n0_feature_map, pb_n1_voltage=pb_n1_voltage, pb_n1_feature_map=pb_n1_feature_map, pb_n2_voltage=pb_n2_voltage, pb_n2_feature_map=pb_n2_feature_map)
    # only feature map
    np.savez('paibox_spikingjelly_compare_simpletest_output_paibox.npz', pb_n0_feature_map=pb_n0_feature_map, pb_n1_feature_map=pb_n1_feature_map, pb_n2_feature_map=pb_n2_feature_map, pb_n3_feature_map=pb_n3_feature_map, pb_n4_feature_map=pb_n4_feature_map, pb_n5_feature_map=pb_n5_feature_map)

    # SpikingJelly 部分

    # SpikingJelly模型初始化
    T = 9
    net = CSNN(T=T, use_cupy=False)
    checkpoint = torch.load('./simpletest_logs/checkpoint2.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint)
    net.eval()

    # 监视器
    spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode)
    
    # 输入数据
    input_data = torch.tensor(raw_data, dtype=torch.float32)

    # 前向传播
    output = net(input_data)

    # 显示SpikingJelly输出
    # print(f'spike_seq_monitor.records=\n{spike_seq_monitor.records}')

    # # 转为numpy数组
    # spike_seq_monitor_records = spike_seq_monitor.records.numpy()

    # # 保存到文件
    # np.savez('spikingjelly_test_sjoutput.npz', spike_seq_monitor_records=spike_seq_monitor_records)

    # print(len(spike_seq_monitor.records)) # = 3
    spike_seq_monitor_records_0 = spike_seq_monitor.records[0].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_1 = spike_seq_monitor.records[1].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_2 = spike_seq_monitor.records[2].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_3 = spike_seq_monitor.records[3].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_4 = spike_seq_monitor.records[4].numpy().squeeze(1).astype(np.int8)
    spike_seq_monitor_records_5 = spike_seq_monitor.records[5].numpy().squeeze(1).astype(np.int8)

    # 保存到文件
    np.savez('paibox_spikingjelly_compare_simpletest_output_spikingjelly.npz', pb_n0_feature_map=spike_seq_monitor_records_0, pb_n1_feature_map=spike_seq_monitor_records_1, pb_n2_feature_map=spike_seq_monitor_records_2, pb_n3_feature_map=spike_seq_monitor_records_3, pb_n4_feature_map=spike_seq_monitor_records_4, pb_n5_feature_map=spike_seq_monitor_records_5)

    # 两个结果可视化
    # pb_n0_feature_map是一个(T, 2, 3, 3) 的numpy数组，表示T个时间步，每个时间步有2个通道，每个通道有3x3的特征图。
    # 要实现可视化，需要分别将特征图转换为灰度图像，并显示出来。可以使用matplotlib库来实现这个功能。
    # matplotlib库最多支持二维的子图可视化，将横轴定义为时间步，纵轴定义为通道，每个子图表示一个特征图。

    # Layer0 (T, 2, 3, 3)
    plt.figure(1,figsize=(2, 8))
    plt.suptitle('PB Layer 0')
    for i in range(T):
        for j in range(2):
            plt.subplot(T, 2, i * 2 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(pb_n0_feature_map[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/PB_L0.png')

    plt.figure(2,figsize=(2, 8))
    plt.suptitle('SJ Layer 0')
    for i in range(T):
        for j in range(2):
            plt.subplot(T, 2, i * 2 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(spike_seq_monitor_records_0[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/SJ_L0.png')

    # Layer1 (T, 2, 3, 3)
    plt.figure(3,figsize=(2, 8))
    plt.suptitle('PB Layer 1')
    for i in range(T):
        for j in range(2):
            plt.subplot(T, 2, i * 2 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(pb_n1_feature_map[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/PB_L1.png')

    plt.figure(4,figsize=(2, 8))
    plt.suptitle('SJ Layer 1')
    for i in range(T):
        for j in range(2):
            plt.subplot(T, 2, i * 2 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(spike_seq_monitor_records_1[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/SJ_L1.png')

    # Layer2 (T, 2, 3, 3)
    plt.figure(5,figsize=(2, 8))
    plt.suptitle('PB Layer 2')
    for i in range(T):
        for j in range(2):
            plt.subplot(T, 2, i * 2 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(pb_n2_feature_map[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/PB_L2.png')

    plt.figure(6,figsize=(2, 8))
    plt.suptitle('SJ Layer 2')
    for i in range(T):
        for j in range(2):
            plt.subplot(T, 2, i * 2 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(spike_seq_monitor_records_2[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/SJ_L2.png')

    # Layer3 (T, 1, 3, 3)
    plt.figure(7,figsize=(2, 8))
    plt.suptitle('PB Layer 3')
    for i in range(T):
        for j in range(1):
            plt.subplot(T, 1, i * 1 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(pb_n3_feature_map[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/PB_L3.png')

    plt.figure(8,figsize=(2, 8))
    plt.suptitle('SJ Layer 3')
    for i in range(T):
        for j in range(1):
            plt.subplot(T, 1, i * 1 + j + 1)
            plt.title(f"T={i}, C={j}")
            plt.imshow(spike_seq_monitor_records_3[i, j, :, :], cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/SJ_L3.png')

    # Layer4 (T, 10)
    plt.figure(9,figsize=(2, 8))
    plt.suptitle('PB Layer 4')
    for i in range(T):
        plt.subplot(T, 1, i + 1)
        plt.title(f"T={i}")
        pb_n4_feature_map_img = np.expand_dims(pb_n4_feature_map[i, :], axis=0)
        plt.imshow(pb_n4_feature_map_img, cmap='coolwarm', vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/PB_L4.png')

    plt.figure(10,figsize=(2, 8))
    plt.suptitle('SJ Layer 4')
    for i in range(T):
        plt.subplot(T, 1, i + 1)
        plt.title(f"T={i}")
        spike_seq_monitor_records_4_img = np.expand_dims(spike_seq_monitor_records_4[i, :], axis=0)
        plt.imshow(spike_seq_monitor_records_4_img, cmap='coolwarm', vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/SJ_L4.png')

    # Layer5 (T, 2)
    plt.figure(11,figsize=(2, 8))
    plt.suptitle('PB Layer 5')
    for i in range(T):
        plt.subplot(T, 1, i + 1)
        plt.title(f"T={i}")
        pb_n5_feature_map_img = np.expand_dims(pb_n5_feature_map[i, :], axis=0)
        plt.imshow(pb_n5_feature_map_img, cmap='coolwarm', vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/PB_L5.png')

    plt.figure(12,figsize=(2, 8))
    plt.suptitle('SJ Layer 5')
    for i in range(T):
        plt.subplot(T, 1, i + 1)
        plt.title(f"T={i}")
        spike_seq_monitor_records_5_img = np.expand_dims(spike_seq_monitor_records_5[i, :], axis=0)
        plt.imshow(spike_seq_monitor_records_5_img, cmap='coolwarm', vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./simpletest_out/SJ_L5.png')

    plt.show()