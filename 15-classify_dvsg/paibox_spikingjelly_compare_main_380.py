# paibox 和 spikingjelly 的推理结果比较，输出为 csv 文件
# 需要修改数据集路径DVS128GESTURE_DATA_DIR
import torch
import numpy as np
import paibox as pb
import csv

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader

from infer_conv2int_380 import PythonNet
from voting import voting
vthr_list = np.load('./logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/vthr_list.npy') # vthr from model_parameters_conv2int.py

SIM_TIMESTEP = 4 # <=16
DVS128GESTURE_DATA_DIR = '../15-classify_dvsg/DVS128Gesture'

# Dataloader
test_set = DVS128Gesture(DVS128GESTURE_DATA_DIR, train=False, data_type='frame', split_by='number', duration=131072)
test_data_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    pin_memory=True)

# PAIBox网络定义
class Conv2d_Net(pb.Network):
    def __init__(self, channels, param_dict):
        super().__init__()

        def fakeout_with_t(t, **kwargs): # ignore other arguments except `t` & `bias`
            if t-1 < SIM_TIMESTEP:
                print(f't = {t}, input = image[{t-1}]')
                return image_64[t-1]
            else:
                print(f't = {t}, input = image[-1]')
                return image_64[-1]

        self.i0 = pb.InputProj(input=fakeout_with_t, shape_out=(2, 64, 64))
        self.n0 = pb.LIF((channels*2, 32, 32), bias=param_dict['conv.0.bias'], threshold=param_dict['conv.0.vthr'], reset_v=0, tick_wait_start=1) # convpool3x3
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=param_dict['conv.0.weight'], padding=2, stride=2)

        self.n1 = pb.LIF((channels*2, 32, 32), bias=param_dict['conv.2.bias'], threshold=param_dict['conv.2.vthr'], reset_v=0, tick_wait_start=2) # conv3x3
        self.conv2d_1 = pb.Conv2d(self.n0, self.n1, kernel=param_dict['conv.2.weight'], padding=2, stride=1)

        self.n2 = pb.LIF((channels*4, 16, 16), bias=param_dict['conv.4.bias'], threshold=param_dict['conv.4.vthr'], reset_v=0, tick_wait_start=3) # convpool3x3
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=param_dict['conv.4.weight'], padding=2, stride=2)

        self.n3 = pb.LIF((channels*4, 16, 16), bias=param_dict['conv.6.bias'], threshold=param_dict['conv.6.vthr'], reset_v=0, tick_wait_start=4) # conv3x3
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=param_dict['conv.6.weight'], padding=2, stride=1)

        self.n4 = pb.LIF((channels*8, 8, 8), bias=param_dict['conv.8.bias'], threshold=param_dict['conv.8.vthr'], reset_v=0, tick_wait_start=5) # convpool3x3
        self.conv2d_4 = pb.Conv2d(self.n3, self.n4, kernel=param_dict['conv.8.weight'], padding=2, stride=2)

        self.n5 = pb.LIF((channels*8, 8, 8), bias=param_dict['conv.10.bias'], threshold=param_dict['conv.10.vthr'], reset_v=0, tick_wait_start=6) # conv3x3
        self.conv2d_5 = pb.Conv2d(self.n4, self.n5, kernel=param_dict['conv.10.weight'], padding=2, stride=1)

        self.n10 = pb.LIF(channels*8 * 4 * 4, threshold=param_dict['fc.2.vthr'], reset_v=0, tick_wait_start=7) # fc
        self.fc_0 = pb.FullConn(self.n5, self.n10, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.2.weight'])

        self.n11 = pb.LIF(channels*8 * 4 * 4, threshold=param_dict['fc.5.vthr'], reset_v=0, tick_wait_start=8) # fc
        self.fc_1 = pb.FullConn(self.n10, self.n11, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.5.weight'])

        self.n12 = pb.LIF(110, threshold=param_dict['fc.8.vthr'], reset_v=0, tick_wait_start=9) # fc
        self.fc_2 = pb.FullConn(self.n11, self.n12, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.8.weight'])

        self.probe1 = pb.Probe(self.n12, "spike")


# PAIBox网络初始化
param_dict = {}
def getNetParam():
    timestep = SIM_TIMESTEP
    layer_num = 9
    delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay

    checkpoint = torch.load('./logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
    param_dict['conv.0.weight']=checkpoint['net']['conv.0.weight'].numpy()
    param_dict['conv.0.bias']=checkpoint['net']['conv.0.bias'].numpy()
    param_dict['conv.2.weight']=checkpoint['net']['conv.2.weight'].numpy()
    param_dict['conv.2.bias']=checkpoint['net']['conv.2.bias'].numpy()
    param_dict['conv.4.weight']=checkpoint['net']['conv.4.weight'].numpy()
    param_dict['conv.4.bias']=checkpoint['net']['conv.4.bias'].numpy()
    param_dict['conv.6.weight']=checkpoint['net']['conv.6.weight'].numpy()
    param_dict['conv.6.bias']=checkpoint['net']['conv.6.bias'].numpy()
    param_dict['conv.8.weight']=checkpoint['net']['conv.8.weight'].numpy()
    param_dict['conv.8.bias']=checkpoint['net']['conv.8.bias'].numpy()
    param_dict['conv.10.weight']=checkpoint['net']['conv.10.weight'].numpy()
    param_dict['conv.10.bias']=checkpoint['net']['conv.10.bias'].numpy()
    param_dict['fc.2.weight']=checkpoint['net']['fc.2.weight'].numpy().T
    param_dict['fc.5.weight']=checkpoint['net']['fc.5.weight'].numpy().T
    param_dict['fc.8.weight']=checkpoint['net']['fc.8.weight'].numpy().T
    param_dict['conv.0.vthr']=vthr_list[0]
    param_dict['conv.2.vthr']=vthr_list[1]
    param_dict['conv.4.vthr']=vthr_list[2]
    param_dict['conv.6.vthr']=vthr_list[3]
    param_dict['conv.8.vthr']=vthr_list[4]
    param_dict['conv.10.vthr']=vthr_list[5]
    param_dict['fc.2.vthr']=vthr_list[6]
    param_dict['fc.5.vthr']=vthr_list[7]
    param_dict['fc.8.vthr']=vthr_list[8]
getNetParam()

# PAIBox仿真器
pb_net = Conv2d_Net(4, param_dict)
sim = pb.Simulator(pb_net)

# PAIBox推理子程序
def pb_inference(image):
    # PAIBox推理

    # Simulation, duration=timestep + delay
    sim.run(param_dict["timestep"] + param_dict["delay"], reset=False)

    # Decode the output
    spike_out = sim.data[pb_net.probe1].astype(np.int32)
    spike_out = spike_out[param_dict["delay"] :]
    spike_out = voting(spike_out, 10)
    spike_sum_pb = spike_out.sum(axis=0)
    pred_pb = np.argmax(spike_sum_pb)
    print("Predicted number:", pred_pb)

    sim.reset()
    return spike_sum_pb, pred_pb

# SpikingJelly网络定义和初始化
vthr_list_tofloat = [float(vthr) for vthr in vthr_list]
net = PythonNet(channels=4, vthr_list=vthr_list_tofloat)
checkpoint = torch.load('./logs_380/T_16_b_256_c_4_SGD_lr_0.2_CosALR_48_amp_cupy-4/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
net.load_state_dict(checkpoint['net'])
net.eval()

# SpikingJelly推理子程序
def sj_inference(image):
    # SpikingJelly推理
    with torch.no_grad():
        out_fr = net(torch.tensor(image).unsqueeze(0).float())
        pred_sj = out_fr.argmax(1).item()
        # test_acc_sj += (out_fr.argmax(1) == label).float().sum().item()
        functional.reset_net(net)
    spike_sum_sj = out_fr.squeeze().numpy()
    spike_sum_sj = (spike_sum_sj*10*SIM_TIMESTEP).round().astype(np.int32)
    return spike_sum_sj, pred_sj

# 测试程序
def test(test_num: int = 100):
    # 结果csv初始化
    with open('paibox_spikingjelly_compare_main_380_result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "pb_spike_sum", "pb_pred", "pb_correct", "sj_spike_sum", "sj_pred", "sj_correct"])

    # 测试主程序
    test_acc_pb = 0
    test_acc_sj = 0
    test_samples = 0
    for i, (image_tensor, label_tensor) in enumerate(test_data_loader):
        if i == test_num:
            break
        print(f"Test sample {i}")
        # 仿真时间 [N, T, C, H, W] -> [N, T=SIM_TIMESTEP, C, H, W]
        image_tensor = image_tensor[:, :SIM_TIMESTEP, :, :, :]

        # 数据集预处理
        # 获取图片和标签
        image, label = image_tensor[0], label_tensor[0]
        # 图片转为 numpy 数组，标签转为 int
        image = image.squeeze()  # 去掉批次维度
        image = image.numpy()  # 转换为 numpy 数组
        image = image.astype(np.uint8)  # 转换为 uint8
        label = label.item()
        # 修改数据，将大于等于1的像素值设为1，否则为0
        image = np.where(image == 0, 0, 1)

        # 对于新的PAIBox网络，需要预先将image的形状从[2, 128, 128]用maxpool2d池化为[2, 64, 64]
        global image_64
        image_64 = image_tensor[0]
        maxpool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        image_64 = maxpool2d(image_64)
        image_64 = image_64.squeeze()  # 去掉批次维度
        image_64 = image_64.numpy()  # 转换为 numpy 数组
        image_64 = image_64.astype(np.uint8)  # 转换为 uint8

        test_samples += 1

        # PAIBox推理
        spike_sum_pb, pred_pb = pb_inference(image_64)
        test_acc_pb += (pred_pb == label)
        if pred_pb != label:
            print("pb: failed")
        else:
            print("pb: success")

        # SpikingJelly推理
        spike_sum_sj, pred_sj = sj_inference(image)
        test_acc_sj += pred_sj == label
        if pred_sj != label:
            print("sj: failed")
        else:
            print("sj: success")

        # 将结果写入csv
        with open('paibox_spikingjelly_compare_main_380_result.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, label, spike_sum_pb, pred_pb, (pred_pb == label), spike_sum_sj, pred_sj, (pred_sj == label)])

    test_acc_pb = test_acc_pb / test_samples
    test_acc_sj = test_acc_sj / test_samples
    print(f'test_acc_pb ={test_acc_pb: .4f}')
    print(f'test_acc_sj ={test_acc_sj: .4f}')

if __name__ == "__main__":
    getNetParam()
    test(test_num=50)