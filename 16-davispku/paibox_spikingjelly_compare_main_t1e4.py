# paibox 和 spikingjelly 的推理结果比较，输出为 csv 文件
# 数据集DAVISPKU
import torch
import numpy as np
import paibox as pb
import csv

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from CustomImageDataset0 import CustomImageDataset0
from torch.utils.data import DataLoader

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

from infer_conv2int_t1e4 import PythonNet
from voting import voting
vthr_list = np.load('./logs_t1e4_gconv/T_4_b_16_c_2_SGD_lr_0.2_CosALR_48_amp_cupy/vthr_list.npy') # vthr from model_parameters_conv2int.py

SIM_TIMESTEP = 4 # <=16
COMPILE_EN = False

# Dataloader
# 设置训练集和测试集的目录
test_dir = './duration_1000/test'
test_dataset = CustomImageDataset0(root_dir=test_dir, target_t=4, expand_factor=1, random_en=True, num_crops_per_video=1)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
print(len(test_data_loader))

# PAIBox网络定义
class Conv2d_Net(pb.Network):
    def __init__(self, channels, param_dict):
        super().__init__()

        def fakeout_with_t(t, **kwargs): # ignore other arguments except `t` & `bias`
            if t-1 < SIM_TIMESTEP:
                print(f't = {t}, input = image[{t-1}]')
                return image_69[t-1]
            else:
                print(f't = {t}, input = image[-1]')
                return image_69[-1]

        self.i0 = pb.InputProj(input=fakeout_with_t, shape_out=(1, 86, 65))
        self.n0 = pb.LIF((2, 42, 32), bias=param_dict['conv.0.bias'], threshold=param_dict['conv.0.vthr'], reset_v=0, tick_wait_start=1) # convpool7x7p2s2
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=param_dict['conv.0.weight'], padding=2, stride=2)

        self.n1_0 = pb.LIF((2, 21, 16), bias=param_dict['conv.2.bias'][:2], threshold=param_dict['conv.2.vthr'], reset_v=0, tick_wait_start=2) # convpool7x7p3s2
        self.conv2d_1_0 = pb.Conv2d(self.n0, self.n1_0, kernel=param_dict['conv.2.weight'][:2], padding=3, stride=2)

        self.n1_1 = pb.LIF((2, 21, 16), bias=param_dict['conv.2.bias'][2:], threshold=param_dict['conv.2.vthr'], reset_v=0, tick_wait_start=2) # convpool7x7p3s2
        self.conv2d_1_1 = pb.Conv2d(self.n0, self.n1_1, kernel=param_dict['conv.2.weight'][2:], padding=3, stride=2)

        self.n2_0 = pb.LIF((2, 21, 16), bias=param_dict['conv.4.bias'][:2], threshold=param_dict['conv.4.vthr'], reset_v=0, tick_wait_start=3) # conv5x5
        self.conv2d_2_0 = pb.Conv2d(self.n1_0, self.n2_0, kernel=param_dict['conv.4.weight'][:2], padding=2, stride=1)

        self.n2_1 = pb.LIF((2, 21, 16), bias=param_dict['conv.4.bias'][2:], threshold=param_dict['conv.4.vthr'], reset_v=0, tick_wait_start=3) # conv5x5
        self.conv2d_2_1 = pb.Conv2d(self.n1_1, self.n2_1, kernel=param_dict['conv.4.weight'][2:], padding=2, stride=1)

        self.n10 = pb.LIF(512, threshold=param_dict['fc.2.vthr'], reset_v=0, tick_wait_start=4) # fc
        self.fc_0_0 = pb.FullConn(self.n2_0, self.n10, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.2.weight'][:2*21*16])
        self.fc_0_1 = pb.FullConn(self.n2_1, self.n10, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.2.weight'][2*21*16:])

        self.n11 = pb.LIF(128, threshold=param_dict['fc.5.vthr'], reset_v=0, tick_wait_start=5) # fc
        self.fc_1 = pb.FullConn(self.n10, self.n11, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.5.weight'])

        self.n12 = pb.LIF(90, threshold=param_dict['fc.8.vthr'], reset_v=0, tick_wait_start=6) # fc
        self.fc_2 = pb.FullConn(self.n11, self.n12, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.8.weight'])

        self.probe1 = pb.Probe(self.n12, "spike")


# PAIBox网络初始化
param_dict = {}
def getNetParam():
    timestep = SIM_TIMESTEP
    layer_num = 6
    delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay

    checkpoint = torch.load('./logs_t1e4_gconv/T_4_b_16_c_2_SGD_lr_0.2_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
    param_dict['conv.0.weight']=checkpoint['net']['conv.0.weight'].numpy().astype(np.int8)
    param_dict['conv.0.bias']=checkpoint['net']['conv.0.bias'].numpy().astype(np.int8)
    param_dict['conv.2.weight']=checkpoint['net']['conv.2.weight'].numpy().astype(np.int8)
    param_dict['conv.2.bias']=checkpoint['net']['conv.2.bias'].numpy().astype(np.int8)
    param_dict['conv.4.weight']=checkpoint['net']['conv.4.weight'].numpy().astype(np.int8)
    param_dict['conv.4.bias']=checkpoint['net']['conv.4.bias'].numpy().astype(np.int8)
    param_dict['fc.2.weight']=checkpoint['net']['fc.2.weight'].numpy().astype(np.int8).T
    param_dict['fc.5.weight']=checkpoint['net']['fc.5.weight'].numpy().astype(np.int8).T
    param_dict['fc.8.weight']=checkpoint['net']['fc.8.weight'].numpy().astype(np.int8).T
    param_dict['conv.0.vthr']=int(vthr_list[0])
    param_dict['conv.2.vthr']=int(vthr_list[1])
    param_dict['conv.4.vthr']=int(vthr_list[2])
    param_dict['fc.2.vthr']=int(vthr_list[3])
    param_dict['fc.5.vthr']=int(vthr_list[4])
    param_dict['fc.8.vthr']=int(vthr_list[5])
getNetParam()

# PAIBox仿真器
pb_net = Conv2d_Net(2, param_dict)
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
net = PythonNet(channels=2, vthr_list=vthr_list_tofloat)
checkpoint = torch.load('./logs_t1e4_gconv/T_4_b_16_c_2_SGD_lr_0.2_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
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
    with open('paibox_spikingjelly_compare_main_t1e4_result.csv', 'w', newline='') as file:
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
        image = image.squeeze(0)  # 去掉批次维度
        image = image.numpy()  # 转换为 numpy 数组
        image = image.astype(np.uint8)  # 转换为 uint8
        label = label.item()
        # 修改数据，将大于等于1的像素值设为1，否则为0
        image = np.where(image == 0, 0, 1)

        # 对于新的PAIBox网络，需要预先将image的形状从[2, 346, 260]用maxpool2d池化为[2, 86, 65]
        global image_69
        image_69 = image_tensor[0]
        maxpool2d = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        image_69 = maxpool2d(image_69)
        image_69 = image_69.squeeze()  # 去掉批次维度
        image_69 = image_69.numpy()  # 转换为 numpy 数组
        image_69 = image_69.astype(np.uint8)  # 转换为 uint8

        test_samples += 1

        # PAIBox推理
        spike_sum_pb, pred_pb = pb_inference(image_69)
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
        with open('paibox_spikingjelly_compare_main_t1e4_result.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, label, spike_sum_pb, pred_pb, (pred_pb == label), spike_sum_sj, pred_sj, (pred_sj == label)])

    test_acc_pb = test_acc_pb / test_samples
    test_acc_sj = test_acc_sj / test_samples
    print(f'test_acc_pb ={test_acc_pb: .4f}')
    print(f'test_acc_sj ={test_acc_sj: .4f}')

if __name__ == "__main__":
    getNetParam()
    test(test_num=1000000)

    if COMPILE_EN:
        mapper = pb.Mapper()

        mapper.build(pb_net)

        graph_info = mapper.compile(
            weight_bit_optimization=True, grouping_optim_target="both"
        )

        # #N of cores required
        print("Core required:", graph_info["n_core_required"])
        print("Core occupied:", graph_info["n_core_occupied"])

        mapper.export(
            write_to_file=True, fp="./debug", format="npy", export_core_params=False
        )

        # Clear all the results
        mapper.clear()