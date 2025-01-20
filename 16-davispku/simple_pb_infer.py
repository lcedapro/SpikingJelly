# paibox 和 spikingjelly 的推理结果比较，输出为 csv 文件
# 数据集DAVISPKU
import torch
import numpy as np
import paibox as pb

from CustomImageDataset import CustomImageDataset
from torch.utils.data import DataLoader

from voting import voting
vthr_list = np.load('./logs_897/T_16_b_4_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy') # vthr from model_parameters_conv2int.py

SIM_TIMESTEP = 8 # <=16
COMPILE_EN = True

# Dataloader
# 设置训练集和测试集的目录
test_dir = './duration_1000/test'
test_dataset = CustomImageDataset(root_dir=test_dir, target_t=8, expand_factor=1, random_en=False)
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

        self.i0 = pb.InputProj(input=fakeout_with_t, shape_out=(2, 69, 52))
        self.n0 = pb.LIF((channels*2, 35, 26), bias=param_dict['conv.0.bias'], threshold=param_dict['conv.0.vthr'], reset_v=0, tick_wait_start=1) # convpool3x3
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=param_dict['conv.0.weight'], padding=2, stride=2)

        self.n1 = pb.LIF((channels*2, 35, 26), bias=param_dict['conv.2.bias'], threshold=param_dict['conv.2.vthr'], reset_v=0, tick_wait_start=2) # conv3x3
        self.conv2d_1 = pb.Conv2d(self.n0, self.n1, kernel=param_dict['conv.2.weight'], padding=2, stride=1)

        self.n2 = pb.LIF((channels*4, 18, 13), bias=param_dict['conv.4.bias'], threshold=param_dict['conv.4.vthr'], reset_v=0, tick_wait_start=3) # convpool3x3
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=param_dict['conv.4.weight'], padding=2, stride=2)

        self.n3 = pb.LIF((channels*4, 18, 13), bias=param_dict['conv.6.bias'], threshold=param_dict['conv.6.vthr'], reset_v=0, tick_wait_start=4) # conv3x3
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=param_dict['conv.6.weight'], padding=2, stride=1)

        self.n10 = pb.LIF(channels*8 * 8 * 8, threshold=param_dict['fc.2.vthr'], reset_v=0, tick_wait_start=5) # fc
        self.fc_0 = pb.FullConn(self.n3, self.n10, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.2.weight'])

        self.n11 = pb.LIF(channels*8 * 4 * 4, threshold=param_dict['fc.5.vthr'], reset_v=0, tick_wait_start=6) # fc
        self.fc_1 = pb.FullConn(self.n10, self.n11, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.5.weight'])

        self.n12 = pb.LIF(80, threshold=param_dict['fc.8.vthr'], reset_v=0, tick_wait_start=7) # fc
        self.fc_2 = pb.FullConn(self.n11, self.n12, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.8.weight'])

        self.probe1 = pb.Probe(self.n12, "spike")


# PAIBox网络初始化
param_dict = {}
def getNetParam():
    timestep = SIM_TIMESTEP
    layer_num = 7
    delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay

    checkpoint = torch.load('./logs_897/T_16_b_4_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
    param_dict['conv.0.weight']=checkpoint['net']['conv.0.weight'].numpy().astype(np.int8)
    param_dict['conv.0.bias']=checkpoint['net']['conv.0.bias'].numpy().astype(np.int8)
    param_dict['conv.2.weight']=checkpoint['net']['conv.2.weight'].numpy().astype(np.int8)
    param_dict['conv.2.bias']=checkpoint['net']['conv.2.bias'].numpy().astype(np.int8)
    param_dict['conv.4.weight']=checkpoint['net']['conv.4.weight'].numpy().astype(np.int8)
    param_dict['conv.4.bias']=checkpoint['net']['conv.4.bias'].numpy().astype(np.int8)
    param_dict['conv.6.weight']=checkpoint['net']['conv.6.weight'].numpy().astype(np.int8)
    param_dict['conv.6.bias']=checkpoint['net']['conv.6.bias'].numpy().astype(np.int8)
    param_dict['fc.2.weight']=checkpoint['net']['fc.2.weight'].numpy().astype(np.int8).T
    param_dict['fc.5.weight']=checkpoint['net']['fc.5.weight'].numpy().astype(np.int8).T
    param_dict['fc.8.weight']=checkpoint['net']['fc.8.weight'].numpy().astype(np.int8).T
    param_dict['conv.0.vthr']=int(vthr_list[0])
    param_dict['conv.2.vthr']=int(vthr_list[1])
    param_dict['conv.4.vthr']=int(vthr_list[2])
    param_dict['conv.6.vthr']=int(vthr_list[3])
    param_dict['fc.2.vthr']=int(vthr_list[4])
    param_dict['fc.5.vthr']=int(vthr_list[5])
    param_dict['fc.8.vthr']=int(vthr_list[6])
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
    spike_out = voting(spike_out, 10) # 投票层
    spike_sum_pb = spike_out.sum(axis=0)
    pred_pb = np.argmax(spike_sum_pb)
    print("Predicted number:", pred_pb)

    sim.reset()
    return spike_sum_pb, pred_pb

# 测试程序
def test():
    for i, (image_tensor, label_tensor) in enumerate(test_data_loader):
        # 仿真时间 [N, T, C, H, W] -> [N, T=SIM_TIMESTEP, C, H, W]
        image_tensor = image_tensor[:, :SIM_TIMESTEP, :, :, :]

        # 获取图片和标签
        image, label = image_tensor[0], label_tensor[0]

        # 对于新的PAIBox网络，需要预先将image的形状从[2, 346, 260]用maxpool2d池化为[2, 69, 52]
        global image_69
        image_69 = image_tensor[0]
        maxpool2d = torch.nn.MaxPool2d(kernel_size=5, stride=5)
        image_69 = maxpool2d(image_69)

        # 图片转为 numpy 数组，标签转为 int
        image_69 = image_69.squeeze()  # 去掉批次维度
        image_69 = image_69.numpy()  # 转换为 numpy 数组
        image_69 = image_69.astype(np.uint8)  # 转换为 uint8
        label = label.item()

        # PAIBox推理
        spike_sum_pb, pred_pb = pb_inference(image_69)
        print("spike_sum_pb:", spike_sum_pb)
        if pred_pb != label:
            print(f"predicted = {pred_pb}, label= {label}, failed")
        else:
            print(f"predicted = {pred_pb}, label= {label}, succcess")

        break

if __name__ == "__main__":
    getNetParam()
    test()

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
