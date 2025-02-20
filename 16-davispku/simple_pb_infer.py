# paibox 和 spikingjelly 的推理结果比较，输出为 csv 文件
# 数据集DAVISPKU
import torch
import numpy as np
import paibox as pb
pb.BACKEND_CONFIG.test_chip_addr = (2, 0)
pb.BACKEND_CONFIG.target_chip_addr = [(1, 0), (0, 0), (1, 1), (0, 1)]
from CustomImageDataset0 import CustomImageDataset0
from torch.utils.data import DataLoader

from voting import voting

SIM_TIMESTEP = 4 # <=16
COMPILE_EN = True

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
        self.sim_timestep = param_dict["timestep"]
        self.image_69 = np.zeros((self.sim_timestep, 1, 86, 65), dtype=np.int8)

        self.i0 = pb.InputProj(input=self.fakeout_with_t, shape_out=(1, 86, 65))
        self.n0 = pb.LIF((1, 42, 32), bias=param_dict['conv.0.bias'], threshold=param_dict['conv.0.vthr'], reset_v=0, tick_wait_start=1) # convpool7x7p2s2
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=param_dict['conv.0.weight'], padding=2, stride=2)

        self.n1 = pb.LIF((2, 21, 16), bias=param_dict['conv.2.bias'], threshold=param_dict['conv.2.vthr'], reset_v=0, tick_wait_start=2) # convpool7x7p3s2
        self.conv2d_1_0 = pb.Conv2d(self.n0, self.n1, kernel=param_dict['conv.2.weight'], padding=3, stride=2)

        self.n10 = pb.LIF(512, threshold=param_dict['fc.2.vthr'], reset_v=0, tick_wait_start=3) # fc
        self.fc_0 = pb.FullConn(self.n1, self.n10, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.2.weight'])

        self.n11 = pb.LIF(128, threshold=param_dict['fc.5.vthr'], reset_v=0, tick_wait_start=4) # fc
        self.fc_1 = pb.FullConn(self.n10, self.n11, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.5.weight'])

        self.n12 = pb.LIF(90, threshold=param_dict['fc.8.vthr'], reset_v=0, tick_wait_start=5) # fc
        self.fc_2 = pb.FullConn(self.n11, self.n12, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.8.weight'])

        self.probe1 = pb.Probe(self.n12, "spike")
    
    def fakeout_with_t(self, t, **kwargs): # ignore other arguments except `t` & `bias`
        # 如果t-1小于self.sim_timestep，则打印t和image[t-1]，并返回image[t-1]
        if t-1 < self.sim_timestep:
            print(f't = {t}, input = image[{t-1}]')
            return self.image_69[t-1]
        # 否则，打印t和image[-1]，并返回image[-1]
        else:
            print(f't = {t}, input = image[-1]')
            return self.image_69[-1]

class PAIBoxNet:
    def __init__(self, channels, timestep, param_dict_path, vthr_list_path):
        # PAIBox网络初始化
        self.param_dict = self._getNetParam(timestep, param_dict_path, vthr_list_path)
        self.pb_net = Conv2d_Net(channels, self.param_dict)
        # PAIBox仿真器
        self.sim = pb.Simulator(self.pb_net)

    def _getNetParam(self, timestep, param_dict_path, vthr_list_path):
        param_dict = {}
        timestep = timestep
        layer_num = 5
        delay = layer_num - 1
        param_dict["timestep"] = timestep
        param_dict["layer_num"] = layer_num
        param_dict["delay"] = delay

        checkpoint = torch.load(param_dict_path, map_location='cpu', weights_only=True)
        vthr_list = np.load(vthr_list_path) # vthr from model_parameters_conv2int.py
        param_dict['conv.0.weight']=checkpoint['net']['conv.0.weight'].numpy().astype(np.int8)
        param_dict['conv.0.bias']=checkpoint['net']['conv.0.bias'].numpy().astype(np.int8)
        param_dict['conv.2.weight']=checkpoint['net']['conv.2.weight'].numpy().astype(np.int8)
        param_dict['conv.2.bias']=checkpoint['net']['conv.2.bias'].numpy().astype(np.int8)
        param_dict['fc.2.weight']=checkpoint['net']['fc.2.weight'].numpy().astype(np.int8).T
        param_dict['fc.5.weight']=checkpoint['net']['fc.5.weight'].numpy().astype(np.int8).T
        param_dict['fc.8.weight']=checkpoint['net']['fc.8.weight'].numpy().astype(np.int8).T
        param_dict['conv.0.vthr']=int(vthr_list[0])
        param_dict['conv.2.vthr']=int(vthr_list[1])
        param_dict['fc.2.vthr']=int(vthr_list[2])
        param_dict['fc.5.vthr']=int(vthr_list[3])
        param_dict['fc.8.vthr']=int(vthr_list[4])

        return param_dict

    # PAIBox推理子程序
    def pb_inference(self, image):
        """
        PAIBox推理
        输入: np.ndarray or torch.Tensor , shape=[2, 346, 260]
        输出: spike_sum_pb, pred_pb
        """
        print(image.shape)
        # 数据预处理：将image的形状从[2, 346, 260]用maxpool2d池化为[2, 86, 65]
        maxpool2d = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        if type(image) == np.ndarray:
            image = image.astype(np.int8)
            image = torch.from_numpy(image) # numpy -> tensor
        elif type(image) == torch.Tensor:
            pass
        else:
            raise TypeError("image must be np.ndarray or torch.Tensor")
        image = maxpool2d(image)
        image = image.numpy().astype(np.uint8) # tensor -> numpy
        image = np.where(image != 0, 1, 0)

        # 图像加载
        self.pb_net.image_69 = image
        # PAIBox推理

        # Simulation, duration=timestep + delay
        self.sim.run(self.param_dict["timestep"] + self.param_dict["delay"], reset=False)

        # Decode the output
        spike_out = self.sim.data[self.pb_net.probe1].astype(np.int32)
        spike_out = spike_out[self.param_dict["delay"] :]
        spike_out = voting(spike_out, 10) # 投票层
        spike_sum_pb = spike_out.sum(axis=0)
        pred_pb = np.argmax(spike_sum_pb)
        print("Predicted number:", pred_pb)

        self.sim.reset()
        return spike_sum_pb, pred_pb

# 测试程序
def test():
    paiboxnet = PAIBoxNet(2, SIM_TIMESTEP,
         './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth',
         './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy')
    for i, (image_tensor, label_tensor) in enumerate(test_data_loader):
        # 仿真时间 [N, T, C, H, W] -> [N, T=SIM_TIMESTEP, C, H, W]
        image_tensor = image_tensor[:, :SIM_TIMESTEP, :, :, :]

        # 获取图片和标签
        image, label = image_tensor[0], label_tensor[0]

        # PAIBox推理
        spike_sum_pb, pred_pb = paiboxnet.pb_inference(image)
        print("spike_sum_pb:", spike_sum_pb)
        if pred_pb != label:
            print(f"predicted = {pred_pb}, label= {label}, failed")
        else:
            print(f"predicted = {pred_pb}, label= {label}, succcess")

        break

if __name__ == "__main__":
    # test()

    if COMPILE_EN:

        paiboxnet = PAIBoxNet(2, SIM_TIMESTEP,
            './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth',
            './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy')
        mapper = pb.Mapper()

        mapper.build(paiboxnet.pb_net)

        graph_info = mapper.compile(
            weight_bit_optimization=True, grouping_optim_target="both"
        )

        # #N of cores required
        print("Core required:", graph_info["n_core_required"])
        print("Core occupied:", graph_info["n_core_occupied"])

        mapper.export(
            write_to_file=True, fp="./debug", format="npy", export_core_params=True
        )

        # Clear all the results
        mapper.clear()