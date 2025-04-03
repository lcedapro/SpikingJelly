# paibox 和 spikingjelly 的推理结果比较，输出为 csv 文件
# 数据集DAVISPKU
import torch
import numpy as np
import paibox as pb
import os
pb.BACKEND_CONFIG.test_chip_addr = (2, 0)
pb.BACKEND_CONFIG.target_chip_addr = [(1, 0), (0, 0), (1, 1), (0, 1)]
state_dict_path = './logs_t1e4_simple/T_16_b_64_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_temporary_datasets'

SIM_TIMESTEP = 4 # <=16

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

        self.n12 = pb.LIF(50, threshold=param_dict['fc.8.vthr'], reset_v=0, tick_wait_start=5) # fc
        self.fc_2 = pb.FullConn(self.n11, self.n12, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.8.weight'])

        self.probe1 = pb.Probe(self.n12, "spike")
    
    def fakeout_with_t(self, t, **kwargs): # ignore other arguments except `t` & `bias`
        # 如果t-1小于self.sim_timestep，则打印t和image[t-1]，并返回image[t-1]
        if t-1 < self.sim_timestep:
            # print(f't = {t}, input = image[{t-1}]')
            return self.image_69[t-1]
        # 否则，打印t和image[-1]，并返回image[-1]
        else:
            # print(f't = {t}, input = image[-1]')
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

if __name__ == "__main__":
    paiboxnet = PAIBoxNet(2, SIM_TIMESTEP,
        './logs_t1e4_simple/T_16_b_64_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_temporary_datasets/checkpoint_max_conv2int.pth',
        './logs_t1e4_simple/T_16_b_64_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_temporary_datasets/vthr_list.npy')
    mapper = pb.Mapper()

    mapper.build(paiboxnet.pb_net)

    graph_info = mapper.compile(
        weight_bit_optimization=True, grouping_optim_target="both"
    )

    # #N of cores required
    print("Core required:", graph_info["n_core_required"])
    print("Core occupied:", graph_info["n_core_occupied"])

    mapper.export(
        write_to_file=True, fp="./debug2", format="npy", export_core_params=True
    )

    # Clear all the results
    mapper.clear()

