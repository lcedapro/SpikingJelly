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
vthr_list = np.load('./logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy') # vthr from model_parameters_conv2int.py

SIM_TIMESTEP = 4 # <=16
COMPILE_EN = False


# Dataloader
# 设置训练集和测试集的目录
test_dir = './duration_1000/test'
test_dataset = CustomImageDataset0(root_dir=test_dir, target_t=4, expand_factor=1, random_en=True, num_crops_per_video=1)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
print(len(test_data_loader))

# PAIBox网络定义
from simple_pb_infer import PAIBoxNet
paiboxnet = PAIBoxNet(2, SIM_TIMESTEP,
     './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth',
     './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy')
# PAIBox推理子程序
def pb_inference(image):
    # PAIBox推理
    spike_sum_pb, pred_pb = paiboxnet.pb_inference(image)
    return spike_sum_pb, pred_pb

# PAIBoard网络定义
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet
timestep = 4
layer_num = 4
baseDir = "./debug"
snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
# snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
# snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
snn.config(oFrmNum=90*4)

# PAIBoard推理子程序
def board_inference(image):
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

    # PAIBoard 推理
    input_spike = np.expand_dims(image[0], axis=0).repeat(timestep, axis=0)
    spike_out = snn(input_spike)
    spike_out = voting(spike_out, 10)
    spike_sum_board = spike_out.sum(axis=0)
    pred_board = np.argmax(spike_sum_board)
    print("Predicted number:", pred_board)

    return spike_sum_board, pred_board

# SpikingJelly网络定义和初始化
vthr_list_tofloat = [float(vthr) for vthr in vthr_list]
net = PythonNet(channels=2, vthr_list=vthr_list_tofloat)
checkpoint = torch.load('./logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
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
    with open('paibox_spikingjelly_compare_main_t1e4_result1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "pb_spike_sum", "pb_pred", "pb_correct", "sj_spike_sum", "sj_pred", "sj_correct", "board_spike_sum", "board_pred", "board_correct"])

    # 测试主程序
    test_acc_pb = 0
    test_acc_sj = 0
    test_acc_board = 0
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

        test_samples += 1

        # PAIBox推理
        spike_sum_pb, pred_pb = pb_inference(image)
        # 在推理时保存图片到/image文件夹下，文件名为{i}.npy
        # np.save(f"仿真输入输出示例/image/label_{label}_iter_{i}_image.npy", image_69[0])
        # np.save(f"仿真输入输出示例/spike_out/label_{label}_iter_{i}_spike_out.npy", original_spike_out)
        test_acc_pb += (pred_pb == label)
        if pred_pb != label:
            print("pb: failed")
        else:
            print("pb: success")

        # PAIBoard推理
        spike_sum_board, pred_board = board_inference(image)
        test_acc_board += (pred_board == label)
        if pred_board != label:
            print("board: failed")
        else:
            print("board: success")

        # SpikingJelly推理
        spike_sum_sj, pred_sj = sj_inference(image)
        test_acc_sj += pred_sj == label
        if pred_sj != label:
            print("sj: failed")
        else:
            print("sj: success")

        # 将结果写入csv
        with open('paibox_spikingjelly_compare_main_t1e4_result1.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, label, spike_sum_pb, pred_pb, (pred_pb == label), spike_sum_sj, pred_sj, (pred_sj == label), spike_sum_board, pred_board, (pred_board == label)])

    test_acc_pb = test_acc_pb / test_samples
    test_acc_sj = test_acc_sj / test_samples
    test_acc_board = test_acc_board / test_samples
    print(f'test_acc_pb ={test_acc_pb: .4f}')
    print(f'test_acc_sj ={test_acc_sj: .4f}')
    print(f'test_acc_board ={test_acc_board: .4f}')

if __name__ == "__main__":
    test(test_num=1000000)

    if COMPILE_EN:
        mapper = pb.Mapper()

        mapper.build(paiboxnet.pb_net)

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