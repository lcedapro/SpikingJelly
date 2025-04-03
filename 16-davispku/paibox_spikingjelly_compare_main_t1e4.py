# PAIBox 和 SpikingJelly 和 PAIBoard (可选,仿真速度较慢) 的推理结果比较，输出为 csv 文件
# 数据集DAVISPKU
import torch
import numpy as np
import paibox as pb
import csv
import os
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)
PAIBOARD_ENABLE = False

# 数据集
from CustomStaticDataset import CustomStaticDataset
from torch.utils.data import DataLoader
state_dict_path = './logs_t1e4_simple/T_16_b_64_c_2_SGD_lr_0.4_CosALR_48_amp_cupy_temporary_datasets'

# PAIBox
from simple_pb_infer import PAIBoxNet

# Spkikingjelly
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from infer_conv2int_t1e4 import PythonNet

# PAIBoard
if PAIBOARD_ENABLE:
    from paiboard import PAIBoard_SIM
    from paiboard import PAIBoard_PCIe
    from paiboard import PAIBoard_Ethernet
    from voting import voting
    baseDir = "./debug"

# 全局仿真时间步
SIM_TIMESTEP = 4 # <=16

# Dataloader
# 设置训练集和测试集的目录
test_dir = './temporary_datasets/duration_2000_0306'
test_dataset = CustomStaticDataset(root_dir=test_dir, expand_factor=4)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
print(len(test_data_loader))

# PAIBox网络定义
paiboxnet = PAIBoxNet(2, SIM_TIMESTEP,
     os.path.join(state_dict_path, 'checkpoint_max_conv2int.pth'),
     os.path.join(state_dict_path, 'vthr_list.npy'))

# PAIBox推理子程序
def pb_inference(image):
    # PAIBox推理
    spike_sum_pb, pred_pb = paiboxnet.pb_inference(image)
    return spike_sum_pb, pred_pb

if PAIBOARD_ENABLE:
    # PAIBoard网络定义
    timestep = 4
    layer_num = 4
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
    snn.config(oFrmNum=50*4)

    # PAIBoard推理子程序
    def board_inference(image):
        # PAIBoard 推理
        input_spike = image
        spike_out = snn(input_spike)
        spike_out = voting(spike_out, 10)
        spike_sum_board = spike_out.sum(axis=0)
        pred_board = np.argmax(spike_sum_board)
        print("Predicted number:", pred_board)
        return spike_sum_board, pred_board

# SpikingJelly网络定义和初始化
vthr_list = np.load(os.path.join(state_dict_path, 'vthr_list.npy')) # vthr from model_parameters_conv2int.py
vthr_list_tofloat = [float(vthr) for vthr in vthr_list]
net = PythonNet(channels=2, vthr_list=vthr_list_tofloat)
checkpoint = torch.load(os.path.join(state_dict_path, 'checkpoint_max_conv2int.pth'), map_location='cpu', weights_only=True)
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
        if PAIBOARD_ENABLE:
            writer.writerow(["index", "label", "pb_spike_sum", "pb_pred", "pb_correct", "sj_spike_sum", "sj_pred", "sj_correct", "board_spike_sum", "board_pred", "board_correct"])
        else:
            writer.writerow(["index", "label", "pb_spike_sum", "pb_pred", "pb_correct", "sj_spike_sum", "sj_pred", "sj_correct"])

    # 测试主程序
    test_acc_pb = 0
    test_acc_sj = 0
    if PAIBOARD_ENABLE:
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
        if PAIBOARD_ENABLE:
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
            if PAIBOARD_ENABLE:
                writer.writerow([i, label, spike_sum_pb, pred_pb, (pred_pb == label), spike_sum_sj, pred_sj, (pred_sj == label), spike_sum_board, pred_board, (pred_board == label)])
            else:
                writer.writerow([i, label, spike_sum_pb, pred_pb, (pred_pb == label), spike_sum_sj, pred_sj, (pred_sj == label)])

    test_acc_pb = test_acc_pb / test_samples
    test_acc_sj = test_acc_sj / test_samples
    if PAIBOARD_ENABLE:
        test_acc_board = test_acc_board / test_samples
    print(f'test_acc_pb ={test_acc_pb: .4f}')
    print(f'test_acc_sj ={test_acc_sj: .4f}')
    if PAIBOARD_ENABLE:
        print(f'test_acc_board ={test_acc_board: .4f}')

if __name__ == "__main__":
    test(test_num=100)