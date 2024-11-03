import argparse
from pathlib import Path

import numpy as np

import paibox as pb
import torch, torchvision
import csv

# python .\paibox_main.py
# [[0 0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0 0 0]]
# [0 0 0 0 4 0 0 0 0 0]
# Predicted number: 4
# test_acc = 0.5800

class Conv2d_Net(pb.Network):
    def __init__(self, weight0, bias0, Vthr0, weight1, bias1, Vthr1, weight2, bias2, Vthr2, weight3, bias3, Vthr3, weight4, Vthr4, weight5, Vthr5):
        super().__init__()

        self.i0 = pb.InputProj(input=None, shape_out=(1, 28, 28))
        self.n0 = pb.LIF((32, 26, 26), bias=bias0, threshold=Vthr0, reset_v=0, tick_wait_start=1)  # 26 * 26
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=weight0, padding=0, stride=1)

        self.n1 = pb.LIF((64, 12, 12), bias=bias1, threshold=Vthr1, reset_v=0, tick_wait_start=2)  # 12 * 12
        self.conv2d_1 = pb.Conv2d(self.n0, self.n1, kernel=weight1, padding=0, stride=2)

        self.n2 = pb.LIF((64, 10, 10), bias=bias2, threshold=Vthr2, reset_v=0, tick_wait_start=3)  # 10 * 10
        self.conv2d_2 = pb.Conv2d(self.n1, self.n2, kernel=weight2, padding=0, stride=1)

        self.n3 = pb.LIF((128, 4, 4), bias=bias3, threshold=Vthr3, reset_v=0, tick_wait_start=4)  # 4 * 4
        self.conv2d_3 = pb.Conv2d(self.n2, self.n3, kernel=weight3, padding=0, stride=2)

        self.n4 = pb.LIF(128*8, threshold=Vthr3, reset_v=0, tick_wait_start=5)
        self.fc4 = pb.FullConn(self.n3, self.n4, weights=weight4, conn_type=pb.SynConnType.All2All)

        self.n5 = pb.LIF(10, threshold=Vthr3, reset_v=0, tick_wait_start=6)
        self.fc5 = pb.FullConn(self.n4, self.n5, weights=weight5, conn_type=pb.SynConnType.All2All)
    
        self.probe1 = pb.Probe(self.n5, "spike")


param_dict = {}


def getNetParam():
    timestep = 4
    layer_num = 6
    delay = layer_num - 1
    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay

    checkpoint = torch.load('./logs/T4_b768_sgd_lr0.1_c32_amp_cupy/checkpoint_max_conv2int.pth', map_location='cpu', weights_only=True)
    param_dict["w0"]=checkpoint['net']['conv_fc.0.weight'].numpy()
    param_dict["b0"]=checkpoint['net']['conv_fc.0.bias'].numpy()
    param_dict["vthr0"]=8925 # 35*255 SpikingJelly使用float输入，范围0-1；PAIbox使用int输入，范围0-255
    param_dict["w1"]=checkpoint['net']['conv_fc.2.weight'].numpy()
    param_dict["b1"]=checkpoint['net']['conv_fc.2.bias'].numpy()
    param_dict["vthr1"]=73
    param_dict["w2"]=checkpoint['net']['conv_fc.4.weight'].numpy()
    param_dict["b2"]=checkpoint['net']['conv_fc.4.bias'].numpy()
    param_dict["vthr2"]=73
    param_dict["w3"]=checkpoint['net']['conv_fc.6.weight'].numpy()
    param_dict["b3"]=checkpoint['net']['conv_fc.6.bias'].numpy()
    param_dict["vthr3"]=65
    param_dict["w4"]=checkpoint['net']['conv_fc.9.weight'].numpy().T
    param_dict["vthr4"]=2944
    param_dict["w5"]=checkpoint['net']['conv_fc.11.weight'].numpy().T
    param_dict["vthr5"]=996

def test(test_num: int = 100):
    # Dataloader
    data_dir = './'
    b = 1
    j = 1
    test_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=b,
        shuffle=True,
        drop_last=False,
        num_workers=j,
        pin_memory=True
    )

    # 结果csv初始化
    with open('paibox_main_result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "spike_sum", "label", "pred", "correct"])

    # 测试主程序
    test_acc = 0
    test_samples = 0
    for i, (image, label) in enumerate(test_data_loader):
        if i == test_num:
            break
        # 获取图片和标签
        image, label = image[0], label[0]

        # 转为 numpy 数组
        image = image.squeeze()  # 去掉批次维度
        image = image.numpy()  # 转换为 numpy 数组
        image = (image * 255).astype(np.uint8)  # 转换为 uint8
        label = label.item()

        # Input
        pb_net.i0.input = image

        # Simulation, duration=timestep + delay
        sim = pb.Simulator(pb_net)
        sim.run(param_dict["timestep"] + param_dict["delay"], reset=False)

        # Decode the output
        spike_out = sim.data[pb_net.probe1].astype(np.int8)
        spike_out = spike_out[param_dict["delay"] :]
        spike_sum = spike_out.sum(axis=0)
        pred = np.argmax(spike_sum)

        test_samples += 1
        test_acc += (pred == label)

        # 将结果写入csv
        with open('paibox_main_result.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, spike_sum, label, pred, (pred == label)])

    test_acc = test_acc / test_samples
    print(f'test_acc ={test_acc: .4f}')

if __name__ == "__main__":
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
        param_dict["vthr5"],
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

    # Decode the output
    spike_out = sim.data[pb_net.probe1].astype(np.int8)
    spike_out = spike_out[param_dict["delay"] :]
    print(spike_out)
    spike_sum = spike_out.sum(axis=0)
    print(spike_sum)
    pred = np.argmax(spike_sum)
    print("Predicted number:", pred)

    assert pred == 4, print("failed")  # Correct result is 4

    test(test_num=100)
    # out_dir = Path(__file__).parent
    # pb.BACKEND_CONFIG.target_chip_addr = (0, 0)

    # mapper = pb.Mapper()
    # mapper.build(pb_net)
    # graph_info = mapper.compile(
    #     weight_bit_optimization=True, grouping_optim_target="both"
    # )

    # # #N of cores required
    # print("Core required:", graph_info["n_core_required"])

    # mapper.export(
    #     write_to_file=True, fp=out_dir / "debug", format="npy", export_core_params=False
    # )

    # # Clear all the results
    # mapper.clear()
