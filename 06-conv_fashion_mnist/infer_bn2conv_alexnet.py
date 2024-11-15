import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing

# python .\infer_bn2conv_alexnet.py -data-dir ./ -out-dir ./logs -opt sgd -resume './logs/T4_b1024_sgd_lr0.1_c16_amp_cupy_alex_dec1/checkpoint_max_bn2conv.pth' -b 16 -device cpu
# epoch = 0, test_loss = 0.0297, test_acc = 0.8147
# test speed = 858.0115 images/s
# escape time = 2024-11-07 11:12:18

class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential( # input = 1 * 28 * 28 = 784 AlexNet Pool1 (channels=128)
        layer.Conv2d(1, channels*2, kernel_size=5, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 26 * 26 * 2C AlexNet CL2
    
        layer.Conv2d(channels*2, channels*2, kernel_size=4, padding=1, stride=2, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 13 * 13 * 2C AlexNet Pool2

        layer.Conv2d(channels*2, channels*3, kernel_size=3, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*3),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 13 * 13 * 3C AlexNet CL3

        # layer.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*3),
        # neuron.IFNode(surrogate_function=surrogate.ATan()), # 13 * 13 * 3C AlexNet CL4

        layer.Conv2d(channels*3, channels*2, kernel_size=3, padding=1, stride=1, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 13 * 13 * 2C AlexNet Pool2

        layer.Conv2d(channels*2, channels*2, kernel_size=3, padding=0, stride=2, bias=True),
        # layer.BatchNorm2d(channels*2),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 6 * 6 * 2C AlexNet CL5

        layer.Flatten(),
        layer.Linear(channels*2*6*6, channels*2*16, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 16C AlexNet FCL6

        layer.Linear(channels*2*16, channels*2*4, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # 16C AlexNet FCL7

        layer.Linear(channels*2*4, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()), # output = 10
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



def main():
    '''
    (sj-dev) wfang@Precision-5820-Tower-X-Series:~/spikingjelly_dev$ python -m spikingjelly.activation_based.examples.conv_fashion_mnist -h

    usage: conv_fashion_mnist.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                                 [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-momentum MOMENTUM] [-lr LR]

    Classify Fashion-MNIST

    optional arguments:
      -h, --help          show this help message and exit
      -T T                simulating time-steps
      -device DEVICE      device
      -b B                batch size
      -epochs N           number of total epochs to run
      -j N                number of data loading workers (default: 4)
      -data-dir DATA_DIR  root dir of Fashion-MNIST dataset
      -out-dir OUT_DIR    root dir for saving logs and checkpoint
      -resume RESUME      resume from the checkpoint path
      -amp                automatic mixed precision training
      -cupy               use cupy neuron and multi-step forward mode
      -opt OPT            use which optimizer. SDG or Adam
      -momentum MOMENTUM  momentum for SGD
      -save-es            dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}
    '''
    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8

    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 4 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -resume ./logs/T4_b256_sgd_lr0.1_c128_amp_cupy/checkpoint_latest.pth -save-es ./logs
    parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=16, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', default=None, help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')

    args = parser.parse_args()
    print(args)

    net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy)

    print(net)

    net.to(args.device)

    test_set = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        net.load_state_dict(checkpoint['net'])

    for epoch in range(0,1):
        start_time = time.time()
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples

        print(args)
        print(f'epoch = {epoch}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
        print(f'test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()