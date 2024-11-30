# py .\infer_bn2conv.py -data_dir ..\15-classify_dvsg\DVS128Gesture\ -out_dir ./logs -channels 32 -resume './logs/T_16_b_48_c_32_Adam_lr_0.01_CosALR_16_amp_cupy_3/checkpoint_max_bn2conv.pth'
# epoch=0, test_loss=0.028904228904543236, test_acc=0.7916666666666666, max_test_acc=0, total_time=28.829838514328003

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse

import numpy as np
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)
    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)

class PythonNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(PythonNet.conv3x3(2, channels))
        conv.extend(PythonNet.convpool2x2(channels, channels))
        for i in range(4):
            conv.extend(PythonNet.conv3x3(channels, channels))
            conv.extend(PythonNet.convpool2x2(channels, channels))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(0.5),
            nn.Linear(channels * 2 * 2, 110, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.vote(self.fc(self.conv(x[0])))
        for t in range(1, x.shape[0]):
            out_spikes += self.vote(self.fc(self.conv(x[t])))
        return out_spikes / x.shape[0]

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        ]
    
    @staticmethod
    def convpool2x2(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0, stride=2, bias=True),
            # nn.BatchNorm2d(out_channels),
            neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        ]

try:
    import cupy

    class CextNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            conv = []
            conv.extend(CextNet.conv3x3(2, channels))
            conv.extend(CextNet.convpool2x2(channels, channels))
            for i in range(4):
                conv.extend(CextNet.conv3x3(channels, channels))
                conv.extend(CextNet.convpool2x2(channels, channels))
            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(2),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False)),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(channels * 2 * 2, 110, bias=False)),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            )
            self.vote = VotingLayer(10)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
            out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
            return self.vote(out_spikes.mean(0))

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
                    # nn.BatchNorm2d(out_channels),
                ),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            ]
        
        @staticmethod
        def convpool2x2(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0, stride=2, bias=True),
                    # nn.BatchNorm2d(out_channels),
                ),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            ]

except ImportError:
    print('Cupy is not installed.')

def main():
    # python classify_dvsg.py -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024
    '''
    * :ref:`API in English <classify_dvsg.__init__-en>`

    .. _classify_dvsg.__init__-cn:

    用于分类DVS128 Gesture数据集的代码样例。网络结构来自于 `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_。

    .. code:: bash

        usage: classify_dvsg.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-channels CHANNELS] [-data_dir DATA_DIR] [-out_dir OUT_DIR] [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-lr LR] [-momentum MOMENTUM] [-lr_scheduler LR_SCHEDULER] [-step_size STEP_SIZE] [-gamma GAMMA] [-T_max T_MAX]

        Classify DVS128 Gesture

        optional arguments:
          -h, --help            show this help message and exit
          -T T                  simulating time-steps
          -device DEVICE        device
          -b B                  batch size
          -epochs N             number of total epochs to run
          -j N                  number of data loading workers (default: 4)
          -channels CHANNELS    channels of Conv2d in SNN
          -data_dir DATA_DIR    root dir of DVS128 Gesture dataset
          -out_dir OUT_DIR      root dir for saving logs and checkpoint
          -resume RESUME        resume from the checkpoint path
          -amp                  automatic mixed precision training
          -cupy                 use CUDA neuron and multi-step forward mode
          -opt OPT              use which optimizer. SDG or Adam
          -lr LR                learning rate
          -momentum MOMENTUM    momentum for SGD
          -lr_scheduler LR_SCHEDULER
                                use which schedule. StepLR or CosALR
          -step_size STEP_SIZE  step_size for StepLR
          -gamma GAMMA          gamma for StepLR
          -T_max T_MAX          T_max for CosineAnnealingLR

    运行示例：

    .. code:: bash

        python -m spikingjelly.clock_driven.examples.classify_dvsg -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024

    阅读教程 :doc:`./clock_driven/14_classify_dvsg` 以获得更多信息。

    * :ref:`中文API <classify_dvsg.__init__-cn>`

    .. _classify_dvsg.__init__-en:

    The code example for classifying the DVS128 Gesture dataset. The network structure is from `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_.


    .. code:: bash

        usage: classify_dvsg.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-channels CHANNELS] [-data_dir DATA_DIR] [-out_dir OUT_DIR] [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-lr LR] [-momentum MOMENTUM] [-lr_scheduler LR_SCHEDULER] [-step_size STEP_SIZE] [-gamma GAMMA] [-T_max T_MAX]

        Classify DVS128 Gesture

        optional arguments:
          -h, --help            show this help message and exit
          -T T                  simulating time-steps
          -device DEVICE        device
          -b B                  batch size
          -epochs N             number of total epochs to run
          -j N                  number of data loading workers (default: 4)
          -channels CHANNELS    channels of Conv2d in SNN
          -data_dir DATA_DIR    root dir of DVS128 Gesture dataset
          -out_dir OUT_DIR      root dir for saving logs and checkpoint
          -resume RESUME        resume from the checkpoint path
          -amp                  automatic mixed precision training
          -cupy                 use CUDA neuron and multi-step forward mode
          -opt OPT              use which optimizer. SDG or Adam
          -lr LR                learning rate
          -momentum MOMENTUM    momentum for SGD
          -lr_scheduler LR_SCHEDULER
                                use which schedule. StepLR or CosALR
          -step_size STEP_SIZE  step_size for StepLR
          -gamma GAMMA          gamma for StepLR
          -T_max T_MAX          T_max for CosineAnnealingLR

    Running Example:

    .. code:: bash

        python -m spikingjelly.clock_driven.examples.classify_dvsg -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024

    See the tutorial :doc:`./clock_driven_en/14_classify_dvsg` for more details.
    '''
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cpu', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-channels', default=128, type=int, help='channels of Conv2d in SNN')
    parser.add_argument('-data_dir', type=str, help='root dir of DVS128 Gesture dataset')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use CUDA neuron and multi-step forward mode')


    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')


    args = parser.parse_args()
    print(args)

    if args.cupy:
        net = CextNet(channels=args.channels)
    else:
        net = PythonNet(channels=args.channels)
    print(net)
    net.to(args.device)




    optimizer = None

    lr_scheduler = None

    test_set = DVS128Gesture(args.data_dir, train=False, data_type='frame', split_by='number', duration=131072)

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.j,
        drop_last=False,
        pin_memory=True)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        net.load_state_dict(checkpoint['net'])

    for epoch in range(1):
        start_time = time.time()
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr = net(frame)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples

        print(args)
        print(f'epoch={epoch}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')

if __name__ == '__main__':
    main()