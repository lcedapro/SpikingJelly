        <!-- self.i0 = pb.InputProj(input=self.fakeout_with_t, shape_out=(1, 86, 65))
        self.n0 = pb.LIF((1, 42, 32), bias=param_dict['conv.0.bias'], threshold=param_dict['conv.0.vthr'], reset_v=0, tick_wait_start=1) # convpool7x7p2s2
        self.conv2d_0 = pb.Conv2d(self.i0, self.n0, kernel=param_dict['conv.0.weight'], padding=2, stride=2)

        self.n1 = pb.LIF((2, 21, 16), bias=param_dict['conv.2.bias'], threshold=param_dict['conv.2.vthr'], reset_v=0, tick_wait_start=2) # convpool7x7p3s2
        self.conv2d_1_0 = pb.Conv2d(self.n0, self.n1, kernel=param_dict['conv.2.weight'], padding=3, stride=2)

        self.n10 = pb.LIF(512, threshold=param_dict['fc.2.vthr'], reset_v=0, tick_wait_start=3) # fc
        self.fc_0 = pb.FullConn(self.n1, self.n10, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.2.weight'])

        self.n11 = pb.LIF(128, threshold=param_dict['fc.5.vthr'], reset_v=0, tick_wait_start=4) # fc
        self.fc_1 = pb.FullConn(self.n10, self.n11, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.5.weight'])

        self.n12 = pb.LIF(90, threshold=param_dict['fc.8.vthr'], reset_v=0, tick_wait_start=5) # fc
        self.fc_2 = pb.FullConn(self.n11, self.n12, conn_type=pb.SynConnType.All2All, weights=param_dict['fc.8.weight']) -->

| #   | Layer     | Input shape | Output shape | Operation             |
| --- | --------- | ----------- | ------------ | --------------------- |
| 0   | InputProj | (1, 86, 65) | (1, 86, 65)  | Input projection      |
| 1   | Conv2d    | (1, 86, 65) | (1, 42, 32)  | 2D convolution        |
| 2   | Conv2d    | (1, 42, 32) | (2, 21, 16)  | 2D convolution        |
| 3   | FullConn  | (2, 21, 16) | 512          | Fully connected layer |
| 4   | FullConn  | 512         | 128          | Fully connected layer |
| 5   | FullConn  | 128         | 90           | Fully connected layer |

