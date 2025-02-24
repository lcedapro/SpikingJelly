import __init__
import numpy as np
import time
from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet

if __name__ == "__main__":
    timestep = 4
    layer_num = 4
    baseDir = "./debug"
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)

    snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
    snn.config(oFrmNum=90*4)

    test_num = 1
    for i in range(test_num):
        # input_spike = np.eye(timestep, dtype=np.int8)
        input_spike = np.load("./仿真输入输出示例/image/label_6_iter_24_image.npy")
        print(input_spike.shape)
        input_spike = input_spike.astype(np.uint8)
        input_spike = np.expand_dims(input_spike, axis=0).repeat(timestep, axis=0)
        print(input_spike.shape)

        t1 = time.time()
        output_spike = snn(input_spike)
        t2 = time.time()

        snn.record_time(t2 - t1)
        # assert np.equal(input_spike, output_spike).all()
        print(output_spike)
        np.save("output_spike.npy", output_spike)
    print("Test passed!")
    snn.paicore_status()
    snn.perf(test_num)
