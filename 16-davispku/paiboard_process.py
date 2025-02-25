from multiprocessing import Process, Event, Queue
import keyboard
import time
import numpy as np

from paiboard import PAIBoard_SIM
from paiboard import PAIBoard_PCIe
from paiboard import PAIBoard_Ethernet
from voting import voting
from numpy_maxpool2d import numpy_maxpool2d

def integrate_events_to_one_frame_1bit_optimized_numpy(events: np.ndarray):
    """
    Integrate all DVS camera events into a single frame.

    Parameters:
        events (np.ndarray): Array containing event data. shape: [[x, y, polarity], ...]

    Returns:
        frame (np.ndarray): Integrated frame as a numpy array with shape [1, 2, 346, 260].
    """
    if len(events) == 0:
        return np.zeros((2, 346, 260), dtype=np.uint8)

    # Initialize an empty frame with shape [1, 2, x_max, y_max]
    frame = np.zeros((2, 346, 260), dtype=np.uint8)

    # Extract the x, y, and polarity columns as NumPy arrays
    x = events[:, 0]
    y = events[:, 1]
    polarity = events[:, 2]

    # Use NumPy's advanced indexing to set the frame values
    frame[polarity, x, y] = 1

    return frame

# import torch
# def maxpool2ds4(frame: np.ndarray):
#     frame = frame.astype(np.uint8)
#     frame = torch.from_numpy(frame)
#     frame = torch.nn.functional.max_pool2d(frame, kernel_size=4, stride=4)
#     frame = frame.numpy()
#     frame = frame.astype(np.uint8)
#     return frame

def paiboard_process(input_queue, output_queue, stop_event, baseDir):
    """后台处理函数，用于PAIBoard推理"""
    timestep = 4
    layer_num = 4

    # 初始化PAIBoard
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
    # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
    snn.config(oFrmNum=90 * 4)

    while True:
        if stop_event.is_set():
            print("paiboard_process stop")
            break

        # 检查输入队列是否为空
        if not input_queue.empty():
            print("Input queue is not empty, start processing")
            # 从输入队列获取数据
            events = input_queue.get()
            
            # image = input_queue.get()
            # print(events.shape) # (3500, 3)
            image = integrate_events_to_one_frame_1bit_optimized_numpy(events)
            # print(image.shape) # (2, 346, 260)
            image = image[0]
            # print(image.shape) # (346, 260)
            image = numpy_maxpool2d(image, 4, 4)
            # print(image.shape) # (86, 65)


            tim1 = time.perf_counter()

            # PAIBoard 推理
            input_spike = np.expand_dims(image, axis=0).repeat(4, axis=0)
            spike_out = snn(input_spike)
            spike_out = voting(spike_out, 10)
            spike_sum_board = spike_out.sum(axis=0)
            pred_board = np.argmax(spike_sum_board)

            tim2 = time.perf_counter()
            print("PAIBoard inference time: ", tim2 - tim1)
            print("spike_sum_board: ", spike_sum_board)
            print("pred_board: ", pred_board)

            # 将结果放入输出队列
            output_queue.put((spike_sum_board, pred_board))
        else:
            print("Input queue is empty, waiting...")
            time.sleep(0.1)

# 示例用法
if __name__ == "__main__":
    stop_event = Event()
    def on_press_callback(event):
        if event.name == 'a':
            print('You pressed the A key')
        if event.name == 'esc':
            stop_event.set()
            print('You pressed the ESC key, exiting...')
    keyboard.on_press(on_press_callback)

    baseDir = "./debug"
    input_queue = Queue(maxsize=20)
    output_queue = Queue()

    # 创建子进程
    process = Process(target=paiboard_process, args=(input_queue, output_queue, stop_event, baseDir))
    process.start()

    # 向输入队列添加数据
    for _ in range(1):
        random_x = np.random.randint(0, 346, (3500,), dtype=np.int16)
        random_y = np.random.randint(0, 260, (3500,), dtype=np.int16)
        random_polarity = np.random.randint(0, 2, (3500,), dtype=np.int16)
        random_input = np.stack([random_x, random_y, random_polarity], axis=1)
        input_queue.put(random_input)
        time.sleep(1)

    # 等待子进程完成
    time.sleep(1)

    process.join()