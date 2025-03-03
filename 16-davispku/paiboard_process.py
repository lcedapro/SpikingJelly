import __init__
from multiprocessing import Process, Event, Queue
import os
if os.name == 'nt': # Windows
    import keyboard
elif os.name == 'posix': # Linux or MacOS
    import signal
else:
    raise Exception("Unsupported OS")
import time
import numpy as np
import csv

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
        frame (np.ndarray): Integrated frame as a numpy array with shape [2, 86, 65]. (maxpool2d, stride=4)
    """
    if len(events) == 0:
        return np.zeros((2, 86, 65), dtype=np.uint8)

    # Initialize an empty frame with shape [1, 2, x_max, y_max]
    frame = np.zeros((2, 86, 65), dtype=np.uint8)

    # Extract the x, y, and polarity columns as NumPy arrays
    x = events[['x']]
    y = events[['y']]
    polarity = events[['polarity']]

    x = np.array(x, dtype=np.int16)
    y = np.array(y, dtype=np.int16)
    polarity = np.array(polarity, dtype=np.int8)

    # maxpool2d, stride=4
    x = x // 4
    x = np.where(x > 85, 85, x)
    y = y // 4
    y = np.where(y > 64, 64, y)

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
    try:
        # 尝试打开 CSV 文件
        csv_file =  open("paiboard_process.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["perf_counter_ns", "log_type", "log_info", "events_timestamp"])
    except Exception as e:
        print(f"Error in paiboard_process: {e}")
        exit(1)  # 终止子程序，返回状态码 1 表示异常退出

    timestep = 4
    layer_num = 4

    # 初始化PAIBoard
    if os.name == 'nt': # Windows
        snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
        # snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
        # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    elif os.name == 'posix': # Linux or MacOS
        # snn = PAIBoard_SIM(baseDir, timestep, layer_num=layer_num)
        snn = PAIBoard_PCIe(baseDir, timestep, layer_num=layer_num)
        # snn = PAIBoard_Ethernet(baseDir, timestep, layer_num=layer_num)
    else:
        raise Exception("Unsupported OS")
    # snn.chip_init([(1, 0), (0, 0), (1, 1), (0, 1)])
    snn.config(oFrmNum=90 * 4)

    tim_total_1 = 0
    tim_total_2 = 0

    while True:
        if stop_event.is_set():
            print("paiboard_process stop")
            break

        # 检查输入队列是否为空
        if not input_queue.empty():
            # print("PAIBoard Process: Input queue is not empty, start processing")
            # 从输入队列获取数据
            events_timestamp, events = input_queue.get()
            tim_total_1 = time.perf_counter_ns()
            csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp])

            
            tim1 = time.perf_counter()
            # image = input_queue.get()
            # print(events.shape) # (3500, 3)
            image = integrate_events_to_one_frame_1bit_optimized_numpy(events)

            # tim2 = time.perf_counter()
            # print(f"PAIBoard Process: integrate_events_to_one_frame_1bit_optimized_numpy time: {(tim2 - tim1)*1000} ms", )
            # tim1 = time.perf_counter()

            # print(image.shape) # (2, 346, 260)
            image = image[0]
            # print(image.shape) # (346, 260)
            # image = maxpool2ds4(image)
            # print(image.shape) # (86, 65)

            # tim2 = time.perf_counter()
            # print(f"PAIBoard Process: numpy_maxpool2d time: {(tim2 - tim1)*1000} ms", )
            # tim1 = time.perf_counter()

            tim2 = time.perf_counter()
            # print(f"PAIBoard Process: PAIBoard preprocess time: {(tim2 - tim1)*1000} ms", )
            # csv_writer.writerow([time.perf_counter_ns(), f"PAIBoard preprocess time: {(tim2 - tim1)*1000} ms"])
            tim1 = time.perf_counter()

            # PAIBoard 推理
            input_spike = np.expand_dims(image, axis=0).repeat(4, axis=0)
            spike_out = snn(input_spike)
            spike_out = voting(spike_out, 10)
            spike_sum_board = spike_out.sum(axis=0)
            pred_board = np.argmax(spike_sum_board)

            tim2 = time.perf_counter()
            # print(f"PAIBoard Process: PAIBoard inference time: {(tim2 - tim1)*1000} ms", )
            # csv_writer.writerow([time.perf_counter_ns(), f"PAIBoard inference time: {(tim2 - tim1)*1000} ms"])

            # 将结果放入输出队列
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Processing finished, Start to put result into output queue blocking", events_timestamp])
            output_queue.put((events_timestamp, events, spike_sum_board, pred_board))
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Put result into output queue finished", events_timestamp])
            tim_total_2 = time.perf_counter_ns()
            csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp])
        else:
            # print("PAIBoard Process: Input queue is empty, waiting...")
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Input queue is empty, waiting..."])
            time.sleep(0.001)
    # close csv file
    csv_file.close()
    exit(0)

# 示例用法
if __name__ == "__main__":
    stop_event = Event()
    def on_press_callback(event):
        if event.name == 'a':
            print('You pressed the A key')
        if event.name == 'q':
            stop_event.set()
            print('You pressed the Q key, exiting...')
    def signal_handler(signal, frame):
        stop_event.set()
    if os.name == 'nt': # Windows
        keyboard.on_press(on_press_callback)
    elif os.name == 'posix': # Linux or MacOS
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        raise Exception("Unsupported OS")

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