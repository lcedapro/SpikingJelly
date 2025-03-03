from multiprocessing import Process, Event, Queue
import keyboard
import time
import numpy as np
import csv

from simple_pb_infer import PAIBoxNet

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
    x = events[['x']]
    y = events[['y']]
    polarity = events[['polarity']]

    x = np.array(x, dtype=np.int16)
    y = np.array(y, dtype=np.int16)
    polarity = np.array(polarity, dtype=np.int8)

    # Use NumPy's advanced indexing to set the frame values
    frame[polarity, x, y] = 1

    return frame


def paiboxnet_process(input_queue, output_queue, stop_event):
    """后台处理函数，用于PAIBox推理"""
    try:
        # 尝试打开 CSV 文件
        csv_file =  open("paiboard_process.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["perf_counter_ns", "log_type", "log_info", "events_timestamp"])
    except Exception as e:
        print(f"Error in paiboard_process: {e}")
        exit(1)  # 终止子程序，返回状态码 1 表示异常退出

    paiboxnet = PAIBoxNet(2, 4,
         './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth',
         './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy')

    tim_total_1 = 0
    tim_total_2 = 0

    while True:
        if stop_event.is_set():
            print("paiboxnet_process stop")
            break

        # 检查输入队列是否为空
        if not input_queue.empty():
            # print("Input queue is not empty, start processing")
            # 从输入队列获取数据
            events_timestamp, events = input_queue.get()
            tim_total_1 = time.perf_counter_ns()
            csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp])

            # image = input_queue.get()
            # print(events.shape) # (3500, 3)
            image = integrate_events_to_one_frame_1bit_optimized_numpy(events)
            # print(image.shape) # (2, 346, 260)
            # image = maxpool2ds4(image)
            # # print(image.shape) # (2, 86, 65)
            # image = image[0]
            # # print(image.shape) # (86, 65)

            spike_sum_pb, pred_pb = paiboxnet.pb_inference(image.repeat(4, axis=0))

            # 将结果放入输出队列
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Processing finished, Start to put result into output queue blocking", events_timestamp])
            output_queue.put((events_timestamp, events, spike_sum_pb, pred_pb))
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Put result into output queue finished", events_timestamp])
            tim_total_2 = time.perf_counter_ns()
            csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp])
        else:
            # print("Input queue is empty, waiting...")
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Input queue is empty, waiting..."])
            time.sleep(0.1)
    # close csv file
    csv_file.close()
    exit(0)

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
    process = Process(target=paiboxnet_process, args=(input_queue, output_queue, stop_event, baseDir))
    process.start()

    # 向输入队列添加数据
    for _ in range(1):
        random_x = np.random.randint(0, 346, (3500,), dtype=np.int16)
        random_y = np.random.randint(0, 260, (3500,), dtype=np.int16)
        random_polarity = np.random.randint(0, 2, (3500,), dtype=np.int16)
        # 改成结构化数组
        dtype = [('x', np.int16), ('y', np.int16), ('polarity', np.int8)]
        random_input = np.array(list(zip(random_x, random_y, random_polarity)), dtype=dtype)
        input_queue.put(random_input)
        time.sleep(1)

    # 等待子进程完成
    time.sleep(1)

    process.join()