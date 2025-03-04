import __init__
from multiprocessing import Process, Queue, Event
import multiprocessing as mp
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

# 假设 events_process 和 PAIBoardProcessor 已经定义在其他模块中
# 如果它们在同一个文件中，可以直接使用
from events_process import events_process
# from paiboard_process import paiboard_process
from paiboxnet_process import paiboxnet_process
from opencv_process import opencv_process
LETTER_LIST = ['D', 'A', 'V', 'I', 'S', 'P', 'K', 'U', 'others']

if __name__ == "__main__":
    stop_event1 = Event()
    stop_event2 = Event()
    stop_event3 = Event()
    def on_press_callback(event):
        if event.name == 'a':
            print('You pressed the A key')
        if event.name == 'q':
            stop_event1.set()
            stop_event2.set()
            stop_event3.set()
            print('You pressed the Q key, exiting...')
    def signal_handler(signal, frame):
        stop_event1.set()
        stop_event2.set()
        stop_event3.set()
    if os.name == 'nt': # Windows
        keyboard.on_press(on_press_callback)
    elif os.name == 'posix': # Linux or MacOS
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        raise Exception("Unsupported OS")

    # 新建主进程log输出
    try:
        # 尝试打开 CSV 文件
        csv_file =  open("main_process.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["perf_counter_ns", "log_type", "log_info", "events_timestamp"])
    except Exception as e:
        print(f"Error in main_process: {e}")
        exit(1)  # 终止子程序，返回状态码 1 表示异常退出

    # 创建共享内存
    events_timestamp_value1 = mp.Value('L', lock=True)
    image_array1 = mp.Array('B', 86*65, lock=True)
    con1 = mp.Condition()

    events_timestamp_value2 = mp.Value('L', lock=True)
    result_array2 = mp.Array('B', 9, lock=True)
    con2 = mp.Condition()

    events_timestamp_value3 = mp.Value('L', lock=True)
    cv_image_array3 = mp.Array('B', 86*65, lock=True)
    cv_result_array3 = mp.Array('B', 9, lock=True)
    con3 = mp.Condition()

    # 定义全局变量
    IS_CAMERA = False
    FRAME_DELAY = 1
    FILE_PATH = "D:\\DV\\SPKU\\7_1.aedat4"
    TIME_SLEEP = 0.1

    # 创建事件处理进程
    events_process_p = Process(target=events_process, args=(events_timestamp_value1, image_array1, con1, stop_event1, IS_CAMERA, FRAME_DELAY, FILE_PATH, TIME_SLEEP))
    events_process_p.start()

    # 创建 OpenCV 处理进程
    opencv_process_p = Process(target=opencv_process, args=(events_timestamp_value3, cv_image_array3, cv_result_array3, con3, stop_event3))
    opencv_process_p.start()

    # 创建 PAIBoard 处理进程
    # paiboard_process_p = Process(target=paiboard_process, args=(events_timestamp_value1, image_array1, con1,
    #                                                              events_timestamp_value2, result_array2, con2,
    #                                                              events_timestamp_value3, cv_image_array3, cv_result_array3, con3,
    #                                                              stop_event2, 2))
    paiboard_process_p = Process(target=paiboxnet_process, args=(events_timestamp_value1, image_array1, con1,
                                                                 events_timestamp_value2, result_array2, con2,
                                                                 events_timestamp_value3, cv_image_array3, cv_result_array3, con3,
                                                                 stop_event2, 2))
    paiboard_process_p.start()


    # 主进程开始数据接收处理
    events_timestamp_main = 0

    tim_total_1 = 0
    tim_total_2 = 0

    while events_process_p.is_alive() or opencv_process_p.is_alive() or paiboard_process_p.is_alive():
        # print("events_process_p is_alive: ",events_process_p.is_alive())
        # print("paiboard_process_p is_alive: ",paiboard_process_p.is_alive())
        # 从前置模块获取数据
        with con2:
            if events_timestamp_value2.value == events_timestamp_main:
                con2.wait(timeout=0.5)
            if events_timestamp_value2.value == events_timestamp_main:
                print("main_process timeout")
                continue
            print("main_process get data")
            events_timestamp_main = events_timestamp_value2.value
            spike_sum_board = np.frombuffer(result_array2.get_obj(), dtype=np.uint8)
    
        # Start main process
        tim_total_1 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp_main])

        pred_board = np.argmax(spike_sum_board)
        print(f"Main Process: Spike sum board:{spike_sum_board}\tPredicted board:{pred_board}\tPredicted letter:{LETTER_LIST[pred_board]}")

        tim_total_2 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp_main])

        if stop_event1.is_set() or stop_event2.is_set() or stop_event3.is_set():
            time.sleep(0.5)
            break

    # 等待进程结束
    print("Main Process: Waiting for processes to finish...")
    events_process_p.join()
    opencv_process_p.join()
    paiboard_process_p.join()

