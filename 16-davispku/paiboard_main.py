import __init__
from multiprocessing import Process, Queue, Event
import os
if os.name == 'nt': # Windows
    import keyboard
elif os.name == 'posix': # Linux or MacOS
    import signal
else:
    raise Exception("Unsupported OS")
import time
import numpy as np

# 假设 events_process 和 PAIBoardProcessor 已经定义在其他模块中
# 如果它们在同一个文件中，可以直接使用
from events_process import events_process
from paiboard_process import paiboard_process
from paiboxnet_process import paiboxnet_process
LETTER_LIST = ['D', 'A', 'V', 'I', 'S', 'P', 'K', 'U', 'others']

if __name__ == "__main__":
    stop_event1 = Event()
    stop_event2 = Event()
    def on_press_callback(event):
        if event.name == 'a':
            print('You pressed the A key')
        if event.name == 'q':
            stop_event1.set()
            stop_event2.set()
            print('You pressed the Q key, exiting...')
    def signal_handler(signal, frame):
        stop_event1.set()
        stop_event2.set()
    if os.name == 'nt': # Windows
        keyboard.on_press(on_press_callback)
    elif os.name == 'posix': # Linux or MacOS
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        raise Exception("Unsupported OS")

    # 定义队列
    input_queue = Queue(maxsize=1)  # events_process 的输出队列
    output_queue = Queue(maxsize=50)  # PAIBoardProcessor 的输出队列

    # 定义全局变量
    IS_CAMERA = True
    FRAME_DELAY = 33
    FILE_PATH = "D:\\DV\\SPKU\\7_1.aedat4"
    TIME_SLEEP = 0
    baseDir = "./debug"

    # 创建事件处理进程
    events_process_p = Process(target=events_process, args=(input_queue, stop_event1, IS_CAMERA, FRAME_DELAY, FILE_PATH, TIME_SLEEP))
    events_process_p.start()

    # 创建 PAIBoard 处理进程
    # paiboard_process_p = Process(target=paiboard_process, args=(input_queue, output_queue, stop_event2, baseDir))
    paiboard_process_p = Process(target=paiboxnet_process, args=(input_queue, output_queue, stop_event2))
    paiboard_process_p.start()

    while events_process_p.is_alive() or paiboard_process_p.is_alive():
        # print("events_process_p is_alive: ",events_process_p.is_alive())
        # print("paiboard_process_p is_alive: ",paiboard_process_p.is_alive())
        # 检查 PAIBoardProcessor 的输出队列
        if not output_queue.empty():
            spike_sum_board, pred_board = output_queue.get()
            print("Received PAIBoard output:")
            print(f"Spike sum board:{spike_sum_board}\tPredicted board:{pred_board}\tPredicted letter:{LETTER_LIST[pred_board]}")
        # if input_queue.full():
            # test_data = input_queue.get()
            # print("Received test data:", test_data)

            time.sleep(0.01)
        else:
            time.sleep(0.1)
    
    # 等待进程结束
    print("Waiting for processes to finish...")
    events_process_p.join()
    paiboard_process_p.join()

