import __init__
from multiprocessing import Process, Event, Queue
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

import cv2
if os.name == 'posix': # Linux
    import serial
LETTER_LIST = ['D', 'A', 'V', 'I', 'S', 'P', 'K', 'U', 'others']
DISPLAYSCALEFACTOR = 240

def opencv_process(events_timestamp_value3:mp.Value, cv_image_array3:mp.Array, cv_result_array3:mp.Array, con3:mp.Condition, stop_event:mp.Event):   
    try:
        # 尝试打开 CSV 文件
        csv_file =  open("opencv_process.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["perf_counter_ns", "log_type", "log_info", "events_timestamp"])
    except Exception as e:
        print(f"Error in opencv_process: {e}")
        exit(1)  # 终止子程序，返回状态码 1 表示异常退出

    # Create a window for image display
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 800, 600)

    events_timestamp = 0

    tim_total_1 = 0
    tim_total_2 = 0

    while True:
        if stop_event.is_set():
            print("opencv_process stop")
            break

        # 从前置模块获取数据
        with con3:
            if events_timestamp_value3.value == events_timestamp:
                con3.wait(timeout=0.5)
            if events_timestamp_value3.value == events_timestamp:
                print("opencv_process timeout")
                continue
            print("opencv_process get data")
            events_timestamp = events_timestamp_value3.value
            cv_image = np.frombuffer(cv_image_array3.get_obj(), dtype=np.uint8).reshape((86, 65))
            spike_sum_board = np.frombuffer(cv_result_array3.get_obj(), dtype=np.uint8)

        # Start main process
        tim_total_1 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp])
        
        # print(f"Opencv Process: Spike sum board:{spike_sum_board}\tPredicted board:{pred_board}\tPredicted letter:{LETTER_LIST[pred_board]}")

        # tim1 = time.perf_counter()
        cv_image_pos = cv_image.transpose(1,0)
        # 创建一个新的 NumPy 数组，形状为 (height, width, 3)，数据类型为 uint8
        frame_BGR = np.zeros((65, 86, 3), dtype=np.uint8)
        # 将双通道数组的第一个通道复制到新数组的红色通道
        frame_BGR[:, :, 2] = cv_image_pos * DISPLAYSCALEFACTOR
        # 将双通道数组的第二个通道复制到新数组的绿色通道
        frame_BGR[:, :, 1] = 0
        # 将新数组的蓝色通道填充为 0
        frame_BGR[:, :, 0] = 0
        # resize frame_BGR to (65*4, 86*4)
        frame_BGR = cv2.resize(frame_BGR, (65*4, 86*4))
        pred_board = np.argmax(spike_sum_board)
        text1 = f"Letter: {LETTER_LIST[pred_board]}"
        text2 = f"Cred: {np.max(spike_sum_board)/40.0}"
        cv2.putText(frame_BGR, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        cv2.putText(frame_BGR, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        cv2.imshow("Result", frame_BGR)
        cv2.waitKey(1)

        tim_total_2 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp])
    cv2.destroyAllWindows()
    exit(0)
