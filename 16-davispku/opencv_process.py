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

import cv2
if os.name == 'posix': # Linux
    import serial
LETTER_LIST = ['D', 'A', 'V', 'I', 'S', 'P', 'K', 'U', 'others']
DISPLAYSCALEFACTOR = 240

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

def opencv_process(queue, stop_event, frame_counter_max):   
    try:
        # 尝试打开 CSV 文件
        csv_file =  open("opencv_process.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["perf_counter_ns", "log_type", "log_info", "events_timestamp"])
    except Exception as e:
        print(f"Error in opencv_process: {e}")
        exit(1)  # 终止子程序，返回状态码 1 表示异常退出

    if os.name == 'posix': # Linux
        # 打开串口
        # 串口放在这吧
        try:
            #端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
            portx="/dev/ttyUSB1"
            #波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
            bps=115200
            #超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
            timex=5
            # 打开串口，并得到串口对象
            ser=serial.Serial(portx,bps,timeout=timex)

            # # 写数据
            # result=ser.write("我是东小东".encode("gbk"))
            # print("写总字节数:",result)

            # ser.close()#关闭串口

        except Exception as e:
            print("---串口初始化异常---：",e)    
    frame_counter = 0
    # Create a window for image display
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 800, 600)
    k_counter = 0
    tim_total_1 = 0
    tim_total_2 = 0
    while True:
        if stop_event.is_set():
            print("opencv_process stop")
            cv2.destroyAllWindows()
            break
        if not queue.empty():
            events_timestamp, events, spike_sum_board, pred_board = queue.get()
            tim_total_1 = time.perf_counter_ns()
            csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp])

            print(f"Opencv Process: Spike sum board:{spike_sum_board}\tPredicted board:{pred_board}\tPredicted letter:{LETTER_LIST[pred_board]}")

            if os.name == 'posix': # Linux
                # 串口
                if LETTER_LIST[pred_board] == 'S':
                    k_counter += 1
                else:
                    k_counter = 0  # 
                try:
                    if k_counter >= 8:
                        ser.write(b'\x01')
                        k_counter = 0 
                    else:
                        ser.write(b'\x00')
                except Exception as e:
                    print("---串口写入异常---：",e)

            if frame_counter >= frame_counter_max:
                # tim1 = time.perf_counter()
                frame = integrate_events_to_one_frame_1bit_optimized_numpy(events)
                frame_pos = frame[0].transpose(1,0)
                frame_neg = frame[1].transpose(1,0)
                frame_neg = np.where(frame_pos == 1, 0, frame_neg)
                # 创建一个新的 NumPy 数组，形状为 (height, width, 3)，数据类型为 uint8
                frame_BGR = np.zeros((260, 346, 3), dtype=np.uint8)
                # 将双通道数组的第一个通道复制到新数组的红色通道
                frame_BGR[:, :, 2] = frame_pos * DISPLAYSCALEFACTOR
                # 将双通道数组的第二个通道复制到新数组的绿色通道
                frame_BGR[:, :, 1] = frame_neg * DISPLAYSCALEFACTOR
                # 将新数组的蓝色通道填充为 0
                frame_BGR[:, :, 0] = 0
                frame_BGR = frame_BGR
                text1 = f"Letter: {LETTER_LIST[pred_board]}"
                text2 = f"Cred: {np.max(spike_sum_board)/40.0}"
                cv2.putText(frame_BGR, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(frame_BGR, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.imshow("Result", frame_BGR)
                cv2.waitKey(1)
                frame_counter = 0
                # tim2 = time.perf_counter()
                # print(f"Opencv Process: Frame processing time: {tim2 - tim1} seconds") # 0.014 seconds
            else:
                frame_counter += 1
            tim_total_2 = time.perf_counter_ns()
            csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp])
        else:
            time.sleep(0.005)
    cv2.destroyAllWindows()
    exit(0)
