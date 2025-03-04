from multiprocessing import Process, Event, Queue, Condition
import multiprocessing as mp
import keyboard
import time
import numpy as np
import csv

from simple_pb_infer import PAIBoxNet

def paiboxnet_process(events_timestamp_value1:mp.Value, image_array1:mp.Array, con1:mp.Condition, events_timestamp_value2:mp.Value, result_array:mp.Array, con2:mp.Condition, events_timestamp_value3:mp.Value, cv_image_array3:mp.Array, cv_result_array3:mp.Array, con3:mp.Condition, stop_event:mp.Event, frame_counter_max:int):
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

    frame_counter = 0
    events_timestamp = 0

    tim_total_1 = 0
    tim_total_2 = 0

    while True:
        if stop_event.is_set():
            print("paiboxnet_process stop")
            break

        # 从前置模块获取数据
        with con1:
            if events_timestamp_value1.value == events_timestamp:
                con1.wait(timeout=10)
            if events_timestamp_value1.value == events_timestamp:
                print("paiboxnet_process timeout")
                continue
            print("paiboxnet_process get data")
            events_timestamp = events_timestamp_value1.value
            image = np.frombuffer(image_array1.get_obj(), dtype=np.uint8).reshape((86, 65))

        # Start main process
        tim_total_1 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp])

        spike_sum_board, pred_board = paiboxnet.pb_inference(np.expand_dims(image, axis=0).repeat(4, axis=0))

        # 将结果放入共享内存
        csv_writer.writerow([time.perf_counter_ns(), "LOG", "Processing finished, Start to put result into shared memory blocking", events_timestamp])
        with con2:
            events_timestamp_value2.value = events_timestamp
            np.copyto(np.frombuffer(result_array.get_obj(), dtype=np.uint8), spike_sum_board)
            con2.notify_all()
        csv_writer.writerow([time.perf_counter_ns(), "LOG", "Put result into shared memory finished", events_timestamp])

        # 将结果放入共享内存
        if frame_counter >= frame_counter_max:
            frame_counter = 0
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Processing finished, Start to put result into shared memory blocking", events_timestamp])
            with con3:
                events_timestamp_value3.value = events_timestamp
                np.copyto(np.frombuffer(cv_image_array3.get_obj(), dtype=np.uint8), image.flatten())
                np.copyto(np.frombuffer(cv_result_array3.get_obj(), dtype=np.uint8), spike_sum_board)
                con3.notify_all()
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "Put result into shared memory finished", events_timestamp])
        frame_counter += 1

        tim_total_2 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp])
    
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

    # 创建子进程
    process = Process(target=paiboxnet_process, args=(events_timestamp_value1, image_array1, con1,
                                                      events_timestamp_value2, result_array2, con2,
                                                      events_timestamp_value3, cv_image_array3, cv_result_array3, con3,
                                                      stop_event, 30))
    process.start()


    # 向共享内存添加数据
    for _i in range(1):
        image = np.random.randint(0, 256, size=(86, 65), dtype=np.uint8)
        events_timestamp = time.perf_counter_ns()
        with con1:
            events_timestamp_value1.value = events_timestamp
            np.copyto(np.frombuffer(image_array1.get_obj(), dtype=np.uint8), image.flatten())
            con1.notify_all()
        time.sleep(0.1)

    # 从共享内存读取数据
    previous_events_timestamp_main = 0

    while not stop_event.is_set():
        with con2:
            tim1 = time.time()
            if events_timestamp_value2.value == previous_events_timestamp_main:
                print("Waiting for result...")
                con2.wait(timeout=4)
            print("main process get data")
            previous_events_timestamp_main = events_timestamp_value2.value
            result = np.frombuffer(result_array2.get_obj(), dtype=np.uint8)
            print("Result:", result)
            tim2 = time.time()
            print("time:", tim2-tim1)
        time.sleep(0.1)

    # 等待子进程完成
    time.sleep(1)

    process.join()