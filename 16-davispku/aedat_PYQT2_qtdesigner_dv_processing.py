import threading
import pandas as pd
from dv import AedatFile
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextBrowser, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore
# QTimer
from PyQt5.QtCore import QTimer

from UI.aedat4 import Ui_MainWindow

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

from base.csv2frames_1process import integrate_events_to_one_frame_1bit_optimized
FRAME_DELAY = 33
DISPLAYSCALEFACTOR = 240
LETTER_LIST = ['D', 'A', 'V', 'I', 'S', 'P', 'K', 'U', 'others']

import time

from simple_pb_infer import PAIBoxNet
import numpy as np


import dv_processing as dv
import cv2 as cv
from datetime import timedelta


CAMERA_OR_READER = 'camera'
# CAMERA_OR_READER = 'reader'
if CAMERA_OR_READER == 'camera':
    TIME_SLEEP = 0
else:
    TIME_SLEEP = 0.1


class AedatProcessor(threading.Thread):
    """后台处理线程，用于读取 .aedat4 文件事件"""
    def __init__(self, file_path, time_interval):
        super().__init__()
        self.file_path = file_path
        self.time_interval = time_interval
        self.data_chunk = None
        self.data_ready = threading.Event()
        self.stop_event = threading.Event()
        # Open the camera, just use first detected DAVIS camera
        if CAMERA_OR_READER == 'camera':
            self.camera_or_reader = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS)
        else:
            self.camera_or_reader = dv.io.MonoCameraRecording(self.file_path)

        # Initialize a multi-stream slicer
        self.slicer = dv.EventStreamSlicer()

        # Initialize a visualizer for the overlay
        self.visualizer = dv.visualization.EventVisualizer(self.camera_or_reader.getEventResolution(), dv.visualization.colors.white(),
                                                    dv.visualization.colors.green(), dv.visualization.colors.red())
        
        # # Create a window for image display
        # cv.namedWindow("Preview", cv.WINDOW_NORMAL)

    # Callback method for time based slicing
    def display_preview(self, data):
        # Retrieve event data
        events = data

        # # Generate a preview and show the final image
        # cv.imshow("Preview", self.visualizer.generateImage(events))

        # # If escape button is pressed (code 27 is escape key), exit the program cleanly
        # if cv.waitKey(2) == 27:
        #     exit(0)
            
        # return events numpy
        # print(events.numpy())
        print(events)
        self.data_chunk = events.numpy()
        self.data_ready.set()

    def run(self):
        # Register a job to be performed every 33 milliseconds
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=FRAME_DELAY), self.display_preview)

        """读取事件流并分块生成数据"""
        # Continue the loop while both cameras are connected
        while self.camera_or_reader.isRunning():
            if self.stop_event.is_set():
                break
            events = self.camera_or_reader.getNextEventBatch()
            if events is not None:
                self.slicer.accept(events)
            time.sleep(TIME_SLEEP)

    def stop(self):
        """停止线程"""
        self.stop_event.set()
        self.data_ready.set()  # 唤醒线程退出


class AedatGUI(QMainWindow, Ui_MainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.setupUi(self)

        # 连接按钮事件
        self.pushButton.clicked.connect(self.show_next_frame)
        self.pushButton_2.clicked.connect(self.quit_application)
        self.pushButton_3.clicked.connect(self.auto_start_stop)

        # 启动处理线程
        self.processor.start()

        # PAIBox仿真器
        self._sim_timestep = 4
        self.paiboxnet = PAIBoxNet(2, self._sim_timestep,
             './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/checkpoint_max_conv2int.pth',
             './logs_t1e4_simple/T_4_b_16_c_2_SGD_lr_0.4_CosALR_48_amp_cupy/vthr_list.npy')
        # 计数器（每4个frame(shape=(346, 260))形成一个frames(shape=(4, 2, 346, 260))
        self.frame_counter = 0
        # 积分好的帧（shape=(4, 2, 346, 260)）
        self.integrated_frames = np.zeros((self._sim_timestep, 2, 346, 260))

        # 自动点击
        self.auto_start_stop_flag = False # False表示上一次是停止状态，True表示上一次是开始状态
        self.frame_delay = FRAME_DELAY  # 间隔时间，单位毫秒

    def show_next_frame(self):
        """显示下一帧数据"""
        if not self.processor.data_ready.is_set():
            self.textBrowser_3.setText("No data ready yet. Please wait...")
            return

        # 锁定数据块并提取
        data_chunk = self.processor.data_chunk
        # 显示数据
        if data_chunk is not None:
            if data_chunk is None:
                self.textBrowser_3.setText("No events in this time window.")
            else:
                data_chunk = pd.DataFrame(data_chunk, columns=['timestamp', 'x', 'y', 'polarity']) # numpy -> dataframe
                # save csv
                data_chunk.to_csv('data_chunk.csv', index=False)
                self.textBrowser_3.setText(data_chunk.to_csv(index=False))
                # DEBUG: 显示integrate_events_to_one_frame_1bit方法的执行时间
                starttime = time.perf_counter()
                frames = integrate_events_to_one_frame_1bit_optimized(data_chunk) # pd.DataFrame -> np.ndarray(shape=(1, 2, 346, 260))
                endtime = time.perf_counter()
                print(f"integrate_events_to_one_frame_1bit的执行时间：{(endtime - starttime)*1000:.6f}毫秒")
                # 更新帧可视化及显示
                self.update_frame_display(frames)
                # 更新PAIBox推理及显示
                self.update_paibox_inference(frames)

            # self.textBrowser.setText(str(self.processor.current_start))
            # self.textBrowser_2.setText(str(self.processor.current_end))

        # 通知后台线程继续处理下一帧
        self.processor.data_ready.clear()  # 清除数据准备信号
    
    def update_frame_display(self, frames: np.ndarray):
        """更新帧可视化及显示"""
        print(frames.shape) # (1, 2, 346, 260)
        # frame_0 = frames[0][0] # np.ndarray(shape=(1, 2, 346, 260)) -> np.ndarray(shape=(346, 260))
        # frame_0 = frame_0 * DISPLAYSCALEFACTOR # displayscale
        # q_image = QImage(frame_0.data, frame_0.shape[1], frame_0.shape[0], QImage.Format_Grayscale8) # np.ndarray -> QImage
        # q_pixmap = QPixmap.fromImage(q_image) # QImage -> QPixmap
        # self.label_4.setPixmap(q_pixmap)
        # self.label_4.setScaledContents(True)

        # 改为：frames[0][0]显示为红色，frames[0][1]显示为绿色
        # 获取数组的形状
        width, height = frames.shape[2], frames.shape[3]
        # 创建一个新的 NumPy 数组，形状为 (height, width, 3)，数据类型为 uint8
        frame_RGB = np.zeros((width, height, 3), dtype=np.uint8)
        # 将双通道数组的第一个通道复制到新数组的红色通道
        frame_RGB[:, :, 0] = frames[0, 0, :, :] * DISPLAYSCALEFACTOR
        # 将双通道数组的第二个通道复制到新数组的绿色通道
        frame_RGB[:, :, 1] = frames[0, 1, :, :] * DISPLAYSCALEFACTOR
        # 将新数组的蓝色通道填充为 0
        frame_RGB[:, :, 2] = 0
        # 将新数组转换为 uint32 格式，并按照 RGB888 格式进行排列
        # frame_RGB = frame_RGB.astype(np.uint32)
        # frame_RGB = (frame_RGB[:, :, 0] << 16) | (frame_RGB[:, :, 1] << 8) | frame_RGB[:, :, 2]
        # 将 NumPy 数组转换为 QImage
        q_image = QImage(frame_RGB.data, height, width, QImage.Format_RGB888)
        # 将 QImage 转换为 QPixmap
        q_pixmap = QPixmap.fromImage(q_image)
        # 设置 QLabel 的 QPixmap
        self.label_4.setPixmap(q_pixmap)
        self.label_4.setScaledContents(True)

    def update_paibox_inference(self, frames: np.ndarray):
        """更新PAIBox推理及显示"""
        # 如果积累到一定数量，则进行PAIBox推理
        self.lcdNumber.display(0)
        # 进行PAIBox推理
        print(frames.shape)
        print(frames[:,0:1,:,:].shape)
        print(frames[:,0:1,:,:].repeat(4, axis=0).shape)
        spike_sum_pb, pred_pb = self.paiboxnet.pb_inference(frames[:,0:1,:,:].repeat(4, axis=0)) # t1e4
        print("spike_sum_pb:", spike_sum_pb)
        print("pred_pb:", pred_pb)
        # 显示PAIBox推理结果
        predicted_letter = LETTER_LIST[pred_pb]
        self.textBrowser_4.setText(str(spike_sum_pb))
        self.textBrowser_5.setText(predicted_letter)
        # 显示置信度
        credit = spike_sum_pb[pred_pb] / 40.0
        self.textBrowser_6.setText(str(credit))

    def quit_application(self):
        """退出程序"""
        self.processor.stop()
        self.close()

    def closeEvent(self, event):
        """重写窗口关闭事件"""
        self.quit_application()
    
    def auto_start_stop(self):
        """
        自动开始/停止按键回调函数
        读取lineEdit中的数字，如果是上一次是停止状态，则使能_auto_show_next_frame函数每隔该数字毫秒触发一次show_next_frame()，即自动点击“开始”按钮
        """
        # static变量，用于记录上一次的自动开始/停止状态
        self.auto_start_stop_flag = not self.auto_start_stop_flag
        if self.auto_start_stop_flag:
            # 获取lineEdit中的数字
            try:
                self.interval = int(self.lineEdit.text())
            except ValueError:
                # 如果lineEdit中不是数字，则使用默认值1000
                self.interval = 1000
            # 注册定时器，每隔interval毫秒触发一次show_next_frame()，即自动点击“开始”按钮
            self.timer = QTimer()
            self.timer.timeout.connect(self.show_next_frame)
            self.timer.start(self.interval)
        else:
            # 如果上一次是开始状态，则停止自动点击“开始”按钮
            self.timer.stop()
            self.timer.timeout.disconnect(self.show_next_frame)
    
    # def auto_show_next_frame(self):
    #     """自动显示下一帧"""
    #     # 获取lineEdit中的数字
    #     try:

if __name__ == "__main__":
    input_aedat4_path = "D:\\DV\\SPKU\\9_1.aedat4"
    time_window = FRAME_DELAY  # 时间窗口大小（单位：微秒）

    # 创建处理器
    processor = AedatProcessor(input_aedat4_path, time_window)

    # 创建应用程序
    app = QApplication(sys.argv)
    gui = AedatGUI(processor)
    gui.show()
    sys.exit(app.exec_())

# time.perf_counter() 调试结果
# 锁定数据块并提取时间: 0.0048999791033566 毫秒
# 设置文本时间: 0.1925000105984509 毫秒
# 转换数据时间: 162.59839999838732 毫秒
# 处理帧时间: 0.12489999062381685 毫秒
# 转换为QImage时间: 0.40610000723972917 毫秒
# 转换为QPixmap时间: 0.10289999772794545 毫秒
# 设置QPixmap时间: 0.03809999907389283 毫秒
# 显示数据总时间: 164.87740000593476 毫秒
# 通知后台线程时间: 0.06150000263005495 毫秒
# 锁定数据块并提取时间: 0.00580001506023109 毫秒
# 设置文本时间: 0.17340001068077981 毫秒
# 转换数据时间: 149.8814999940805 毫秒
# 处理帧时间: 0.1621000119484961 毫秒
# 转换为QImage时间: 0.039000005926936865 毫秒
# 转换为QPixmap时间: 0.062000006437301636 毫秒
# 设置QPixmap时间: 0.042400002712383866 毫秒
# 显示数据总时间: 151.6704999958165 毫秒
# 通知后台线程时间: 0.02549999044276774 毫秒
# 锁定数据块并提取时间: 0.0051000097300857306 毫秒
# 设置文本时间: 0.1936999906320125 毫秒
# 转换数据时间: 148.97550002206117 毫秒
# 处理帧时间: 0.05699999746866524 毫秒
# 转换为QImage时间: 0.0247000134550035 毫秒
# 转换为QPixmap时间: 0.06389999180100858 毫秒
# 设置QPixmap时间: 0.029499991796910763 毫秒
# 显示数据总时间: 150.97039999091066 毫秒
# 通知后台线程时间: 0.02920001861639321 毫秒