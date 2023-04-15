from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QGridLayout, QRadioButton
from PyQt6.QtGui import QPixmap, QImage
import sys
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QSize, QMutex, QObject
import numpy as np
import os
import ipdb
import random
from collections import defaultdict
import time
import romp
from queue import Queue
from imutils.video import FPS
import ffmpeg

import faulthandler
faulthandler.enable()
import time
import subprocess as sp
import queue


############################ 使用ffmpeg的方式处理视频 #######################################
class RTSPReader(QThread):
    url_signal = pyqtSignal(str)
    def __init__(self, rtsp_url, frame_buffer):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.frame_buffer = frame_buffer
        self.video_height = 720
        self.video_width = 1280
        self.ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer", 
            "-flags", "low_delay",
            "-i", self.rtsp_url, 
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-y", "pipe:1"
         ]
        self.ffmpeg_process = sp.Popen(self.ffmpeg_cmd, stdout=sp.PIPE)
        self.url_flag = False

    def run(self):
        while True:
            time_1 = time.time()
            raw_frame = self.ffmpeg_process.stdout.read(self.video_height * self.video_width * 3)
            if len(raw_frame) != self.video_height * self.video_width * 3:
                continue
            try:
                self.frame_buffer.put(raw_frame, block=False)
            except queue.Full: 
                print("Buffer is full\n *\n *\n *\n *\n *\n *\n *\n *\n *\n *\n")
                with self.frame_buffer.mutex:
                    self.frame_buffer.queue.clear()
            if self.url_flag:
                with self.frame_buffer.mutex:
                    self.frame_buffer.queue.clear()
                self.url_flag = False
                print("*"*60)
                print(self.url_flag)
            
            time_2 = time.time()
            print('RTSPReader fps' + str(1 / (time_2 - time_1)))
    
    @pyqtSlot(str)
    def url_update(self, url_):
        if url_ != self.rtsp_url:
            self.rtsp_url = url_
            self.ffmpeg_cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer", 
                "-flags", "low_delay",
                "-i", self.rtsp_url, 
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-y", "pipe:1"
            ]
            self.ffmpeg_process = sp.Popen(self.ffmpeg_cmd, stdout=sp.PIPE)
            self.url_flag = True


class VideoProcessThread(QThread):
    """读取需要处理的Video."""
    change_pixmap_signal = pyqtSignal(QPixmap, QPixmap)
    bg_singal = pyqtSignal(str)

    def __init__(self, webcam_id, width, height, bg_dir, frame_buffer):
        super().__init__()
        self._run_flag = True
        self.display_height = height
        self.display_width = width
        self.ori_dir = bg_dir
        self.bg_dir = None
        self.webcam_id = webcam_id
        # The original background image.
        self.bg_image = cv2.resize(cv2.imread(self.ori_dir), (self.display_width, self.display_height))

        # self.probe = ffmpeg.probe(self.webcam_id)
        # self.cap_info = next(x for x in self.probe['streams'] if x['codec_type'] == 'video')
        # # print("fps: {}".format(self.cap_info['r_frame_rate']))
        # self.video_width = self.cap_info['width']           # 获取视频流的宽度
        # self.video_height = self.cap_info['height']         # 获取视频流的高度
        # self.frame_buffer = frame_buffer

        self.video_width = 1280
        self.video_height = 720
        self.frame_buffer = frame_buffer
    def bg_check(self):
        """check the if the button was pressed. if pressed, transfer the background image dir."""
        if self.bg_dir != None and self.bg_dir != self.ori_dir:    # 仅仅当不为空且和上一次背景路径不相同的时候，输出。
            self.bg_image = cv2.resize(cv2.imread(self.bg_dir), (self.display_width, self.display_height))
            self.ori_dir = self.bg_dir
            print("==== Reloading the background. ====")

    def run(self):
        while True:
            self.bg_check()
            time_1 = time.time()
            frame = self.frame_buffer.get()
            frame = np.frombuffer(frame, dtype=np.uint8)
            cv_img = frame.reshape((self.video_height, self.video_width, 3))
            """Return the mesh image. And add the mesh image on the background image."""
            cv_img = cv2.resize(cv_img, (self.display_width, self.display_height))
            outputs = romp_model(cv_img)
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w

            #### 处理背景图像 ###
            #### 首先要进行抠图，
            cv_img_mesh = cv2.cvtColor(cv2.resize(outputs['rendered_image'],(self.display_width, self.display_height)), cv2.COLOR_BGR2RGB)
            mesh_grey =  cv2.cvtColor(cv_img_mesh, cv2.COLOR_BGR2GRAY)              # mesh部分为白色，背景为黑色。
            ret, mesh_mask = cv2.threshold(mesh_grey, 1, 255, cv2.THRESH_BINARY)    # mesh部分为白色，背景为黑色
            maskInv = cv2.bitwise_not(mesh_mask)

            mesh_bg = cv2.bitwise_and(self.bg_image, self.bg_image, mask = maskInv)             # 掩码.
            mesh_fg = cv2.bitwise_and(cv_img_mesh, cv_img_mesh, mask = mesh_mask)               

            combine_mesh = cv2.add(mesh_bg, mesh_fg)
            p_ori_convert = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            p_mesh_convert = QtGui.QImage(combine_mesh.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            p_ori = p_ori_convert.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
            p_mesh = p_mesh_convert.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(QPixmap.fromImage(p_ori), QPixmap.fromImage(p_mesh))
            time_2 = time.time()
            print('run fps: ' + str(1 / (time_2 - time_1)))

    
    @pyqtSlot(str)
    def accept(self, bg_dir):
        self.bg_dir = bg_dir


class VideoThread(QThread):
    """读取需要处理的Video."""
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self, webcam_id, width, height):
        super().__init__()
        self._run_flag = True
        self.webcam_id = webcam_id
        self.display_height = height
        self.display_width = width

    def run(self):
        cap = cv2.VideoCapture(self.webcam_id)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(cv2.resize(cv_img, (self.display_width, self.display_height)), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(QPixmap.fromImage(p))
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class App(QWidget):
    main_signal = pyqtSignal(str)
    video_choose_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Human Digitalization")

        # video ids.
        self.webcam_ids = [
            "rtsp://admin:1.2.3.4.5@192.168.1.4/Streaming/Channels/101",
            "rtsp://admin:1.2.3.4.5@192.168.1.11/Streaming/Channels/101",
            "rtsp://admin:1.2.3.4.5@192.168.1.15/Streaming/Channels/101",
            "rtsp://admin:1.2.3.4.5@192.168.1.16/Streaming/Channels/101"
        ]

        # the size of showing window.
        self.display_width = 960
        self.display_height = 720

        self.display_width_small = int(self.display_width * 0.9) // 4
        self.display_height_small = int(self.display_height * 0.9) // 4

        ## background的路径地址
        self.bg_img_dirs_list = ['./bgs/cp_1.jpg', './bgs/cp_10.jpg', './bgs/cp_2.jpg', './bgs/cp_3.jpg', './bgs/cp_4.jpg', './bgs/cp_5.jpg', 
        './bgs/cp_6.jpg', './bgs/cp_7.jpg', './bgs/cp_8.jpg', './bgs/cp_9.jpg', './bgs/ct_1.jpg', './bgs/ct_10.jpg', './bgs/ct_2.jpg', 
        './bgs/ct_3.jpg', './bgs/ct_4.jpg', './bgs/ct_5.jpg', './bgs/ct_6.jpg', './bgs/ct_7.jpg', './bgs/ct_8.jpg', './bgs/ct_9.jpg', 
        './bgs/sf_1.jpg', './bgs/sf_10.jpg', './bgs/sf_2.jpg', './bgs/sf_3.jpg', './bgs/sf_4.jpg', './bgs/sf_5.jpg', './bgs/sf_6.jpg', 
        './bgs/sf_7.jpg', './bgs/sf_8.jpg', './bgs/sf_9.jpg', './bgs/st_1.jpg', './bgs/st_10.jpg', './bgs/st_2.jpg', './bgs/st_3.jpg', 
        './bgs/st_4.jpg', './bgs/st_5.jpg', './bgs/st_6.jpg', './bgs/st_7.jpg', './bgs/st_8.jpg', './bgs/st_9.jpg']
        
        self.bg_img_dirs = {
            'Cartoon':[x for x in self.bg_img_dirs_list[10:20]],
            'Science_Fiction':[x for x in self.bg_img_dirs_list[20:30]],
            'Steampunk': [x for x in self.bg_img_dirs_list[30:40]],
            'Cyberpunk': [x for x in self.bg_img_dirs_list[:10]]}
        
        self.bg_modify_dir = './bg_1.jpg'
        
        # TODO: 增加自适应图像label的功能。
        self.ori_webcam = QLabel(self)        # Original Video Stream.
        self.ori_webcam.resize(self.display_width, self.display_height)   # 左边是原始的视频，右边是处理后的视频。
        # self.ori_webcam.setScaledContents(True)

        # Background + Mesh Image.
        self.video_mesh = QLabel(self)
        self.video_mesh.resize(self.display_width, self.display_height)
        # self.video_mesh.setScaledContents(True)

        self.video_1 = QLabel(self)
        self.video_1.resize(self.display_width_small, self.display_height_small)

        self.video_2 = QLabel(self)
        self.video_2.resize(self.display_width_small, self.display_height_small)

        self.video_3 = QLabel(self)
        self.video_3.resize(self.display_width_small, self.display_height_small)

        self.video_4 = QLabel(self)
        self.video_4.resize(self.display_width_small, self.display_height_small)

        # Background Choose Button.
        self.bg1 = QPushButton("Scene Transfer:    Cartoon     ", self)
        self.bg2 = QPushButton("Scene Transfer: Science_Fiction", self)
        self.bg3 = QPushButton("Scene Transfer:    Steampunk   ", self)
        self.bg4 = QPushButton("Scene Transfer:    Cyperpunk   ", self)
        # button.setGeometry(200, 150, 100, 40)
        self.bg1.resize(self.display_width_small, self.display_height_small // 2)
        self.bg2.resize(self.display_width_small, self.display_height_small // 2)
        self.bg3.resize(self.display_width_small, self.display_height_small // 2)
        self.bg4.resize(self.display_width_small, self.display_height_small // 2)

        self.video_choose_1 = QPushButton("Camera - I", self)
        self.video_choose_2 = QPushButton("Camera - II", self)
        self.video_choose_3 = QPushButton("Camera - III", self)
        self.video_choose_4 = QPushButton("Camera - IV", self)

        # TODO: Button Style Setting.
        self.bg1.setIcon(QtGui.QIcon(self.bg_img_dirs['Cartoon'][0]))   # 添加路径
        self.bg1.setIconSize(QSize(70, 50))
        self.bg1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        self.bg2.setIcon(QtGui.QIcon(self.bg_img_dirs['Science_Fiction'][0]))   # 添加路径
        self.bg2.setIconSize(QSize(70, 50))
        self.bg2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        self.bg3.setIcon(QtGui.QIcon(self.bg_img_dirs['Steampunk'][0]))   # 添加路径
        self.bg3.setIconSize(QSize(70, 50))
        self.bg3.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        

        self.bg4.setIcon(QtGui.QIcon(self.bg_img_dirs['Cyberpunk'][0]))   # 添加路径
        self.bg4.setIconSize(QSize(70, 50))
        self.bg4.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        # self.video_choose_1.setIcon(QtGui.QIcon(self.bg_img_dirs['Cyberpunk'][0]))   # 添加路径
        # self.video_choose_1.setIconSize(QSize(50, 30))
        self.video_choose_1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        self.video_choose_2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        self.video_choose_3.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        self.video_choose_4.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        

        self.video_choose_1.resize(self.display_width_small, self.display_height_small // 2)
        self.video_choose_2.resize(self.display_width_small, self.display_height_small // 2)
        self.video_choose_3.resize(self.display_width_small, self.display_height_small // 2)
        self.video_choose_4.resize(self.display_width_small, self.display_height_small // 2)

        # 触发相应的背景选择，class中设置一个函数，然后选择不同的按钮，随意赋值一个背景图像。
        self.bg1.clicked.connect(self.background_bg1_change)
        self.bg2.clicked.connect(self.background_bg2_change)
        self.bg3.clicked.connect(self.background_bg3_change)
        self.bg4.clicked.connect(self.background_bg4_change)

        self.video_choose_1.clicked.connect(self.video_1_emit)
        self.video_choose_2.clicked.connect(self.video_2_emit)
        self.video_choose_3.clicked.connect(self.video_3_emit)
        self.video_choose_4.clicked.connect(self.video_4_emit)

        layout = QGridLayout()

        layout.addWidget(self.ori_webcam, 0, 0, 4, 4)
        layout.addWidget(self.video_mesh, 0, 4, 4, 4)

        layout.addWidget(self.video_1, 4, 0, 2, 1)
        layout.addWidget(self.video_2, 4, 1, 2, 1)
        layout.addWidget(self.video_3, 4, 2, 2, 1)
        layout.addWidget(self.video_4, 4, 3, 2, 1)

        layout.addWidget(self.bg1, 4, 4, 2, 2)
        layout.addWidget(self.bg2, 4, 6, 2, 2)
        layout.addWidget(self.bg3, 5, 4, 2, 2)
        layout.addWidget(self.bg4, 5, 6, 2, 2)

        layout.addWidget(self.video_choose_1, 6, 0, 1, 1)
        layout.addWidget(self.video_choose_2, 6, 1, 1, 1)
        layout.addWidget(self.video_choose_3, 6, 2, 1, 1)
        layout.addWidget(self.video_choose_4, 6, 3, 1, 1)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        self.setLayout(layout)
        self.frame_buffer = Queue(maxsize=5)

        self.rtsp_reader = RTSPReader(self.webcam_ids[0], self.frame_buffer)

        self.vpt = VideoProcessThread(self.webcam_ids[0], self.display_width, self.display_height, self.bg_modify_dir, self.frame_buffer)
        self.vpt.change_pixmap_signal.connect(self.update_image_mesh)
        self.main_signal.connect(self.vpt.accept)
        self.video_choose_signal.connect(self.rtsp_reader.url_update)
        self.rtsp_reader.start()
        self.vpt.start()

        ### Others.
        self.thread_1 = VideoThread(self.webcam_ids[-1], self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_1.change_pixmap_signal.connect(self.update_image_1)
        self.thread_1.start()

        self.thread_2 = VideoThread(self.webcam_ids[1],self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_2.change_pixmap_signal.connect(self.update_image_2)
        self.thread_2.start()

        self.thread_3 = VideoThread(self.webcam_ids[2],self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_3.change_pixmap_signal.connect(self.update_image_3)
        self.thread_3.start()

        self.thread_4 = VideoThread(self.webcam_ids[3],self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_4.change_pixmap_signal.connect(self.update_image_4)
        self.thread_4.start()


    @pyqtSlot(QPixmap, QPixmap)
    def update_image_mesh(self, cv_img_ori, cv_img_mesh):
        self.ori_webcam.setPixmap(cv_img_ori)
        self.video_mesh.setPixmap(cv_img_mesh)
        QApplication.processEvents()

    @pyqtSlot(QPixmap)
    def update_image_1(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.video_1.setPixmap(cv_img)

    @pyqtSlot(QPixmap)
    def update_image_2(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.video_2.setPixmap(cv_img)

    @pyqtSlot(QPixmap)
    def update_image_3(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.video_3.setPixmap(cv_img)

    @pyqtSlot(QPixmap)
    def update_image_4(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.video_4.setPixmap(cv_img)

    def background_bg1_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs['Cartoon']))

    def background_bg2_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs['Science_Fiction']))
    
    def background_bg3_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs['Steampunk']))

    def background_bg4_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs['Cyberpunk']))

    def video_1_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.video_choose_signal.emit(self.webcam_ids[0])
    
    def video_2_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.video_choose_signal.emit(self.webcam_ids[1])
    
    def video_3_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.video_choose_signal.emit(self.webcam_ids[2])
    
    def video_4_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.video_choose_signal.emit(self.webcam_ids[3])

if __name__=="__main__":
    settings = romp.main.default_settings
    settings.calc_smpl = True
    settings.render_mesh_only = True
    settings.render_mesh_bg = True
    settings.bg = 'bg_1.jpg'
    romp_model = romp.ROMP(settings)

    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())