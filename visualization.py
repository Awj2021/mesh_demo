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
from collections import defaultdict
import json

############################ 使用ffmpeg的方式处理视频 #######################################
class RTSPReader(QThread):
    # url_signal = pyqtSignal(str)
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
        # self.url_flag = False

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

            time_2 = time.time()
            print('RTSPReader fps' + str(1 / (time_2 - time_1)))
    
    # @pyqtSlot(str)
    # def url_update(self, url_):
    #     if url_ != self.rtsp_url:
    #         self.rtsp_url = url_
    #         self.ffmpeg_cmd = [
    #             "ffmpeg",
    #             "-rtsp_transport", "tcp",
    #             "-fflags", "nobuffer", 
    #             "-flags", "low_delay",
    #             "-i", self.rtsp_url, 
    #             "-f", "rawvideo",
    #             "-pix_fmt", "rgb24",
    #             "-y", "pipe:1"
    #         ]
    #         self.ffmpeg_process = sp.Popen(self.ffmpeg_cmd, stdout=sp.PIPE)
    #         self.url_flag = True


class VideoProcessThread(QThread):
    """读取需要处理的Video."""
    change_pixmap_signal = pyqtSignal(QPixmap, QPixmap)
    bg_singal = pyqtSignal(str)
    camera_signal = pyqtSignal(str)

    def __init__(self, width, height, bg_dir, frame_buffer_4, frame_buffer_8, frame_buffer_11, frame_buffer_16):
        super().__init__()
        self._run_flag = True
        self.display_height = height
        self.display_width = width
        self.ori_dir = bg_dir
        self.bg_dir = None

        self.bg_image = cv2.resize(cv2.imread(self.ori_dir), (self.display_width, self.display_height))

        self.video_width = 1280
        self.video_height = 720
        
        self.frame_buffer_4 = frame_buffer_4
        self.frame_buffer_8 = frame_buffer_8
        self.frame_buffer_11 = frame_buffer_11
        self.frame_buffer_16 = frame_buffer_16
        self.default_frame_buffer = frame_buffer_4

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
            frame = self.default_frame_buffer.get()
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

    @pyqtSlot(str)
    def camera_shift(self, buffer_No):
        exec('self.default_frame_buffer = self.frame_buffer_{}'.format(buffer_No))  

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
        self.setWindowTitle("Metaverse 3D Human Digitalization")

        # the size of showing window.
        self.display_width = 960
        self.display_height = 720

        self.display_width_small = int(self.display_width * 0.95) // 4
        self.display_height_small = int(self.display_height * 0.95) // 4

        self.video_numbers = [str(x) for x in [4, 8, 11, 16]]
        self.default_webcam_id_No = self.video_numbers[0]
        # TODO: please check the camera_id & No.
        self.webcam_ids = {
            "4": "rtsp://admin:1.2.3.4.5@192.168.1.4/Streaming/Channels/101",
            "8": "rtsp://admin:1.2.3.4.5@192.168.1.8/Streaming/Channels/101",
            "11": "rtsp://admin:1.2.3.4.5@192.168.1.11/Streaming/Channels/101",
            "16": "rtsp://admin:1.2.3.4.5@192.168.1.16/Streaming/Channels/101"
        }

        self.bg_img_dirs = bg_dirs_from_json(json_dir)

        # print(self.bg_img_dirs)
        self.bg_modify_dir = './bg_3.jpg'
        
        # TODO: 增加自适应图像label的功能。
        self.ori_webcam = QLabel(self)        # Original Video Stream.
        self.ori_webcam.resize(self.display_width, self.display_height)   # 左边是原始的视频，右边是处理后的视频。

        # Background + Mesh Image.
        self.video_mesh = QLabel(self)
        self.video_mesh.resize(self.display_width, self.display_height)

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

        self.video_choose_1 = QPushButton("Camera - I", self)
        self.video_choose_2 = QPushButton("Camera - II", self)
        self.video_choose_3 = QPushButton("Camera - III", self)
        self.video_choose_4 = QPushButton("Camera - IV", self)

        # TODO: Button Style Setting.
        self.bg1.setIcon(QtGui.QIcon(self.bg_img_dirs['4']['Cartoon'][0]))   # 添加路径
        self.bg1.setIconSize(QSize(70, 50))
        self.bg1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        self.bg2.setIcon(QtGui.QIcon(self.bg_img_dirs['4']['Science_Fiction'][0]))   # 添加路径
        self.bg2.setIconSize(QSize(70, 50))
        self.bg2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        self.bg3.setIcon(QtGui.QIcon(self.bg_img_dirs['4']['Steampunk'][0]))   # 添加路径
        self.bg3.setIconSize(QSize(70, 50))
        self.bg3.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        self.bg4.setIcon(QtGui.QIcon(self.bg_img_dirs['4']['Cyberpunk'][0]))   # 添加路径
        self.bg4.setIconSize(QSize(70, 50))
        self.bg4.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

        self.video_choose_1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        self.video_choose_2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        self.video_choose_3.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')
        self.video_choose_4.setStyleSheet('QPushButton {background-color: #A3C1DA; color: blue;}')

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

        layout.addWidget(self.bg1, 4, 4, 1, 2)
        layout.addWidget(self.bg2, 4, 6, 1, 2)
        layout.addWidget(self.bg3, 5, 4, 1, 2)
        layout.addWidget(self.bg4, 5, 6, 1, 2)

        layout.addWidget(self.video_choose_1, 6, 0, 1, 1)
        layout.addWidget(self.video_choose_2, 6, 1, 1, 1)
        layout.addWidget(self.video_choose_3, 6, 2, 1, 1)
        layout.addWidget(self.video_choose_4, 6, 3, 1, 1)
        # layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        self.setLayout(layout)
        self.frame_buffer_4 = Queue(maxsize=5)
        self.frame_buffer_8 = Queue(maxsize=5)
        self.frame_buffer_11 = Queue(maxsize=5)
        self.frame_buffer_16 = Queue(maxsize=5)

        self.rtsp_reader_4 = RTSPReader(self.webcam_ids["4"], self.frame_buffer_4)
        self.rtsp_reader_8 = RTSPReader(self.webcam_ids["8"], self.frame_buffer_8)
        self.rtsp_reader_11 = RTSPReader(self.webcam_ids["11"], self.frame_buffer_11)
        self.rtsp_reader_16 = RTSPReader(self.webcam_ids["16"], self.frame_buffer_16)

        self.vpt = VideoProcessThread(self.display_width, self.display_height, self.bg_modify_dir,  \
                    self.frame_buffer_4, self.frame_buffer_8,self.frame_buffer_11,self.frame_buffer_16)   
        self.vpt.change_pixmap_signal.connect(self.update_image_mesh)
        self.main_signal.connect(self.vpt.accept)
        self.video_choose_signal.connect(self.vpt.camera_shift)
        self.rtsp_reader_4.start()
        self.rtsp_reader_8.start()
        self.rtsp_reader_11.start()
        self.rtsp_reader_16.start()
        self.vpt.start()

        ### Others.
        self.thread_1 = VideoThread(self.webcam_ids["4"], self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_1.change_pixmap_signal.connect(self.update_image_1)
        self.thread_1.start()

        self.thread_2 = VideoThread(self.webcam_ids["8"],self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_2.change_pixmap_signal.connect(self.update_image_2)
        self.thread_2.start()

        self.thread_3 = VideoThread(self.webcam_ids["11"],self.display_width_small, self.display_height_small)         # 应该传入IP. list
        self.thread_3.change_pixmap_signal.connect(self.update_image_3)
        self.thread_3.start()

        self.thread_4 = VideoThread(self.webcam_ids["16"],self.display_width_small, self.display_height_small)         # 应该传入IP. list
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
        self.main_signal.emit(random.choice(self.bg_img_dirs[self.default_webcam_id_No]['Cartoon']))

    def background_bg2_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs[self.default_webcam_id_No]['Science_Fiction']))
    
    def background_bg3_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs[self.default_webcam_id_No]['Steampunk']))

    def background_bg4_change(self):
        """由button跳转, 选择不同的background"""
        self.main_signal.emit(random.choice(self.bg_img_dirs[self.default_webcam_id_No]['Cyberpunk']))

    def video_1_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.default_webcam_id_No = self.video_numbers[0]
        # self.default_webcam_id = self.webcam_ids[self.default_webcam_id_No]
        self.video_choose_signal.emit(self.default_webcam_id_No)
    
    def video_2_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.default_webcam_id_No = self.video_numbers[1]
        # self.default_webcam_id = self.webcam_ids[self.default_webcam_id_No]
        self.video_choose_signal.emit(self.default_webcam_id_No)
    
    def video_3_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.default_webcam_id_No = self.video_numbers[2]
        # self.default_webcam_id = self.webcam_ids[self.default_webcam_id_No]
        self.video_choose_signal.emit(self.default_webcam_id_No)
    
    def video_4_emit(self):
        """由button跳转，选择不同的视频展示"""
        self.default_webcam_id_No = self.video_numbers[3]
        # self.default_webcam_id = self.webcam_ids[self.default_webcam_id_No]
        self.video_choose_signal.emit(self.default_webcam_id_No)


def bg_dirs_from_json(json_dir):
    """
    Transfer all these bg dirs into json file. Json file directly save to the current folder. 
    If the json file has already exists, directly upload these json file.

    Dict structure:
    {
        id_No1: 
            {
                "cartoon": [image1, image2, ..., ],    # image is the absolute path.
                "Scientific_Fiction": []
                ...
            }
        id_No2: 
            {
                ...
            }
        ...
    }
    """
    assert type(json_dir) == str
    if not os.path.exists(json_dir):
        bgs_dict = defaultdict(dict)
        ids_list = [str(x) for x in [4, 8, 11, 16]]
        for id_folder in ids_list:
            images = os.listdir(os.path.join(bg_dir, id_folder))
            bgs_dict[id_folder] = {
                'Cartoon':[os.path.join(bg_dir, id_folder, x) for x in images if x.startswith('ct')],
                'Science_Fiction':[os.path.join(bg_dir, id_folder, x) for x in images if x.startswith('sf')],
                'Steampunk': [os.path.join(bg_dir, id_folder, x) for x in images if x.startswith('sp')],
                'Cyberpunk': [os.path.join(bg_dir, id_folder, x) for x in images if x.startswith('cp')]
                }
        bgs_json = json.dumps(bgs_dict)
        with open(json_dir, 'w') as outfile:
            outfile.write(bgs_json)
        return bgs_json
        
    else:
        print("*"*60)
        print("Json file of background images has already exists!!!")
        print("*"*60)

        with open(json_dir, 'r') as json_file:
            bgs_json = json.load(json_file)
        
        return bgs_json


if __name__=="__main__":
    settings = romp.main.default_settings
    settings.calc_smpl = True
    settings.render_mesh_only = True
    settings.render_mesh_bg = True
    settings.bg = 'bg_1.jpg'
    romp_model = romp.ROMP(settings)

    bg_dir = '/home/yanchao/Desktop/bgs_v2'
    json_dir = '/home/yanchao/Desktop/mesh_demo/bgs.json'

    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())