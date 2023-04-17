# Introduction
~~The is a simple, but a real-time demo. We mainly use the PyQt5 to conduct an interactive interface, which shows some original video streams captured from different camera ids and people's mesh image added on different style's backgrounds. When the different style's button is pressed, these background images could be changed.~~  

This demo is Metaverse 3d human digitalization, we achieved realtime multiperson 3d mesh recovery on single camera.  3d human digitalization is fundamental structure for metaverse application. e.g. virtual reality meeting or class. Existing method required multi-camera system and/or motion capture device, which is money cost. Our group research focus on efficient 3d mesh recovery from 2d, without acquire camera parameters. 

Beyond that, we also add 5g to our pipeline to achieve real-time remote 3d mesh reconstruct ability. We add the 5g to our video stream collection device, thus our sever can access to the video stream through larger bandwith low latency 5g network remotely. when we work anywhere, we just need to carry out camera and collection device.

And we also add the background virtualization to the convert the realistic background to several styles.

Welcome to our groundbreaking demo of Metaverse 3D human digitalization! We have achieved real-time, multi-person 3D mesh recovery using just a single camera, making 3D human digitalization a more accessible and cost-effective solution for various Metaverse applications, such as virtual reality meetings or classes.

Traditionally, 3D mesh recovery has required expensive multi-camera systems and/or motion capture devices. Our research group has revolutionized this process by focusing on efficient 3D mesh recovery from 2D images. With our approach, there is no need for camera calibration, making it user-friendly and hassle-free. All the necessary 3D information for mesh reconstruction is self-estimated by our advanced AI model, streamlining the process and ensuring accuracy.

Furthermore, we have incorporated 5G technology into our pipeline to enable real-time remote 3D mesh reconstruction. By integrating 5G into our video stream collection device, our server can access the video stream remotely through a high-bandwidth, low-latency 5G network. This means that you only need a camera and our collection device to create 3D digitalizations from anywhere you go.

Additionally, we have implemented background virtualization to convert the realistic background into various styles, adding a touch of customization to your Metaverse experience. Join us and witness the power of our cutting-edge Metaverse 3D human digitalization technology!

**Key components**
- Two big windows, one is used for capturing video from a camera and the other is used for showing the processed video stream.   
- Eight small windows showing different videos from different monitors.   
- Four buttons that modifying background image styles, e.g., Cyberpunk, Cartoon, Steampunk and Science_Fiction.   

**Key Tech**  
- FFMPEG: Getting video stream in a real-time manner. 
- ROMP: processing one frame and return a mesh image. 
- PyQt6: Interact.

# Performance
A video of performance.
![demo_video](./show_video/demo_video.gif)

# Environment  
Reproduce the environment: (ONLY using the environment.yml file.)  
```
conda env create -f environment.yml
conda activate py310
```

Install ROMP please refer to the original reporitory.  
https://github.com/Arthur151/ROMP
We strongly suggest that installing **ROMP** with source code, i.e., by `make *.sh`.

Please make sure these necessory packages should be installed: (including in the environment.yml)
```
PyQt6
cv2
ffmpeg
imutils
romp
```

# Running
The lattest code is visualization.py file.
```
python visualization.py
```
If you want to run the older versions of code, you could try to run the visualization_bak files, e.g., `visualization_bak405.py`. But I cannot guarantee that these codes will work correctly.

- **If there are any new background images added into the showing, please remove the bgs.json file firstly, and the main code will generate this file again.**

By the way, we could directly run the shell file: `run.sh`.
```
bash ./run.sh
```

# Version Recording
Several different version is seperately saved with different file name for convient showing. Here recoading the file name only and do not upload the corresponding files.  
- [ x ] visualization_0414.py: add a reset button for the mesh QThread.
- [ x ] visualization_button_layout.py: re-arrange the buttons' layout and add the button of changing webcam id.
- [ x ] Sovling the delay accumulation and stop problem immediately.
- [ x ] Add more background images for different cameras. Firstly we generate a json file according to the background images.
- ~~[ x ] Re-arrange the botton layout.~~
- [ x ] Add more images for showing on the README.md. 
- [ x ] Solving the delay problem of switching different cameras.
- [  ] Refine the layout of buttons.
- [ x ] Regenerate the background images.
- [ x ] Refine the **Introduction**, combine the command in the shell file.


# Reference
Thanks to these below reporitories.   
[ROMP](https://github.com/Arthur151/ROMP)  
[ffmpeg-python](https://github.com/kkroening/ffmpeg-python)


# Authors
If you have any questions about this demo, please contact us.  
**Ai Wenjie**   *awenji10@gmail.com*  
**Li Yanchao** 