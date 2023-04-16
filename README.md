# Introduction
The is a simple, but a real-time demo. We mainly use the PyQt5 to conduct an interactive interface, which shows some original video streams captured from different camera ids and people's mesh image added on different style's backgrounds. When the different style's button is pressed, these background images could be changed.  

**Key components**
- Two big windows, one is used for capturing video from a camera and the other is used for showing the processed video stream.   
- Eight small windows showing different videos from different monitors.   
- Four buttons that modifying background image styles, e.g., Cyberpunk, Cartoon, Steampunk and Science_Fiction.   

**Key Tech**  
- FFMPEG: Getting video stream in a real-time manner. 
- ROMP: processing one frame and return a mesh image. 

# Performance

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

- If there are any new background images added into the showing, please remove the bgs.json file firstly, and the main code will generate this file again.

# Version Recording
Several different version is seperately saved with different file name for convient showing. Here recoading the file name only and do not upload the corresponding files.  
- [ x ] visualization_0414.py: add a reset button for the mesh QThread.
- [ x ] visualization_button_layout.py: re-arrange the buttons' layout and add the button of changing webcam id.
- [ x ] Sovling the delay accumulation and stop problem immediately.
- [ x ] Add more background images for different cameras. Firstly we generate a json file according to the background images.
- ～～[ x ] Re-arrange the botton layout.～～
- [ x ] Add more images for showing on the README.md. 
- [  ] Solving the delay problem of switching different cameras.
- [  ] Refine the layout of buttons.



# Reference
Thanks to these below reporitories.   
[ROMP](https://github.com/Arthur151/ROMP)  
[ffmpeg-python](https://github.com/kkroening/ffmpeg-python)


# Authors
If you have any questions about this demo, please contact us.  
**Ai Wenjie**   *awenji10@gmail.com*  
**Li Yanchao** 