#!/bin/bash
set -e
# conda activate py310
cd /home/lyc/Desktop/mesh_demo/ROMP/simple_romp
python setup.py install 

cd /home/lyc/Desktop/mesh_demo

python visualization.py
