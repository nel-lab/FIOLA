# FIOLA
An accelerated pipeline for Fluorescence Imaging OnLine Analysis. 

FIOLA exploits computational graphs and accelerated hardware to preprocess fluorescence imaging movies to extract neural activity at speeds in excess of 300Hz on standard fields of view for calcium imaging and at over 400Hz for voltage imaging movies.

## Requirements
Tested on Ubuntu 18.04 operating systems. Software dependencies can be found in requirements_simple.txt file. Tested with python 3.8, tensorflow 2.4.1. Need GPU to run the pipeline.

## Installation guide
It takes less than 3 mins to run the following code for installation on a normal desktop computers:

git clone https://github.com/nel-lab/FIOLA.git

cd FIOLA

conda create --name fiola python==3.8

conda activate fiola

pip install -r requirements_simple.txt

conda install spyder

pip install -e.

## Demo
Google colab demo can be found with the following link: https://colab.research.google.com/drive/1yKoyi1Fz9bzNtOhrjuC8h12_WzWHYIvz?usp=sharing

Data can be found at the following link: https://www.dropbox.com/sh/1bhi3l5onoykjya/AACBchSTajeZIrRIGtE2-hqXa?dl=0

Python demo: demo_fiola_pipeline.py

It takes less than 5 mins to run the demo on a normal desktop computer


