# FIOLA
An accelerated pipeline for Fluorescence Imaging OnLine Analysis.

FIOLA exploits computational graphs and accelerated hardware to preprocess fluorescence imaging movies to extract neural activity at speeds in excess of 300Hz on standard fields of view for fluorescence imaging movies.

![fiola overview image](/fiola_overview.png)

## Requirements
FIOLA was tested on Ubuntu 18.04 operating systems. Before installation, one needs to install anaconda, cuda 11.0 (for tensorflow 2.4.1) and cudnn beforehand.

## Installation guide
It takes 5 mins to run the installation on a normal desktop computer:

```
git clone https://github.com/nel-lab/FIOLA.git
git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13
cd FIOLA
conda create --name fiola python==3.8
conda activate fiola
pip install -r requirements.txt 
pip install -e.
cd ../CaImAn
pip install -e . 
```

## Demo
It takes 5 mins to run the demo on a normal desktop computer. The output is online extracted of fluorescnece traces for neurons from the demo movie.

Python demo: demo_fiola_pipeline.ipynb 

Google colab demo can be found at the following link: https://colab.research.google.com/drive/1y98SHqjAqalJ0LXvVF2drjtVdH8tzMa2?usp=sharing
