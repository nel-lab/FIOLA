# FIOLA
An accelerated pipeline for Fluorescence Imaging OnLine Analysis.

FIOLA exploits computational graphs and accelerated hardware to preprocess fluorescence imaging movies to extract neural activity at speeds in excess of 300Hz on standard fields of view for fluorescence imaging movies.

![fiola overview image](/fiola_overview.png)

## Requirements
Tested on Ubuntu 18.04 operating systems. Software dependencies can be found in requirements_with_caiman.txt file. FIOLA is mainly tested with python 3.8, tensorflow 2.4.1. It is also compatible with tensorflow 2.5.0. Need GPU to run the pipeline.

## Installation guide
It takes 5 mins to run the installation on a normal desktop computers:

```
git clone https://github.com/nel-lab/FIOLA.git
git clone https://github.com/flatironinstitute/CaImAn.git
cd FIOLA
```

### Pip installation
```
conda create --name fiola python==3.8
conda activate fiola
pip install -r requirements_with_caiman.txt
pip install -e.
cd ../CaImAn
pip install -e . 
pip install h5py==2.10.0
```


## Demo
It takes 5 mins to run the demo on a normal desktop computer. Python demo: demo_fiola_pipeline.py. 

Google colab demo can be found at the following link: https://colab.research.google.com/drive/1VSc4yvsLfRRcSgjDzRB4jCK9cmEgrWMu?usp=sharing

The output is online extracted of traces from the demo movie

