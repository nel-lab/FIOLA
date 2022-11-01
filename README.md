# FIOLA
An accelerated pipeline for Fluorescence Imaging OnLine Analysis. 

FIOLA exploits computational graphs and accelerated hardware to preprocess fluorescence imaging movies to extract neural activity at speeds in excess of 300Hz on standard fields of view for calcium imaging and at over 400Hz for voltage imaging movies.

## Requirements
Tested on Ubuntu 18.04 operating systems. Software dependencies can be found in requirements.txt file. Tested with python 3.8, tensorflow 2.4.1. Need GPU to run the pipeline.

## Installation guide
It takes 5 mins to run the installation on a normal desktop computers:

```
git clone https://github.com/nel-lab/FIOLA.git
git clone https://github.com/flatironinstitute/CaImAn.git
cd FIOLA
```

### Pip installation (Recommended)
```
conda create --name fiola python==3.8
conda activate fiola
pip install -r requirements_with_caiman.txt
pip install -e.
cd CaImAn
pip install -e . 
conda install spyder
pip install jinja2==3.0.1
```

### mamba installation (Outdated)
In your base environment install mamba
``` conda install mamba ```

then proceed to install fiola

``` 
mamba env create -f environment.yml -n fiola 
conda activate fiola
pip install -e . 
```

## Demo
It takes 5 mins to run the demo on a normal desktop computer

Google colab demo can be found at the following link: https://colab.research.google.com/drive/1yKoyi1Fz9bzNtOhrjuC8h12_WzWHYIvz?usp=sharing

Python demo: demo_fiola_pipeline.py. 

The output is online extracted of traces from the demo movie

