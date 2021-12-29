# FIOLA
An accelerated pipeline for Fluorescence Imaging OnLine Analysis. 

FIOLA exploits computational graphs and accelerated hardware to preprocess fluorescence imaging movies to extract neural activity at speeds in excess of 300Hz on standard fields of view for calcium imaging and at over 400Hz for voltage imaging movies.

## Requirements
Tested on Ubuntu 18.04 operating systems. Software dependencies can be found in requirements.txt file. Tested with python 3.8, tensorflow 2.4.1. Need GPU to run the pipeline.

## Installation guide
It takes less than 3 mins to run the following code for installation on a normal desktop computers:

git clone https://github.com/nel-lab/FIOLA.git

cd FIOLA

### Pip installation
conda create --name fiola python==3.8
conda activate fiola
pip install -r requirements.txt
pip install -e.

### mamba installation (suggested)
In your base environment install mamba
``` conda install mamba ```

then proceed to install fiola

``` 
mamba env create -f environment.yml -n fiola 
conda activate fiola
pip install -e . 
```

## Demo
It takes less than 5 mins to run the demo on a normal desktop computer

Google colab demo can be found at the following link: https://colab.research.google.com/drive/1yKoyi1Fz9bzNtOhrjuC8h12_WzWHYIvz?usp=sharing

Python demo: demo_fiola_pipeline.py. 

The output is online extracted of traces from the demo movie

## License
COPYRIGHT AND PERMISSION NOTICE
UNC Software FIOLA
Copyright (C) 2021 The University of North Carolina at Chapel Hill
All rights reserved. 

The University of North Carolina at Chapel Hill (“UNC”) and the developers of the FIOLA software (“Software”) give recipient (“Recipient”) permission to download a single copy of the Software in executable form and use for non-commercial purposes only provided that the following conditions are met:
1. Recipient may use the Software for any purpose, EXCEPT for commercial benefit.
2. Recipient will not copy the Software.
3. Recipient will not sell the Software.
4. Recipient will not give the Software to any third party.
5. Any party desiring a license to use the Software for commercial purposes shall contact:
The Office of Technology Commercialization at UNC at otc@unc.edu or 919-966-3929.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS, CONTRIBUTORS, AND THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER, CONTRIBUTORS OR THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
