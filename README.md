
## [Installation and train DerainNet](https://xueyangfu.github.io/projects/tip2017.html) 


## Installation
Run the script below to install the environment (Conda):
```
conda create -n derainnet python=3.7
conda activate derainnet
conda install -y numpy=1.19.5 tensorflow-gpu=1.15 matplotlib
pip install scikit-image==0.17.2

```

## Datasets

To train the models, please download the synthetic datasets (Rain100L, Rain100H, Rain1400) and and place the unzipped folders into the project folder:
https://drive.google.com/file/d/1PES0IFPQ24MxpP1_6f_TvSBw4U8KdK_E/view?usp=sharing


## Getting Started

### 1) Testing
Run shell scripts to test the models:
```bash
bash test.sh
```
You can modify the arguments in the `test.sh` or `testing.py`

### 2) Training

Run shell scripts to train the models:
```bash
bash train.sh      
```
The models are saved into `./model/` folder by default
You can modify the arguments in the `training.sh`, `training.py` or `network.py`



