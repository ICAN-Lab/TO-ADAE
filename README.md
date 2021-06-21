# TO-ADAE

UH-Sense is a UAV-based wireless sensing system capable of detecting and localizing a target in an area. The paper has been prepared to submit.

#### Novelty  
Given received signal strength (RSS) measurements of the commodity WiFi receiver at the UAV, UH-Sense can **detect** and **localize** moving human targets in the monitored area.
- A combination of neural network-based classifier and radio tomography imaging technique (RTI) is utilized to determine the presence of targets and their relative positions, respectively.

#### Task-oriented adversarial denoising autoencoder (TO-ADAE)
The novelty of the system lies in the unique combination of adversarial denoising autoencoder (ADAE) and RTI to blindly denoise the corrupted measurement data without prior knowledge of the noise characteristics.
- It achieves the a human detection rate of approximately 87\% and the localization error about 4m.

## Prerequisites

- Python 3.7
- tensorflow-gpu 1.15
- GPU
 
## Getting started
### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/download/)
- Install [CUDA 10.0](https://developer.nvidia.com/cuda-90-download-archive)
- install [cuDNN 7.6](https://developer.nvidia.com/cudnn)
- Install [tensorflow-gpu](https://www.tensorflow.org/install/gpu?hl=zh-tw)

Noted that our code is tested based on [tensorflow-gpu 1.15](https://www.tensorflow.org/install/gpu?hl=zh-tw)

### Dataset & Preparation

Before training or test, please make sure you have prepared the dataset
by the following steps:
- **Step1:** Organize the directory as: 
`your_dataset_path/conditionst /environmets`.
E.g. `dataset/humanPresence/LOS-RD/`.
E.g. `CASIA-B/humanAbsence/NLOS-RD/`.

- **Step2:** Align the dimention.
The input dimention of TO-ADAE is **72x8**.
- 72 is the number of RSS in a sample
- 8 is the number of WiFi APs

### Configuration 

In `config.py`, you might want to change the following settings:
- `WORK_PATH` path to save/load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs

### Train
Train a model by
```bash
python train.py
```

### Evaluation
Evaluate the trained model by
```bash
python test.py
```
It will output the localizztion of all three enviroments and detection accuracy in three enviroments. 


## Authors & Contributors
TO-ADE is authored by
[Wei Chen],
[YuJia Chen],
from Central University.
The code is developed by
[Wei Chen].
Currently, it is being maintained by
[Wei Chen].

