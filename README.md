# UniPCB

### PCB\_control

☕️ : Training

You should first download the pretrained weights of Stable Diffusion and put it to PCB\_control/ckpt/ folder. Then, you can get the initial weights for training by:

python utils/prepare\_weights.py init\_local ckpt/v1-5-pruned.ckpt configs/local\_v15.yaml ckpt/init\_local.ckpt

The 4 arguments are mode, pretrained SD weights, model configs and output path for the initial weights.



To prepare the training data, please ensure that they are placed in the ./data/ folder and organized in the following manner:

data/

├── anno.txt

├── images/

├── conditions/

    ├── condition-1/

    ├── condition-2/

    ...

...

Specifically, you have to put the original images into PCB\_control/data/images/ folder and the extracted conditions into Pcb\_control/data/conditions/condition-N/ folder. And PCB\_control/data/anno.txt is the annotation file, where each line represents a training sample and is divided into two parts: 1) file ID and 2) annotation. Please ensure the consistency between the file IDs in ./data/anno.txt， ./data/images/ and ./data/conditions/condition-N/ directories.



Now, you can train with you own data simply by:

python src/train/train.py

Kindly note that the local adapter and global adapter must be trained separately. Additionally, you can customize the training configurations in PCB\_control/src/train/train.py and PCB\_control/configs/.



### PCB\_detect

📌 Usage Instructions (Training \& Validation)

This repository provides the Python/PyTorch implementation of my customized modules and model variants based on the Ultralytics YOLO framework.
All training, validation, and inference workflows follow the official Ultralytics documentation.

🚀 Training \& Validation

Since this project is built on top of Ultralytics, users can directly follow the official guides:

Ultralytics Documentation: https://docs.ultralytics.com



Training Guide: https://docs.ultralytics.com/modes/train/

Trainning Example: yolo task=detect mode=train model="path/to/rtdetr-IRSA-CAMF.yaml" data="path/to/dataset.yaml" device=0,1 pretrained=False imgsz=512



Inference Guide: https://docs.ultralytics.com/modes/predict/

Inference Example: yolo predict imgsz=512 model="path/to/rtdetr\_irsa\_camf.pt" source="path/to/images/" device=1



🛠 Installation

python=3.9

pytorch=2.40



📁 Model Configuration

The custom model configuration files are located in:

PCB\_detect/ultralytics/cfg/models/rt-detr

Simply reference the corresponding YAML file during training.

