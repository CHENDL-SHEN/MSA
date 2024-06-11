

# Weakly Supervised Semantic Segmentation Via Multi-Type Semantic Affinity Learning

This repository is the official implementation of "Weakly Supervised Semantic Segmentation Via Multi-Type Semantic Affinity Learning".

## Abstract
Weakly supervised semantic segmentation (WSSS) methods usually employ class activation maps (CAMs) to obtain pixel-level pseudo annotations for alleviating the human-labor burden. However, existing methods fail to address the problem of sparse localization in CAMs, which leads to inaccurate pseudo annotations. This sparsity problem is mainly caused by the lack of local details, the neglect of semantic context, and the absence of pixel-level supervision. To overcome the above challenges, a novel multi-type semantic affinity learning” method is proposed to infer CAMs with more complete object regions compared to the conventional generation process. Specifically, we propose a Dynamic Convolutional Reconstruction(DCR) module, designed to embed pixel-to-superpixel semantic affinity within the classification network, which enhances the capability of network to capture spatial details, thereby generating low-resolution CAMs enriched with fine local details. We then employ pixel-to-superpixel semantic affinity-based up-sampling scheme to generate full-resolution CAMs as the final localization cues while preserving spatial coherence in object regions. Moreover, we propose an Affinity Propagation (AP) module, developed to infuse pixel-to-pixel semantic affinity into the low-resolution CAMs, which improves the accuracy of object localizations by better leveraging semantic context information. In addition, we propose an Intra-Patch Prototype Contrast(IPC) loss, employed to exploit pixel-to-prototype semantic affinity to reconstruct input image, which allows for pixel-level supervision of CAMs by minimizing the distance between the reconstructed image and original input image, thus activating more accurate object regions. Extensive experiments conducted on the PASCAL VOC 2012 and MS COCO 2014 datasets demonstrate that our proposed method achieves state-of-the-art performance. 

## Prerequisite
- Python 3.6, PyTorch 1.8.0, and more in requirements.txt
- CUDA 11.1
- 2 x  RTX 3090 GPUs

## Usage

### 1. Install python dependencies
```bash
python3 -m pip install -r requirements.txt
```
### 2. Download Dataset & Pre-trained Models
- PASCAL VOC 2012
    - [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
    - [Saliency maps](https://drive.google.com/file/d/1Za0qNuIwG64-eteuz5SMbWVFL6udsLWd/view?usp=sharing) 
      using [PoolNet]
    - [Cls Labels](https://drive.google.com/file/d/1b5wfoIbmUFKKaiVQNYIURFZcZkJFpf5T/view) 

- MS-COCO 2014
    - [Images](https://cocodataset.org/#home) 
    - [Saliency maps](https://drive.google.com/file/d/1amJWDeLOj567JQMGGsSyqi7-g65dWxr0/view?usp=sharing)  using [PoolNet] 
    - [Segmentation masks](https://drive.google.com/file/d/16wuPinx0rdIP_PO0uYeCn9rfX2-evc-S/view?usp=sharing)
    - [Cls Labels](https://drive.google.com/file/d/18jtoeizNMS8cWt6PDXFFxglq7SZzgDNG/view) 

- Pre-trained Models
    - [Superpixel Segmentation Network](https://drive.google.com/drive/folders/1BVQDIhaxCD6iB8XsycwDJL7gL6Xl_t1w?usp=sharing) 
    - [Classification Network](https://download.pytorch.org/models/resnet50-19c8e375.pth)

### 3. Generate Pixel-level Pseudo Annotations  
- Step1. Set the datasets root at ```dataset_root.py```.
    ```python
    # \dataset_root.py.
    VOC_ROOT='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    VOC_SAL_ROOT='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/saliency_map/'
    COCO_ROOT='../COCO/'                                        
    COCO_SAL_ROOT='../COCO/saliency_maps_poolnet/'
- Step2. Set the "Cls_labels" at "../data/voc/" or "../data/coco/" 

- Step3. Train superpixel segmentation network.
    ```python
    # Please see these files for the detail of execution.

    python train_scn.py
- Step4. Train Classification Network at ```train_cls_image.py``` or  ```train_cls_sal.py``` . 
    ```python
    # Input images provide supervision information for IPC loss.
    python train_cls_image.py
    
    # Saliency maps provide supervision information for IPC loss.
    python train_cls_sal.py
- Step5. Infer class activation maps(CAMs) at ```evaluator_cls_image.py``` or  ```evaluator_cls_sal.py``` . 
    ```python
    python evaluator_cls_image.py
    
    python evaluator_cls_sal.py
### 4. Train Semantic Segmentation network
- We utilize [DeepLab-V2](https://arxiv.org/abs/1606.00915) 
  for the segmentation network. 
- Please see [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) for the implementation in PyTorch.

