# **Cross-Modal Fusion and Progressive Decoding Network For RGB-D Salient Object Detection**

The paper was accepted by the International Journal of Computer Vision on January 11, 2024. The paper link is: - [Link](https://link.springer.com/article/10.1007/s11263-024-02020-y#citeas)

## CPNet
Most existing RGB-D salient object detection (SOD) methods tend to achieve higher performance by integrating additional modules, such as feature enhancement and edge generation. There is no doubt that these modules will inevitably produce feature redundancy and performance degradation. To this end, we exquisitely design a crossmodal fusion and progressive decoding network to achieve RGB-D SOD tasks. The designed network structure only includes three indispensable parts: feature encoding, feature fusion and feature decoding. Specifically, in the feature encoding part, we adopt a two-stream Swin Transformer encoder to extract multi-level and multi-scale features from RGB images and depth images respectively to model global information. In the feature fusion part, we design a cross-modal attention fusion module, which can leverage the attention mechanism to fuse multi-modality and multi-level features. In the feature decoding part, we design a progressive decoder to gradually fuse low-level features and filter noise information to accurately predict salient objects. Extensive experimental results on 6 benchmarks demonstrated that our network surpasses 12 state-of-the-art methods in terms of four metrics. In addition, it is also verified that for the RGB-D SOD task, the addition of the feature enhancement module and the edge generation module is not conducive to improving the detection performance under this framework, which provides new insights into the salient object detection task. Our codes will be available at https://github.com/hu-xh/CPNet.

## Network Architecture
![fig1.png](figs/fig1.jpg)

## Results and Saliency maps
We perform quantitative comparisons and qualitative comparisons with 12 RGB-D SOD
methods on six RGB-D datasets.
![fig2.jpg](figs/fig2.jpg)
![fig3.jpg](figs/fig3.jpg)

### Prerequisites
- Python 3.6
- Pytorch 1.10.2
- Torchvision 0.11.3
- Numpy 1.19.2

### Pretrained Model
Download the following `pth` and put it into main folder
- [Swin-B](https://pan.baidu.com/s/1VkWOrdrw3RHOp0Ir5rLGgw) with the fetch code:ja95.

### Datasets
- [Train Datasets](https://pan.baidu.com/s/148IZcZAB5qSSWBJYzhvoYw) with the fetch code:1234.
- [Test Datasets](https://pan.baidu.com/s/18dbNDpkV7hV43UOW7v8huA) with the fetch code:1234.

### Results
You can download the tested results map at - [Baidu Pan link] (https://pan.baidu.com/s/1PlmqAvlAwSzsH2YGR4VzKQ) with the fetch code:dq2w.

### Contact
Feel free to send e-mails to me (1558239392@qq.com).
