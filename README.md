# CDL with PyTorch

This is the **PyTorch** implementation of "Credible Dual-Experts for Weakly Supervised Semantic Segmentation (IJCV 2023)"
* **MS COCO** dataset and **PASCAL VOC 2012** dataset are supported.
* This project heavily rely on kazuto1011's work, for more details, please see the [link](https://github.com/kazuto1011/deeplab-pytorch)
* Many thanks for kazuto1011's great work!
* For the code of Res2Net, please email me: [Bingfeng.Z@outlook.com](Bingfeng.Z@outlook.com) as the original backbone code is from other's private code. 
* For any problem related to this project, please email me: [Bingfeng.Z@outlook.com](Bingfeng.Z@outlook.com), thanks.



## Setup


Please follow the "Setup" part in the [link](https://github.com/kazuto1011/deeplab-pytorch) to prepare your environment




## Datasets

```

### Download datasets

* Download PASCAL VOC 2012 Aug from this [link](https://drive.google.com/file/d/1jhtdjj3xrEp60zO3B7jZ14yxxZkCJMeM/view)
* Download MS COCO from this [link](https://cocodataset.org/#download)
* Download the GT mask of MS COCO from this [link](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view)
* All used [pseudo labels] on PASCAL VOC 2012 can be found in this [link](https://drive.google.com/file/d/1jeYdx3oCcsfmjiPkeIPVXU0GH2HGqhxH/view?usp=sharing)
* All used [pseudo labels] on MS COCO can be found in this [link](https://drive.google.com/file/d/1xxQjfQZeqSXvjTtEkAVfh-5AxQsqorrA/view?usp=sharing)


For PASCAL VOC, During training, please put the used [pseudo label] folder into path/to/VOCdevkit/VOC2012, and rename the folder name as "SegmenatationCLassAug"
For COCO 2014, During training, please put the used [pseudo label] folder into path/to/COCO2014, and rename rename the folder name as "coco_seg_anno"

During tesing or inferencing, please use the original "SegmenatationClassAug" and "coco_seg_anno" which are GTs.

The final structure of the dataset is:
Dataset
    ├── VOCdevkit/VOC2012
    |   |── ImageSets
    |   |── JPEGImages
    |   |── SegmentationClass
    |   |── SegmentationClassAug (pseudo labels or GT)     
    ├── coco             
    │   ├── coco_seg_anno (pseudo labels or GT)
    |   └── JPEGImages        

```



## Init models
```
coco-pretrain-resnet101 init weights can be download in this [link](https://drive.google.com/file/d/196TkMJPL3GA3CaXxKJrtsgX0CcXvWqsn/view?usp=sharing)
ImageNet-pretrain-resnet101 init weights can be download in this [link](https://drive.google.com/file/d/14soMKDnIZ_crXQTlol9sNHVPozcQQpMn/view)
```


## Training & Evaluation

To train DeepLab v2 on PASCAL VOC 2012 with CDL-b/e and coco init weights:

```sh
python main_CDL_b/e.py train \
    --config-path configs/voc12_coco_pretrain.yaml
```


To train DeepLab v2 on PASCAL VOC 2012 with CDL-b/e and imagenet init weights:

```sh
python main_CDL_b/e.py train \
    --config-path configs/voc12_imageNet_pretrain.yaml
```



To train DeepLab v2 on MS COCO with CDL-b/e and imagenet init weights:

```sh
python main_CDL_b/e.py train \
    --config-path configs/coco.yaml

```

To evaluate the performance on a validation set with CDL-b/e:

```sh
python main_CDL_b/e.py test \
    --config-path configs/voc12_coco_pretrain.yaml \
    --model-path data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
```

Note: This command saves the predicted logit maps (`.npy`) and the scores (`.json`).

To re-evaluate with a CRF post-processing:<br>

```sh
python main_CDL_b/e.py crf \
    --config-path configs/voc12_coco_pretrain.yaml
```

For CDL-e or ImageNet-pretrain, folloing above operations and change the corresponding name to 
"main_CDL-e" or "voc12_ImageNet_pretrain.yaml" or both of them.

To monitor a loss, run the following command in a separate terminal.

```sh
tensorboard --logdir data/logs
```



Common settings:

- **Model**: DeepLab v2 with ResNet-101 backbone. Dilated rates of ASPP are (6, 12, 18, 24). Output stride is 8.
- **GPU**: All the GPUs visible to the process are used. Please specify the scope with
```CUDA_VISIBLE_DEVICES=```.
- **Multi-scale loss**: Loss is defined as a sum of responses from multi-scale inputs (1x, 0.75x, 0.5x) and element-wise max across the scales. The *unlabeled* class is ignored in the loss computation.
- **Gradient accumulation**: The mini-batch of 10 samples is not processed at once due to the high occupancy of GPU memories. Instead, gradients of small batches of 5 samples are accumulated for 2 iterations, and weight updating is performed at the end (```batch_size * iter_size = 10```). GPU memory usage is approx. 11.2 GB with the default setting (tested on the single Titan X). You can reduce it with a small ```batch_size```.
- **Learning rate**: Stochastic gradient descent (SGD) is used with momentum of 0.9 and initial learning rate of 2.5e-4. Polynomial learning rate decay is employed; the learning rate is multiplied by ```(1-iter/iter_max)**power``` at every 10 iterations.
- **Monitoring**: Moving average loss (```average_loss``` in Caffe) can be monitored in TensorBoard.
- **Preprocessing**: Input images are randomly re-scaled by factors ranging from 0.5 to 1.5, padded if needed, and randomly cropped to 321x321.



## Dataset and Deeplab References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

1. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)


# Pseudo labels referenes:
1. [A2GNN] Zhang B, Xiao J, Jiao J, et al. Affinity attention graph neural network for weakly supervised semantic segmentation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(11): 8082-8096. 

2. [EPS] Lee S, Lee M, Lee J, et al. Railroad is not a train: Saliency as pseudo-pixel supervision for weakly supervised semantic segmentation[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 5495-5505.

3. [RIB] Lee J, Choi J, Mok J, et al. Reducing information bottleneck for weakly supervised semantic segmentation[J]. Advances in Neural Information Processing Systems, 2021, 34: 27408-27421.

4. [VML] Ru L, Du B, Zhan Y, et al. Weakly-supervised semantic segmentation with visual words learning and hybrid pooling[J]. International Journal of Computer Vision, 2022, 130(4): 1127-1144.