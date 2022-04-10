# Image Captioning with Attentive Capsule Network
# abstract:
Image captioning is a fundamental bridge linking computer vision and natural language processing. State-of-the-art methods mainly focus on improving the learning of image features using visual-based attention mechanisms. However, they are limited by the immutable attention parameters and cannot capture spatial relationships of salient objects in an image adequately. To fill this gap, we propose an Attentive Capsule Network (ACN) for image captioning, which can well utilize the spatial information especially positional relationships delivered in an image to generate more accurate and detailed descriptions. In particular, the proposed ACN model is composed of a channel-wise bilinear attention block and an attentive capsule block. The channel-wise bilinear attention block helps to obtain the 2nd order correlations of each feature channel; while the attentive capsule block treats region-level image features as capsules to further capture the hierarchical pose relationships via transformation matrices. To our best knowledge, this is the first work to explore the image captioning task by utilizing capsule networks. Extensive experiments show that our ACN model can achieve remarkable performance, with the competitive CIDEr performance of 133.7% on the MS-COCO Karpathy test split.

# Data preparation：
* Download the [bottom up features](https://github.com/peteanderson80/bottom-up-attention) and convert them to npz files
```
python tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_36
```
* Download the [annotations](https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS) into the mscoco folder.
* Download [coco-caption](https://github.com/ruotianluo/coco-caption) and setup the path of __C.INFERENCE.COCO_PATH in lib/config.py


# Training：
### Train ACN model
```
bash experiments/acn/train.sh
```
### Train ACN model using self critical
Copy the pretrained model from experiments/acn/snapshot into experiments/acn_rl/snapshot and run the script
```
bash experiments/acn_rl/train.sh
```
