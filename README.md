# ACN
# paper title:Image Captioning with Attentive Capsule Network
# paper abstract:
In image captioning, though state-of-the-art methods have well advocated the use of visual attention mechanisms for more fine-grained image understanding, less attention has been paid to the position relationships of salient regions in an image. With
the aim of filling this gap, we customize the Attentive Capsule Network (ACN) for image captioning, which maximizes the information delivered in the images to generate more accurate and detailed descriptions. The ACN model contains a channelwise bilinear attention block and an attentive capsule block, which respectively capture the channel
and spatial relationships of region-level image features. Meanwhile, the model mitigates the drawback of traditional visual encoders by improving the quality of extracted image features with positional information. Experiments show the effectiveness of our ACN model over the compared advanced methods, with the best CIDEr performance
of 133.7% on MS-COCO Karpathy test split announced so far. This paper innovatively and targeted investigates the capsule network to image captioning. The improved quality of generated captions proves the positive significance of the proposed ACN model on image captioning.

# Data preparation：
* 1.Download the bottom up features（https://github.com/peteanderson80/bottom-up-attention） and convert them to npz files
python tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_36
* 2.Download the annotations（https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS） into the mscoco folder.
* 3.Download coco-caption（https://github.com/ruotianluo/coco-caption） and setup the path of __C.INFERENCE.COCO_PATH in lib/config.py


# Training：
Train ACN model
bash experiments/acn/train.sh
