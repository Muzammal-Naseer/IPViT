# Intriguing Properties of Vision Transformers (NeurIPS'21--Spotlight)

[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Kanchana Ranasinghe](https://scholar.google.com/citations?user=K2WBZTwAAAAJ),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en),
[Munawar Hayat](https://scholar.google.ch/citations?user=Mx8MbWYAAAAJ&hl=en&oi=ao), 
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao), &
[Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)

[Paper](https://openreview.net/pdf?id=o2mbl-Hmfgd) ([arxiv](https://arxiv.org/abs/2105.10497)), [Reviews & Response](https://openreview.net/forum?id=o2mbl-Hmfgd), [Video Presentation](https://neurips.cc/virtual/2021/poster/27802), [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202021/c404a5adbf90e09631678b13b05d9d7a_ZnmODYn.png)

> **Abstract:** 
>*Vision transformers (ViT) have demonstrated impressive performance across various machine vision tasks. These models are based on multi-head self-attention mechanisms that can flexibly attend to a sequence of image patches to encode contextual cues. An important question is how such flexibility (in attending image-wide context conditioned on a given patch) can facilitate handling nuisances in natural images e.g., severe occlusions, domain shifts, spatial permutations, adversarial and natural perturbations. We systematically study this question via an extensive set of experiments encompassing three ViT families and provide comparisons with a high-performing convolutional neural network (CNN). We show and analyze the following intriguing properties of ViT: (a) Transformers are highly robust to severe occlusions, perturbations and domain shifts, e.g., retain as high as 60% top-1 accuracy on ImageNet even after randomly occluding 80% of the image content.  (b) The robust performance to occlusions is not due to a bias towards local textures, and ViTs are significantly less biased towards textures compared to CNNs. When properly trained to encode shape-based features, ViTs demonstrate shape recognition capability comparable to that of human visual system, previously unmatched in the literature. (c) Using ViTs to encode shape representation leads to an interesting consequence of accurate semantic segmentation without pixel-level supervision. (d) Off-the-shelf features from a single ViT model can be combined to create a feature ensemble, leading to high accuracy rates across a range of classification datasets in both traditional and few-shot learning paradigms.  We show effective features of ViTs are due to flexible and dynamic receptive fields possible via self-attention mechanisms. Our code will be publicly released.* 

![demo](.github/demo.png)


## Contents
1) [Shape Biased Models](#shape-biased-models)
2) [Off the Shelf Features](#off-the-shelf-features)
3) [Global Image Context](#global-image-context-occlusion--shuffle)
4) [Varying Patch Sizes and Positional Encoding](#varying-patch-sizes-and-positional-encoding)
5) [References](#references)
6) [Citation](#citation)

## Requirements
```bash
pip install -r requirements.txt
```


## Shape Biased Models 
<sup>([top](#contents))</sup>
Our shape biased pretrained models can be downloaded from [here](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/tag/v0). We summarise the performance of each model below.

|          Model         	| Jaccard Index 	| Pretrained 	|
|:----------------------:	|:-------------:	|:----------:	|
|         DeiT-T         	|      32.2     	|    baseline    	|
|       DeiT-T-SIN       	|      29.4     	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_t_sin.pth)    	|
| DeiT-T-SIN (distilled) 	|      40.0     	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_t_sin_dist.pth)    	|
|         DeiT-S         	|      29.2     	|    baseline    	|
|       DeiT-S-SIN       	|      37.5     	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin.pth)    	|
| DeiT-S-SIN (distilled) 	|      42.0     	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth)    	|


Code for evaluating their shape bias using auto segmentation on the PASCAL VOC dataset can be found under scripts. Please fix any paths as necessary. You may place the VOC devkit folder under `data/voc` or fix the paths appropriately. 

### Direct Implementation
Running segmentation evaluation on models (pre-trained models will be auto downloaded): 
```bash
./scripts/eval_segmentation.sh
```

Visualizing segmentation for images in a given folder (pre-trained models will be auto downloaded):
```bash
./scripts/visualize_segmentation.sh
``` 

### Additional Details 
We compare the different shape bias capacities of the class token and the shape (distillation) token for shape distilled models. As explained in `scripts/eval_segmentation.sh`, these evaluations can be carried out as follows. 

Evaluate pre-trained model using the class token (default mode):
```bash
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9
```

Evaluate pre-trained model using the shape (distillation) token:
```bash
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9 \
  --use_shape
```

Note that the threshold parameter (controlling the quantity of attention flow similar to [DINO](https://github.com/facebookresearch/dino)) can be varied to obtain different segmentations. We have selected 0.9 for optimal segmentation performance on PASCAL VOC '12. Results for different threshold settings can be visualized using the following code (adjust threshold parameter appropriately). Add `--use_shape` argument to use the shape token for generating visualizations.
```bash
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth" \
  --threshold 0.9 \
  --patch_size 16 \
  --generate_images \
  --test_dir "data/sample_images" \
  --save_path "data/sample_segmentations"

```


## Off the Shelf Features
<sup>([top](#contents))</sup>
Training code for off-the-shelf experiment in `classify_metadataset.py`. Seven datasets (aircraft, CUB, DTD, fungi, GTSRB, Places365, and INAT) available by default. Set the appropriate dir path in `classify_md.sh` by fixing `DATA_PATH`. Note that for the ResNet baselines, we adopt the PyTorch official models. All training on transfer dataset is limited to updating a final linear layer using a standard training schedule.  

[comment]: < ![off_shelf](.github/off_shelf.png) >

### Direct Implementation
Run training and evaluation for a selected dataset (aircraft by default) using selected model (DeiT-T by default):
```bash
./scripts/classify_md.sh
```

### Additional Details
Set the `DATASET` variable to one of `aircraft`, `CUB`, `DTD`, `fungi`, `GTSRB`, `Places365`, or `INAT` and model to one of `resnet50`, `deit-tiny`, or `deit-small`. Variable `EXP_NAME` can be set to any name (will be used for logging). Environment variable `DATA_PATH` should direct to the relevant dataset root directory. Note that all dataset classes are simple modifications of the standard torchvision ImageFolder class. 
```bash
python classify_metadataset.py \
  --datasets "$DATASET" \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --batch_size 256 \
  --epochs 10 \
  --project "off_the_shelf" \
  --exp "$EXP_NAME"
```


## Global Image Context (Occlusion & Shuffle)
<sup>([top](#contents))</sup>
We apply various occlusions and shuffle operations on images to explore the robust properties of ViT models. All evaluation is carried out on the ImageNet 2012 validation set. 

### Direct Implementation
For direct evaluation on ImageNet val set (change path in script) using our proposed occlusion techniques and shuffle operation run:
```bash
./scripts/evaluate_occlusion.sh
./scripts/evaluate_shuffle.sh
./scripts/evaluate_occlusion_supp.sh
```

### Additional Occlusion Details
We present three patch based occlusion methods Random, Salient, and Non-Salient PatchDrop. For all scripts, the environment variable `DATA_PATH` should direct to the ImageNet dataset validation directory. Evaluation of a pretrained model under Random PatchDrop technique can be done as below:
```bash
python evaluate.py \
  --model_name deit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --pretrained "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth" \
  --random_drop
```
Evaluation of a pretrained model under Salient & Non-Salient PatchDrop technique can be done as below. This method borrows from [DINO](https://github.com/facebookresearch/dino) to select foreground and background pixels. 
```bash
python evaluate.py \
  --model_name deit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --pretrained "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth" \
  --dino
```

We also experiment with Random PatchDrop under varied settings involving different grids sizes, at pixel level, with grid offsets, and also applying on intermediate feature maps. The defaults settings evaluates on a 14x14 grid. To evaluate on different grid sizes, run following (8x8 default - replace with desired):
 ```bash
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size 8 8 
```

Evaluate at pixel level as follows (when grid size is equivalent to image dimensions):
```bash
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size 224 224
```

Using a grid with an offset: 
```bash
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --random_offset_drop
```

Intermediate feature drop on DeiT models:
```bash
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --lesion \
  --block_index 0 2 4 8 10
```

Intermediate feature drop on ResNet50:
```bash
python evaluate.py \
  --model_name resnet_drop \
  --test_dir "$DATA_PATH" \
  --lesion \
  --block_index 1 2 3 4 5
```

Note that intermediate feature drop is dependent on network architecture. Currently, only three DeiT variants (tiny, small, base) and ResNet50 are supported.

### Additional Shuffle Details
Evaluate under the shuffle operation setting using a range of grid sizes to observe robustness to permutation invariance as follows:
```bash
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16 \

```


## Varying Patch Sizes and Positional Encoding
<sup>([top](#contents))</sup>
We present pretrained models to study the effect of varying patch sizes and positional encoding: 
| DeiT-T Model 	| Top-1 	| Top-5 	| Pretrained 	|
|:------------:	|:-----:	|:-----:	|:----------:	|
| No Pos. Enc. 	|  68.3 	|  89.0 	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/no_pos_deit_t.pth)    	|
|   Patch 22   	|  68.7 	|  89.0 	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/patch_22_deit_t.pth)    	|
|   Patch 28   	|  65.2 	|  86.7 	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/patch_28_deit_t.pth)    	|
|   Patch 32   	|  63.1 	|  85.3 	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/patch_32_deit_t.pth)    	|
|   Patch 38   	|  55.2 	|  78.8 	|    [Link](https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/patch_38_deit_t.pth)    	|


## Adversarial Patches
TBA

## Stylized ImageNet Training
We generate Stylized ImageNet (SIN) following the [SIN repository](https://github.com/rgeirhos/Stylized-ImageNet). We use the [DeiT](https://github.com/facebookresearch/deit) codebase to perform standard training of DeiT models using the SIN dataset instead of natural ImageNet to obtain SIN only models. The shape distilled models are obtained by using the DeiT distillation mechanism. Training is performed using natural ImageNet. Instead of a CNN trained on natural ImageNet, a CNN (or ViT) trained on SIN is used as the teacher model.  

<!-- 
## To Be Added
1) ~~Pretrained ViT models trained on Stylized ImageNet (along with distilled ones). We will provide code to use these models for auto-segmentation~~.
2) ~~Training and Evaluations for our proposed off-the-shelf ensemble features.~~
3) ~~Code to evaluate any model on our proposed occulusion stratagies (random, foreground and background).~~ 
4) ~~Code for evaluation of permutation invaraince.~~ 
5) ~~Pretrained models to study the effect of varying patch sizes and positional encoding.~~
6) Pretrained adversarial patches and code to evaluate them.
7) ~~Training on Stylized Imagenet.~~
-->

## References
<sup>([top](#contents))</sup>
Code borrowed from [DeiT](https://github.com/facebookresearch/deit) and [DINO](https://github.com/facebookresearch/dino) repositories as well as [TIMM](https://github.com/rwightman/pytorch-image-models) library. We thank them for their wonderful code bases. 


## Citation
If you find our work, this repository and pretrained Transformers useful. Please consider giving a star :star: and citation.
```bibtex
@inproceedings{
naseer2021intriguing,
title={Intriguing Properties of Vision Transformers},
author={Muzammal Naseer and Kanchana Ranasinghe and Salman Khan and Munawar Hayat and Fahad Khan and Ming-Hsuan Yang},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=o2mbl-Hmfgd}
}
```
