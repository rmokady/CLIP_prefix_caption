# CLIP prefix captioning.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  
Inference Notebook: <a href="https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  





## implementation for the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)




## Description  
- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734)
- [original ClipCap github](https://github.com/rmokady/CLIP_prefix_caption.git) : CLIP_prefix_caption

code references
- [transformers(OPT) github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py)
- [BLIP2](https://github.com/salesforce/BLIP.git)


## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## COCO training

Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`.

Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).

Extract CLIP features using (output is `data/coco/oscar_split_ViT-B_32_train.pkl`):
```
python parse_coco.py --clip_model_type ViT-B/32
```
Train with fine-tuning of GPT2:
```
python train.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/
```

__In case you want to train model with OPT, please look directly "Swith your language model from GPT-2 to OPT"__  
Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40
```

**If you wish to use ResNet-based CLIP:** 
https://github.com/Jhryu30/cvpr2023_challenge_clipcap.git
```
python parse_coco.py --clip_model_type RN50x4
```
```
python train.py --only_prefix --data ./data/coco/oscar_split_RN50x4_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```




## Swith your language model from GPT-2 to OPT
We enabled to train your ClipCap model with OPT. We are looking forward to make this code work well with [BLIP model](https://github.com/salesforce/BLIP.git). 
Training code is available at `train.py` and inference code will be updated on `predict_OPT.py`, which is basically running Predictor function in predict.py. 
Please note that you manullay have to make sure your desired language model is 'facebook/opt-125m' (variable named as OPT_MODEL) on both `predict.py` and `train.py`.

```
python train_OPT.py --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir /data/daisy/clipcap_output/coco_train/ --only_prefix --device
```
```
python predict_nice.py
```

### model parallelization
- OPT-1.3b : 2-GPU, 16GB (per GPU), 1h13m per epoch


## Inference Notebooks
To help visualize the results we provide a Colab notebook found in `notebooks/clip_prefix_captioning_inference.ipynb`.   
The notebook will download the pretrained models and run inference on a sample images or 
on images of your choosing. It is recommended to run this in [Google Colab](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing).
Inference notebook for the **transformer mapping network (without fine-tune GPT-2)** can be found [here](https://colab.research.google.com/drive/180L3rMFmGujudwO1EJNF-lHIpAsAZ5xq?usp=sharing) for the COCO model (also in `notebooks/transformer_inference.ipynb`).



Both [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models are available for mlp mapping network. For the transformer (without fine-tuning GPT-2) we provide [COCO](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view?usp=sharing) pretrained model.



## Inference GUI
1. Run it [in the browser](https://replicate.ai/rmokady/clip_prefix_caption) using replicate.ai UI.
2. Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/CLIP_prefix_captioning) (currently not supporting beam search)




*latest update : 2023-03-28*

## Citation
If you use this code for your research, please cite:
```
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```




## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

## Contact
For any inquiry please contact us at our email addresses: ron.mokady@gmail.com or amirhertz@mail.tau.ac.il.


