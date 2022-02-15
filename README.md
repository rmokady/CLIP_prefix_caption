# CLIP prefix captioning.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  
Inference Notebook: <a href="https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  





## Official implementation for the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)




## Description  
Image captioning is a complicated task, where usually a pretrained detection network is used, requires additional supervision in the form of object annotation. We present a new approach that does not requires additional information (i.e. requires only images and captions), thus can be applied to any data. In addition, our model's training time is much faster than similar methods while achieving comparable to state-of-the-art results, even for the Conceptual Captions dataset contains over 3M images. 

In our work, we use the [CLIP](https://github.com/openai/CLIP) model, which was already trained over an extremely large number of images, thus is capable of generating semantic encodings for arbitrary images without additional supervision. To produce meaningful sentences we fine-tune a pretrained language model, which has been proven to be successful for other natural language tasks. The key idea is to use the CLIP encoding as a prefix to the textual captions by employing a simple mapping network over the raw encoding, and then fine-tune our language model to generate a valid caption. In addition, we present another variant, where we utilize a transformer architecture for the mapping network and avoid the fine-tuning of GPT-2. Still, our light model achieve comaparable to state-of-the-art over nocaps dataset.

## COCO Examples

<table>
  <tr>
    <td><img src="Images/COCO_val2014_000000562207.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000165547.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000579664.jpg" ></td>
  </tr>
  <tr>
    <td>A couple of people standing next to an elephant. </td>
     <td>A wooden table sitting in front of a window.</td>
     <td>A bunch of bananas sitting on top of a table.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/COCO_val2014_000000060623.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000386164.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000354533.jpg" ></td>
  </tr>
  <tr>
    <td>A woman holding a plate with a piece of cake in front of her face. </td>
     <td>A wooden table topped with lots of wooden utensils.</td>
     <td>A red motorcycle parked on top of a dirt field.</td>
  </tr>
 </table>


## Conceptual Captions Examples

<table>
  <tr>
    <td><img src="Images/CONCEPTUAL_01.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_02.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_03.jpg" ></td>
  </tr>
  <tr>
    <td>3D render of a man holding a globe.</td>
     <td>Students enjoing the cherry blossoms</td>
     <td>Green leaf of lettuce on a white plate.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/CONCEPTUAL_04.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_05.jpg" ></td>
    <td><img src="Images/CONCEPTUAL_06.jpg" ></td>
  </tr>
  <tr>
    <td>The hotel and casino on the waterfront. </td>
     <td>The triangle is a symbol of the soul.</td>
     <td>Cartoon boy in the bath.</td>
  </tr>
 </table>


## Inference Notebooks
To help visualize the results we provide a Colab notebook found in `notebooks/clip_prefix_captioning_inference.ipynb`.   
The notebook will download the pretrained models and run inference on a sample images or 
on images of your choosing. It is recommended to run this in [Google Colab](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing).
Inference notebook for the **transformer mapping network (without fine-tune GPT-2)** can be found [here](https://colab.research.google.com/drive/180L3rMFmGujudwO1EJNF-lHIpAsAZ5xq?usp=sharing) for the COCO model (also in `notebooks/transformer_inference.ipynb`).



Both [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models are available for mlp mapping network. For the transformer (without fine-tuning GPT-2) we provide [COCO](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view?usp=sharing) pretrained model.



## Inference GUI
1. Run it [in the browser](https://replicate.ai/rmokady/clip_prefix_caption) using replicate.ai UI.
2. Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/CLIP_prefix_captioning) (currently not supporting beam search)


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

Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40
```

**If you wish to use ResNet-based CLIP:** 

```
python parse_coco.py --clip_model_type RN50x4
```
```
python train.py --only_prefix --data ./data/coco/oscar_split_RN50x4_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layres 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```

## Conceptual training

Download the .TSV train/val files from [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download) and place them under <data_root> directory.

Download the images and extract CLIP features using (outputs are `<data_root>/conceptual_clip_ViT-B_32_train.pkl` and  `<data_root>/conceptual_clip_ViT-B_32_val.pkl`):
```
python parse_conceptual.py --clip_model_type ViT-B/32 --data_root <data_root> --num_threads 16
```
Notice, downloading the images might take a few days.

Train with fine-tuning of GPT2:
```
python train.py --data <data_root>/conceptual_clip_ViT-B_32_train.pkl --out_dir ./conceptual_train/
```
Similarly to the COCO training, you can train a transformer mapping network, and / or parse the images using a ResNet-based CLIP. 

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


