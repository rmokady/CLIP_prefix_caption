# CLIP prefix captioning.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  
Inference Notebook: <a href="https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  


## Description  
Image captioning used to be a very complicated tasks, usually a pretrained detection network is used, requires additional supervision in the form of object annotation. The features of the detected objects are then fed to an additional network that is trained to output the correct caption. We present a new approach which do not requires additional information (i.e. requires only images and captions), thus can be applied to any data. In addition, our model's training time is much faster then similar methods while achieving close to state-of-the-art results.

In our work, we use the [CLIP](https://github.com/openai/CLIP) model, which was already trained over an extremely large number of images, thus is capable of generating semantic encodings for arbitrary images without additional supervision. To produce meaningful sentences we fine-tune a pretrained language model, as been proven to be successful for other natural language tasks. The key idea is to use the CLIP encoding as a prefix to the textual captions by employing a simple Multi Layer Perceptron (MLP) over the raw encoding, and then fine-tune our language model to generate a valid caption.

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





## Inference Notebooks
To help visualize the results  we provide a Colab notebook found in `notebooks/clip_prefix_captioning_inference.ipynb`.   
The notebook will download the pretrained models and run inference on a sample images or 
on images of your choosing. It is recommended to run this in [Google Colab](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing).


## Acknowledgments
This project was created by Ron Mokady and Amir Hertz Advanced-NLP course by Omer Levy @ TAU.
This repository is heavily based on [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home).
The project was also inspired from [this paper](https://arxiv.org/abs/2101.00190).

## Contact
For any inquiry please contact us at our email addresses: ron.mokady@gmail.com or amirhertz@mail.tau.ac.il.


