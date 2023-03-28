import os
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
import argparse


from predict import *

# options
parser = argparse.ArgumentParser()
parser.add_argument('--language_model', type=str, default='opt', help='gpt2/opt')
parser.add_argument('--prefix_length', type=int, default=32, help='must match prefix_length of your trained model')
parser.add_argument('--device', default='03')
args = parser.parse_args()

def make_device(args):
    device_num = len(args.device)
    devices = []
    for i in range(device_num):
        device = "cuda:" + args.device[i]
        devices.append(torch.device(device))
    return devices

device1, device2 = make_device(args)

# file path : CVPR2023challenge 
fpath_nice = os.path.join('/data/img_cap/nice', 'NICE_val')
flist_nice = os.listdir(fpath_nice)
annot_csv = pd.read_csv(os.path.join('/data/img_cap/nice', 'nice-val-5k.csv'))
output_file = f'./output_caption/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(output_file, exist_ok=True)

# Setup predictor
predict = Predictor()
predict.setup(language_model=args.language_model, prefix_length=args.prefix_length, device1=device1, device2=device2)
print('Ready to predict captions of CVPR2023-NICE dataset')

# example
image = os.path.join(fpath_nice, flist_nice[0])
generated_caption_coco_2 = predict.predict(image=image, model='coco', use_beam_search=True)
print("Exammple Caption :", generated_caption_coco_2)

# start generating captions
data_coco_2 = {}
data_coco_beam = {}
for img_nice in tqdm(flist_nice):
    image = os.path.join(fpath_nice, img_nice)
    
    generated_caption_coco_2 = predict.predict(image=image, model='coco', use_beam_search=False)
    generated_caption_coco_beam = predict.predict(image=image, model='coco', use_beam_search=True)
    
    target_caption = annot_csv[annot_csv['public_id']==int(img_nice[:-4])]['caption_gt'].item()
    
    data_coco_2[int(img_nice[:-4])] = [target_caption, generated_caption_coco_2]
    data_coco_beam[int(img_nice[:-4])] = [target_caption, generated_caption_coco_beam]
    
# save generated caption
with open(os.path.join(output_file, f'clipcap_2_opt13b_{args.language_model}.json'), 'w') as fp:
    json.dump(data_coco_2, fp)
with open(os.path.join(output_file, f'clipcap_beam_opt13b_{args.language_model}.json'), 'w') as fp:
    json.dump(data_coco_beam, fp)
