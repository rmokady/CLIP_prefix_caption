import argparse
import json
import os
from datetime import datetime

# import pandas as pd
from tqdm import tqdm

from predict import *

# options
parser = argparse.ArgumentParser()
parser.add_argument('--language_model', type=str, default='opt', help='gpt2/opt')
parser.add_argument('--prefix_length', type=int, default=32, help='must match prefix_length of your trained model')
parser.add_argument('--checkpoint', type=str, default='001', help='checkpoint weight path')
parser.add_argument('--ofile', type=str, default='clipcap')
parser.add_argument('--device', default='123')
parser.add_argument('--pn', default='111')
args = parser.parse_args()

# file path : CVPR2023challenge 

# fpath_nice = os.path.join('/data1/IC/nice-eval', 'images')
# flist_nice = os.listdir(fpath_nice)
# output_folder = './output_caption'
# output_file = args.ofile + '.json'
# clipcap_path_nice = os.path.abspath('/data1/IC/nice_val_features/clipcap')
# os.makedirs(output_folder, exist_ok=True)

# annot_csv = pd.read_csv(os.path.join('/data/IC/nice-eval', 'nice-val-5k.csv'))
# output_file = f'./output_caption/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
# os.makedirs(output_file, exist_ok=True)


fpath_nice_test = '/data1/IC/nice-test/images'
flist_nice_test = os.listdir(fpath_nice_test)
output_folder = './output_caption'
output_file = args.ofile + '.json'
os.makedirs(output_folder, exist_ok=True)



# folder = 'val2017'
# fpath_coco = os.path.join('/data1/IC/coco/images', folder)
# flist_coco = os.listdir(fpath_coco)
# clipcap_path_coco = os.path.join('/data1/IC/coco_features/clipcap', folder[:-4])



OPT_MODEL = 'facebook/opt-2.7b'

# Setup predictor
predict = Predictor()
predict.setup(args)
print('Ready to predict captions of CVPR2023-NICE dataset')


# start generating captions
data= {}
for img_nice in tqdm(flist_nice_test):
    image = os.path.join(fpath_nice_test, img_nice)
    
    generated_caption, _ = predict.predict(image=image, model=f'opt_{args.checkpoint}')
    
    # torch.save(prefix_embed, os.path.join(clipcap_path_nice, img_nice[:-4] + '.pt'))
    # target_caption = annot_csv[annot_csv['public_id']==int(img_nice[:-4])]['caption_gt'].item()
    
    data[int(img_nice[:-4])] = generated_caption


# data= {}
# for img_coco in tqdm(flist_coco):
#     image = os.path.join(fpath_coco, img_coco)
    
#     _, prefix_embed = predict.predict(image=image, model=f'opt_{args.checkpoint}')
    
#     torch.save(prefix_embed, os.path.join(clipcap_path_coco, img_coco[:-4] + '.pt'))



# save generated caption
# with open(os.path.join(output_file, f'clipcap_2_opt13b_{args.language_model}.json'), 'w') as fp:
#     json.dump(data_coco_2, fp)
# with open(os.path.join(output_file, f'clipcap_beam_opt13b_{args.language_model}.json'), 'w') as fp:
#     json.dump(data_coco_beam, fp)

# save generated caption
with open(os.path.join(output_folder, output_file), 'w') as fp:
    json.dump(data, fp, default=str)