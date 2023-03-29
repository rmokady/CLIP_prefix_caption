import argparse
import json
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from predict import *

# options
parser = argparse.ArgumentParser()
parser.add_argument('--language_model', type=str, default='opt', help='gpt2/opt')
parser.add_argument('--prefix_length', type=int, default=32, help='must match prefix_length of your trained model')
parser.add_argument('--device', default='12')
parser.add_argument('--pn', default='47')
args = parser.parse_args()

# file path : CVPR2023challenge 
fpath_nice = os.path.join('/data/img_cap/nice', 'NICE_val')
flist_nice = os.listdir(fpath_nice)
annot_csv = pd.read_csv(os.path.join('/data/img_cap/nice', 'nice-val-5k.csv'))
output_file = f'./output_caption/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(output_file, exist_ok=True)

# Setup predictor
predict = Predictor()
predict.setup(args)
print('Ready to predict captions of CVPR2023-NICE dataset')

# example
for i in [0, 1, 2, 3]:
    print(i)
    image = os.path.join(fpath_nice, flist_nice[i])

    image = io.imread(image)
    model = predict.models['coco']; tokenizer = predict.tokenizer
    pil_image = PIL.Image.fromarray(image)
    image = predict.preprocess(pil_image).unsqueeze(0).to(predict.device1)
    with torch.no_grad():
        prefix = predict.clip_model.encode_image(image).to(
            predict.device1, dtype=torch.float32
        )
        prefix_embed = model.clip_project(prefix).reshape(1, predict.prefix_length, -1)

    use_nucleus_sampling=False 
    num_beams=5
    max_length=30
    min_length=1
    top_p=0.9
    repetition_penalty=1.5
    length_penalty=1.0
    num_captions=1
    temperature=1  

    atts_opt = torch.ones(prefix_embed.size()[:-1], dtype=torch.long).to(predict.device1)
    opt_tokens = tokenizer([""], return_tensors='pt').to(predict.device1)
    input_ids = opt_tokens.input_ids
    query_embeds = prefix_embed #.repeat_interleave(num_beams, dim=0)
    attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

    outputs = model.gpt.generate(
                    input_ids=input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    eos_token_id=tokenizer('\n', add_special_tokens=False).input_ids[0],
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

    prompt_length = input_ids.shape[1]
    output_text = tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
    output_text = [text.strip() for text in output_text]
    print(output_text)

# generated_caption_coco_2 = predict.predict(image=image, model='coco', use_beam_search=True)
# print("Exammple Caption :", generated_caption_coco_2)

# # start generating captions
# data_coco_2 = {}
# data_coco_beam = {}
# for img_nice in tqdm(flist_nice):
#     image = os.path.join(fpath_nice, img_nice)
    
#     generated_caption_coco_2 = predict.predict(image=image, model='coco', use_beam_search=False)
#     generated_caption_coco_beam = predict.predict(image=image, model='coco', use_beam_search=True)
    
#     target_caption = annot_csv[annot_csv['public_id']==int(img_nice[:-4])]['caption_gt'].item()
    
#     data_coco_2[int(img_nice[:-4])] = [target_caption, generated_caption_coco_2]
#     data_coco_beam[int(img_nice[:-4])] = [target_caption, generated_caption_coco_beam]
    
# # save generated caption
# with open(os.path.join(output_file, f'clipcap_2_opt13b_{args.language_model}.json'), 'w') as fp:
#     json.dump(data_coco_2, fp)
# with open(os.path.join(output_file, f'clipcap_beam_opt13b_{args.language_model}.json'), 'w') as fp:
#     json.dump(data_coco_beam, fp)
