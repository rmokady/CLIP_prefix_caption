import json

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


# your file path
file_path = ## YOUR FILE PATH (.json)
# split your file with GroundTruth & Prediction
gt_file_name = f'./clipcap_opt27_gt.json'
gr_file_name = f'./clipcap_opt27_gr.json'

gt = {}; gr = {}
with open(file_path,'r') as f:
    json_data = json.load(f)

gt["annotations"] = []; gt["images"] = []
gr["annotations"] = []; gr["images"] = []
for key, value in json_data.items():
    temp_1, temp_2, temp_3 = {}, {}, {}
    temp_1["image_id"] = key; temp_2["image_id"] = key
    temp_1["caption"] = value[0]; temp_2["caption"] = value[1]
    temp_1["id"] = key; temp_2["id"] = key; temp_3["id"] = key
    gt["annotations"].append(temp_1); gt["images"].append(temp_3)
    gr["annotations"].append(temp_2); gr["images"].append(temp_3)

with open(gt_file_name, 'w') as f_gt, open(gr_file_name, 'w') as f_gr:
    json.dump(gt, f_gt)
    json.dump(gr, f_gr)


# evaluate CIDEr, SPICE, METEOR, BLEU-4, ROUGE, BLEU-3, BLEU-2, BLEU-1 score
coco_gt = COCO(gt_file_name)
coco_pred = COCO(gr_file_name)
coco_eval = COCOEvalCap(coco_gt, coco_pred)

coco_eval.evaluate()
