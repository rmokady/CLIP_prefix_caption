import torch, os, json
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from modeling_opt_pp import OPTForCausalLM
from collections import OrderedDict


"""
FeatureDataset = 
Feauture = Dataloader( Feature)
optimizer = AdamW
lr = 

"""

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
            
class Hprod(nn.Module):
    def __init__(self, num_feature, num_query_token, query_dimension, device):
        self.weights = nn.Parameter(torch.Tensor(1, num_feature, num_query_token, query_dimension)).to(device)    

    def forward(self, features):
        prod = features * self.weights
        output = torch.sum(prod, dim=1, keepdim=True)
        return output
    
OPT_MODEL = 'facebook/opt-2.7b'
MODEL = {
    "cnn" : nn.Conv2d,
    "hprod" : Hprod,
    "mix_qdim" : nn.Linear,
    }

class ConnectLayer(nn.Module):
    def __init__(
        self,
        connect_model_key = "cnn",
        num_feature = 3,
        num_query_token = 32,
        query_dimension = 2560,
        batch_size = 16,
        prompt="a photo of",
        max_txt_len=32,
        device = "123",
        pn = "411",
    ):
        super().__init__()

        self.model_key = connect_model_key
        assert self.model_key in MODEL.keys()
        self.device1, device2, device3, pn1, pn2 = make_device_pn(device=device, pn=pn)
        assert pn1>0

        if self.model_key == "cnn":
            self.connect_model = MODEL[self.model_key](
                in_channels=2,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                device=self.device1
            )
        elif self.model_key == "hprod":
            self.connect_model = MODEL[self.model_key](
                num_feature=num_feature,
                num_query_token=num_query_token,
                query_dimension=query_dimension,
                device=self.device1
            )
        elif self.model_key == "mix_qdim":
            self.connect_model = MODEL[self.model_key](
                in_feature=num_query_token * query_dimension,
                out_features=query_dimension,
                bias=False,
                device=self.device1
            )

        self.opt_model = OPTForCausalLM.from_pretrained(OPT_MODEL, torch_dtype=torch.float16)
        self.opt_tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL, use_fast=False)

        self.opt_model.model.decoder.setting_device(device1=self.device1, device2=device2, device3=device3, pn1=pn1, pn2=pn2)

        for _, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        self.batch_size = batch_size
        self.num_query_token = num_query_token
        self.query_dimension = query_dimension
        self.max_txt_len = max_txt_len

    def forward(self, samples):
        features = samples["features"]
        features = features.to(self.device1)
        if self.model_key == "cnn":
            query_embeds = self.connect_model(features)
        elif self.model_key == "hprod":
            query_embeds = self.connect_model(features)
        elif self.model_key == "mix_qdim":
            features = features.permute(1, 2).view(
                self.batch_size,
                self.num_query_token,
                self.num_query_token * self.query_dimension
            )
            query_embeds = self.connect_model(features).unsqueeze(1)
        
        atts_opt = torch.ones(query_embeds.size()[:-1], dtype=torch.long).to(features.device)

        self.opt_tokenizer.padding_side = "right"
        
        text = [samples["text_input"] + "\n"]
        # text = [t + "\n" for t in [samples["text_input"]]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(features.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(features.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([query_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        # loss = outputs.loss

        return outputs.loss #{"loss": loss}


class CaptionDataset_WithFeature(BaseDataset, __DisplMixin):
    def __init__(self, feature_paths, ann_paths, vis_processor=None, text_processor=None, vis_root=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # ann_paths = ['/data1/IC/coco/annotations/coco_karpathy_train.json']
        self.feature_paths = feature_paths
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        _, jpg_file = os.path.split(ann['image'])
        file_name, _ = os.path.splitext(jpg_file)
        pt_file = file_name + '.pt'

        # feature_paths = ['/data1/IC/coco_features/blip2OPT', '/data1/IC/coco_features/clipcap']

        features = [ torch.load( os.path.join(path, pt_file) ).squeeze(0)  for path in self.feature_paths  ] # [1 x 32 x 2560 이 num_features 만큼] 
        features = torch.stack(features) # num_features x 32 x 2560

        # image_path = os.path.join(self.vis_root, ann["image"])
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)/
        caption = ann["caption"]       ## ann["caption"] == caption 인데 그러면 저 self.text_processor는 왜 있는거지?
                                                            ## self.text_processor = lavis.processors.blip_processors.BlipCaptionProcessor

        return {
            "features": features,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

def make_device_pn(device, pn):
    device_num = len(device)
    devices = []
    for i in range(device_num):
        device_name = "cuda:" + device[i]
        devices.append(torch.device(device_name))
    
    assert len(devices) < 4
    assert len(pn) < 5

    if len(devices) == 1:
        devices *= 3
        device1, device2, device3 = devices
        pn1, pn2 = 12, 12
    elif len(devices) == 2:
        device1 = devices[0]
        device2 = devices[1]
        device3 = devices[1]
        pn1, pn2 = int(pn), 12
    else:
        device1, device2, device3 = devices
        length = len(pn)
        if length < 4:
            pn1, pn2 = int(pn[0]), int(pn[1:])
        else:
            pn1, pn2 = int(pn[:2]), int(pn[2:])
            
    return device1, device2, device3, pn1, pn2
