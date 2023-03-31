# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer # OPTForCausalLM
from modeling_opt_pp import OPTForCausalLM
import skimage.io as io
import PIL.Image

import cog

# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

def direct_weight_paths(language_model):
    if language_model == 'gpt2':
        WEIGHTS_PATHS = {
            "coco": "/data/IC/clipcap/coco_prefix-000.pt",
            "coco_gpt008": "/data/IC/clipcap/coco_prefix-000.pt",
        }
        print('your language model is : GPT-2')
        return WEIGHTS_PATHS
    elif language_model == 'opt':
        WEIGHTS_PATHS = {
        "opt_000": "/data/IC/clipcap/model_coco_prefix-000.pt",
        "opt_001": "/data/IC/clipcap/model_coco_prefix-001.pt",
        }
        print('your language model is : OPT')
        return WEIGHTS_PATHS

WEIGHTS_PATHS = direct_weight_paths('opt')


D = torch.device
CPU = torch.device("cpu")
OPT_MODEL = 'facebook/opt-2.7b'

class Predictor(cog.Predictor):
    def setup(self, args):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.device = torch.device("cuda")
        self.device1 = make_device(args)[0]
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device1, jit=False
        )
        self.args = args
        
        if self.args.language_model == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif self.args.language_model == 'opt':
            self.tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL)

        self.models = {}
        self.prefix_length = args.prefix_length
        for key, weights_path in WEIGHTS_PATHS.items():
            
            model = ClipCaptionModel(args)
            model.load_state_dict(torch.load(weights_path, map_location=CPU))
            model = model.eval()
            # model = model.to(self.device)
            self.models[key] = model

    @cog.input("image", type=cog.Path, help="Input image")
    @cog.input(
        "model",
        type=str,
        options=WEIGHTS_PATHS.keys(),
        default="customized",
        help="Model to use",
    )
    @cog.input(
        "use_beam_search",
        type=bool,
        default=False,
        help="Whether to apply beam search to generate the output text",
    )
    def predict(self, image, model, use_beam_search):
        """Run a single prediction on the model"""
        image = io.imread(image)
        model = self.models[model]
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device1)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device1, dtype=torch.float32
            )
            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            
        return generate(model, self.tokenizer, prefix_embed, self.device1)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.args.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        if self.args.language_model == 'gpt2':
            embedding_text = self.gpt.transformer.wte(tokens)
        elif self.args.language_model == 'opt':
            embedding_text = self.gpt.model.decoder.embed_tokens(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.args.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text.to(self.device1)), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, args, clip_length: Optional[int] = 32, prefix_size: int = 512, num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.args = args
        self.prefix_size = prefix_size
        self.clip_length = clip_length
        self.num_layers = num_layers
        self.device1, device2, device3 = make_device(args)
        pn1, pn2 = int(args.pn[0]), int(args.pn[1:])
        
        if self.args.language_model == 'gpt2':
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        elif self.args.language_model == 'opt':
            print('clipcaption - LM : OPT')
            self.gpt = OPTForCausalLM.from_pretrained(OPT_MODEL)
            self.gpt_embedding_size = self.gpt.model.decoder.embed_tokens.weight.shape[1]
            self.gpt.model.decoder.setting_device(device1=self.device1, device2=device2, device3=device3, pn1=pn1, pn2=pn2)
            
        self.clip_project = TransformerMapper(dim_clip=self.prefix_size, dim_embedding=self.gpt_embedding_size, 
                                              prefix_length=self.args.prefix_length, clip_length=self.clip_length, num_layers=self.num_layers).to(self.device1)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def make_device(args):
    device_num = len(args.device)
    devices = []
    for i in range(device_num):
        device = "cuda:" + args.device[i]
        devices.append(torch.device(device))
    
    assert len(devices) < 4
    if len(devices) == 1:
        devices *= 3
        device1, device2, device3 = devices
    elif len(devices) == 2:
        device1 = devices[0]
        device2 = devices[1]
        device3 = devices[1]
    else:
        device1, device2, device3 = devices
    return device1, device2, device3


def generate(model, tokenizer, prefix_embed, device1,
             use_nucleus_sampling=False,
             num_beams=5,
             max_length=30,
             min_length=1,
             top_p=0.9,
             repetition_penalty=1.5,
             length_penalty=1.0,
             num_captions=1,
             temperature=1,
             prompt=""):
    
    atts_opt = torch.ones(prefix_embed.size()[:-1], dtype=torch.long).to(device1)
    opt_tokens = tokenizer([prompt], return_tensors='pt').to(device1)
    input_ids = opt_tokens.input_ids
    query_embeds = prefix_embed
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
    
    return output_text[0]