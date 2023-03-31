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

# WEIGHTS_PATHS = {
#     "coco_gpt": "coco_train/gpt-finetuned/coco_prefix-009.pt",
#     "coco_gpt008": "coco_train/gpt-finetuned/coco_prefix-008.pt",
#     # "conceptual-captions": "conceptual_weights.pt",
# }

def direct_weiht_paths(language_model):
    if language_model == 'gpt2':
        WEIGHTS_PATHS = {
            "coco": "/data/daisy/clipcap_output/gpt2_32quries/coco_prefix-009.pt",
            "coco_gpt008": "/data/daisy/clipcap_output/gpt-finetuned/coco_prefix-008.pt",
        }
        print('your language model is : GPT-2')
        return WEIGHTS_PATHS
    elif language_model == 'opt':
        WEIGHTS_PATHS = {
        "coco": "/data/daisy/clipcap_output/opt13b_32query/coco_prefix-018.pt",
        "coco_gpt008": "/data/daisy/clipcap_output/opt13b_32query/coco_prefix-008.pt",
        }
        print('your language model is : OPT')
        return WEIGHTS_PATHS

WEIGHTS_PATHS = direct_weiht_paths('opt')


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
        if use_beam_search:
            return prefix_embed, generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return prefix_embed, generate2(model, self.tokenizer, embed=prefix_embed)


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


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt="a photo of",
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = "/n",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = embed.device
    embed = embed.type(torch.DoubleTensor)
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                # generated = model.gpt.transformer.wte(tokens) # GPT-2
                generated = model.gpt.decoder.embed_tokens(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            # next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
            #     generated.shape[0], 1, -1
            # ) # GPT-2
            next_token_embed = model.gpt.model.decoder.embed_tokens(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1) # OPT
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt="a photo of",
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = "",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = embed.device
    embed = embed.type(torch.DoubleTensor)

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # next_token_embed = model.gpt.transformer.wte(next_token) # GPT-2
                next_token_embed = model.gpt.model.decoder.embed_tokens(next_token).to(device) # OPT
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

