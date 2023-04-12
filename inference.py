import os, torch, json, argparse
from tqdm import tqdm
from modeling_opt_pp import OPTForCausalLM
from transformers import AutoTokenizer
import copy

OPT_MODEL = "facebook/opt-2.7b"

def generate(
        model, tokenizer, prefix_embed,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        prompt="",
        device=torch.device('cuda:0'),
    ):

    with torch.cuda.amp.autocast(
        enabled=(prefix_embed.device != torch.device("cpu"))
    ):
        eos_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
        atts_opt = torch.ones(prefix_embed.size()[:-1], dtype=torch.long).to(device)
        opt_tokens = tokenizer([prompt], return_tensors='pt').to(device)
        input_ids = opt_tokens.input_ids
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        
        outputs = model.generate(
                        input_ids=input_ids,
                        query_embeds=prefix_embed,
                        attention_mask=attention_mask,
                        do_sample=use_nucleus_sampling,
                        top_p=top_p,
                        temperature=temperature,
                        num_beams=num_beams,
                        max_new_tokens=max_length,
                        min_length=min_length,
                        eos_token_id=eos_token_id,
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

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=3)
parser.add_argument('--ofile', type=str, default='snow')
parser.add_argument('--prompt', type=str, default="a photo of")
args = parser.parse_args()

device = torch.device('cuda:' + args.device)
prompt = args.prompt

clipcap_path = os.path.abspath('/data1/IC/nice_val_features/clipcap_ori')
blip2_path = os.path.abspath('/data1/IC/nice_val_features/blip2OPT')

feature_flist = os.listdir(clipcap_path)

output_folder = './output_caption'
output_file = args.ofile + '.json'
os.makedirs(output_folder, exist_ok=True)

opt_model = OPTForCausalLM.from_pretrained(OPT_MODEL, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL, use_fast=False)

opt_model.model.decoder.setting_device(device1 = device, device2 = device, device3 = device)
opt_model.eval()

data = {}
for feature_file in tqdm(feature_flist):
    clipcap_feature = torch.load(os.path.join(clipcap_path, feature_file)).to(device)
    blip2_feature = torch.load(os.path.join(blip2_path, feature_file)).to(device)
    feature = (clipcap_feature + blip2_feature)/2 #*torch.sqrt(torch.tensor(3))

    # feature = torch.load(os.path.join(blip2_path, feature_file)).to(device)
    # feature_shuffle = copy.deepcopy(feature)
    # indices = torch.randperm(32)
    # feature_shuffle = feature_shuffle[indices]

    # feature = feature.unsqueeze(0)
    # feature_shuffle = feature_shuffle.unsqueeze(0)

    # generated = generate(opt_model, tokenizer, feature)
    # generated_shuffle = generate(opt_model, tokenizer, feature_shuffle)
    # data[int(feature_file[:-3])] = [indices, generated, generated_shuffle]

    generated_caption = generate(model=opt_model, tokenizer=tokenizer, prefix_embed=blip2_feature, prompt=prompt, device=device)
    data[int(feature_file[:-3])] = generated_caption
with open(os.path.join(output_folder, output_file), 'w') as fp:
    json.dump(data, fp, default=str)