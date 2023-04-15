import os, torch, random, json, argparse #, yaml, wandb
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
from train_models import ConnectLayer, CaptionDataset_WithFeature, CNN, HPROD, MIX
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datetime import datetime
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

SEED = 27

def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def save_config(args: argparse.Namespace, output_dir):
    config = {}
    for key, item in args._get_kwargs():
        if key in ["total_epoch", "batch_size", "lr"]:
            config[key] = item
    out_path = os.path.join(output_dir, "config.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

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
    with torch.autocast("cuda"):
        eos_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
        atts_opt = torch.ones(prefix_embed.size()[:-1], dtype=torch.long).to(device)
        opt_tokens = tokenizer([prompt]*prefix_embed.shape[0], return_tensors='pt').to(device)
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
        
        return output_text

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', type=str, default='cnn', help='cnn, hprod, mix-qdim')
    parser.add_argument('--output_folder', type=str, default='/data1/checkpoint/connect_layer')
    parser.add_argument('--total_epoch', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='123')
    parser.add_argument('--pn', type=str, default='411', help='splitting OPT layer for pipeline parallelization')
    args = parser.parse_args()

    model_key = args.model_key
    output_folder = args.output_folder
    total_epoch = args.total_epoch
    warmup_steps = args.warmup_steps
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    pn = args.pn

    setup_seeds(SEED)
    # wandb.init(project='connect_layer : ' + model_key)
    
    blip2_feature_path = '/data1/IC/coco_features/blip2OPT/'
    clipcap_feature_path = '/data1/IC/coco_features/clipcap_ori/'
    coco_ann_path = '/data1/IC/coco/annotations/'

    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    output_dir = os.path.join(output_folder, model_key, folder_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_config(args=args, output_dir=output_dir)

    train_feature_paths = [blip2_feature_path + 'train', clipcap_feature_path + 'train']
    train_ann_paths = [coco_ann_path + 'coco_karpathy_train.json']
    
    val_feature_paths = [blip2_feature_path + 'val', clipcap_feature_path + 'val']
    val_ann_paths = [coco_ann_path + 'coco_karpathy_val.json']

    model = ConnectLayer(
        connect_model_key=model_key, num_feature=len(train_feature_paths), device=device, pn=pn
    )
    model.train()
    
    optimizer = torch.optim.AdamW(model.connect_model.parameters(), lr=lr)

    train_dataset = CaptionDataset_WithFeature(feature_paths=train_feature_paths, ann_paths=train_ann_paths)
    val_dataset = CaptionDataset_WithFeature(feature_paths=val_feature_paths, ann_paths=val_ann_paths)


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_epoch*len(train_loader)
    )

    cider_tokenizer = PTBTokenizer()
    best_cider = 0
    best_epoch = 0
    for cur_epoch in trange(total_epoch):
        # print(f">>>>>> epoch : {cur_epoch}")
        # print(">>> train")
        # progress_train = tqdm(total=len(train_loader), desc=f"epoch {cur_epoch} train")
        # for idx, samples in enumerate(train_loader):
        #     model.zero_grad()
        #     loss = model(samples=samples)
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
        #     # wandb.log({'loss' : loss.item()})
        #     progress_train.set_postfix({"loss" : loss.item()})
        #     progress_train.update()
        # progress_train.close()

        with torch.no_grad():
            print(">>> val")

            gts = {}
            res = {}
            
            progress_val = tqdm(total=len(val_loader), desc=f"epoch {cur_epoch} validation")
            for samples in val_loader:
                features = samples["features"]
                features = features.to(model.device1)
                with torch.autocast("cuda"):
                    if model.model_key == CNN:
                        feature = model.connect_model(features).squeeze(1)
                    elif model.model_key == HPROD:
                        feature = model.connect_model(features)
                    elif model.model_key == MIX:
                        features = features.transpose(1, 2).reshape(
                            features.shape[0],
                            model.num_query_token,
                            model.num_feature * model.query_dimension
                        )
                        feature = model.connect_model(features)

                targets = samples['text_input']
                generated_caption = generate(
                    model=model.opt_model, tokenizer=model.opt_tokenizer, prefix_embed=feature, prompt=model.prompt, device=model.device1
                )
                for idx, imgId in enumerate(samples["image_id"]):
                    gts[imgId] = [{"image_id" : imgId, "caption" : targets[idx], "id" : imgId}]
                    res[imgId] = [{"image_id" : imgId, "caption" : generated_caption[idx], "id" : imgId}]
                
                progress_val.update()
            gts = cider_tokenizer.tokenize(gts)
            res = cider_tokenizer.tokenize(res)
            new_cider = Cider().compute_score(gts, res)[0]
            progress_val.close()

        if best_cider < new_cider:
            best_epoch = cur_epoch
            torch.save(
            model.connect_model.state_dict(), os.path.join(output_dir, f"model_{model_key}_max_cider.pt")
            )
            with open(os.path.join(output_dir, 'generated_caption.json'), 'w') as f:
                json.dump(res, f)

        torch.save(
            model.connect_model.state_dict(), os.path.join(output_dir, f"model_{model_key}.pt")
        )
        torch.save(
            scheduler.state_dict(), os.path.join(output_dir, f"schedular_{model_key}.pt")
        )
        torch.save(
            optimizer.state_dict(), os.path.join(output_dir, f"optimizer_{model_key}.pt")
        )

    print(f"best cider : {best_cider} at epoch {best_epoch}")

if __name__ == "__main__":
    main()