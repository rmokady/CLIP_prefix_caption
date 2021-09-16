import skimage.io as io
import clip   # installed from https://github.com/openai/CLIP
from PIL import Image
from custom_types import *
import pickle
import json


def main():
    device = CUDA(0)
    suffix = "train"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)

    print("%0d captions loaded from json " % len(data))

    all_embeddings = []
    all_captions = []
    counter = 0

    for ii, d in enumerate(data):
        img_id = d["image_id"]

        try:
            filename = "./data/coco/train2014/COCO_train2014_%012d.jpg" % int(img_id)
            image = io.imread(filename)
        except FileNotFoundError:
            filename = "./data/coco/val2014/COCO_val2014_%012d.jpg" % int(img_id)
            image = io.imread(filename)

        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        d["clip_embedding"] = counter
        all_embeddings.append(prefix)
        all_captions.append(d)

        if counter % 10000 == 0:
             with open(f"./data/coco/oscar_split_{suffix}.pkl", 'wb') as f:
                 pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
        counter += 1

    with open(f"./data/coco/oscar_split_{suffix}.pkl", 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    main()