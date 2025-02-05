# Code adapted from https://github.com/Shilin-LU/MACE/blob/main/metrics/evaluate_clip_score.py
import os
from PIL import Image
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from argparse import ArgumentParser
import torch


@torch.no_grad()
def mean_clip_score(image_dir, prompts_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # device = torch.device("cpu") 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_df=pd.read_csv(prompts_path)
    

    case_number_to_prompt = {}
    for idx, row in text_df.iterrows():
        case_number_str = str(row['case_number'])
        case_number_to_prompt[case_number_str] = row['prompt']

    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    similarities = []

    for path in tqdm(image_paths):
        filename = os.path.basename(path)
        parts = filename.split('_')
        case_number_str = parts[-2]
        if case_number_str not in case_number_to_prompt:
            continue
        text = case_number_to_prompt[case_number_str]

        image = Image.open(path)
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True,truncation=True,max_length=77)
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        clip_score = outputs.logits_per_image[0][0].detach().cpu()  # CLIP 相似度分数
        similarities.append(clip_score)

    
    similarities=np.array(similarities)
    
    mean_similarity=np.mean(similarities)
    std_similarity = np.std(similarities)

    print('-------------------------------------------------')
    print('\n')
    print(f"Mean CLIP score ± Standard Deviation: {mean_similarity:.4f}±{std_similarity:.4f}")   
    with open("clipscore.log", "a+") as f:
        f.write('\n\n')
        f.write( "=" * 30)
        f.write(f'\nDir1: {image_dir}\n')
        f.write(f'Dir2: {prompts_path}\n')
        f.write(f"Mean CLIP score ± Standard Deviation: {mean_similarity:.4f} ± {std_similarity:.4f}\n")

        f.write("=" * 30)
        f.write('\n\n')
        f.close()

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='/home/guest/data/tzh/sdxl-unbox/SD/coco')
    parser.add_argument("--prompts_path", type=str, default='/home/guest/data/nsr/UnlearningSAE/prompt/coco_30k.csv')
    args = parser.parse_args()

    image_dir=args.image_dir
    prompts_path=args.prompts_path
    
    mean_clip_score(image_dir, prompts_path)