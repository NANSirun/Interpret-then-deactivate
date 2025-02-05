from torchvision.models import vit_h_14, ViT_H_14_Weights, resnet50, ResNet50_Weights
from torchvision.io import read_image
from PIL import Image
import os, argparse
import torch
import pandas as pd
from tqdm import tqdm

def get_image_paths(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'ImageClassification',
                    description = 'Takes the path of images and generates classification scores')
    parser.add_argument('--folder_path', help='path to images', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--topk', type=int, required=False, default=5)
    parser.add_argument('--batch_size', type=int, required=False, default=250)
    args = parser.parse_args()

    folder = args.folder_path
    topk = args.topk
    device = args.device
    batch_size = args.batch_size
    save_path = args.save_path
    if save_path is None:
        name_ = folder.split('/')[-1]
        save_path = f'{folder}/{name_}_classification.csv'
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()
    

    scores = {}
    categories = {}
    indexes = {}
    for k in range(1,topk+1):
        scores[f'top{k}']= []
        indexes[f'top{k}']=[]
        categories[f'top{k}']=[]

    image_paths = get_image_paths(folder)
    
    preprocess = weights.transforms()

    images = []
    for name in image_paths:
        img = Image.open(os.path.join(folder,name))
        batch = preprocess(img)
        images.append(batch)

    if batch_size == None:
        batch_size = len(image_paths)
    if batch_size > len(image_paths):
        batch_size = len(image_paths)
        
    
    images = torch.stack(images)
    # Step 4: Use the model and print the predicted category
    for i in tqdm(range(((len(image_paths) - 1) // batch_size) + 1), desc="Processing Batches"):
    # for i in range(((len(names)-1)//batch_size)+1):
        batch = images[i*batch_size: min(len(image_paths), (i+1)*batch_size)].to(device)
        with torch.no_grad():
            prediction = model(batch).softmax(1)
        probs, class_ids = torch.topk(prediction, topk, dim = 1)
        
        for k in range(1,topk+1):
            scores[f'top{k}'].extend(probs[:,k-1].detach().cpu().numpy())
            indexes[f'top{k}'].extend(class_ids[:,k-1].detach().cpu().numpy())
            categories[f'top{k}'].extend([weights.meta["categories"][idx] for idx in class_ids[:,k-1].detach().cpu().numpy()])

    if save_path is not None:

        labels = []
        classes = []

        for path in image_paths:
            name = os.path.basename(path)
            labels.append(int(name.split('_')[0]))  
            classes.append(name.split('_')[1]) 

        dict_final = {'labels': labels, 'classes': classes}

        for k in range(1,topk+1):
            dict_final[f'category_top{k}'] = categories[f'top{k}'] 
            dict_final[f'index_top{k}'] = indexes[f'top{k}'] 
            dict_final[f'scores_top{k}'] = scores[f'top{k}'] 

        df_results = pd.DataFrame(dict_final)

        df_results["top1_match"] = df_results["labels"] == df_results["index_top1"]
        df_results["top5_match"] = df_results.apply(
            lambda row: row["labels"] in [row[f"index_top{i}"] for i in range(1, 6)],
            axis=1,
        )
        
        df_results["top1_score"] = df_results["scores_top1"]
        df_results["top5_score"] = df_results.apply(
            lambda row: sum([row[f"scores_top{i}"] for i in range(1, 6)]),
            axis=1,
        )

        # Group by labels and count matches
        sample_counts = df_results.groupby("classes").agg(
            top1_count=("top1_match", "mean"),
            top5_count=("top5_match", "mean"),
            top1_score=("top1_score", "mean"),
            top5_score=("top5_score", "mean"),
        )

        sample_counts.to_csv(save_path)
