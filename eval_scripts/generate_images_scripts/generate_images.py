import sys
from tqdm import tqdm
import argparse
import torch
from datasets import load_dataset
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from PIL import Image

from src.SDLens import HookedTextEncoder

from src.SAE.sae import SparseAutoencoder
from src.SAE.sae_utils import Config

import json
import os

from diffusers import StableDiffusionPipeline


def find_the_threshold_for_nudity(sae, tokenizer,text_encoder,erase_data, coco_remain,datasets_threshold, blocks, device):
    kwargs = {
        'positions_to_cache': blocks,
        'save_input': False,
        'save_output': True,
    }    
    def model_run(batch,):
        input_ids = tokenizer(batch, 
                             return_tensors='pt', 
                             padding=True,  
                             truncation=True, 
                             max_length=tokenizer.model_max_length)['input_ids'].to(text_encoder.device)
        
        with torch.no_grad():
            kwargs['input_ids'] = input_ids
            _, cache = text_encoder.run_with_cache(**kwargs)
        return cache['output'][blocks[0]]
    

    global_top_features = set()
    for j in range(len(coco_remain)):
        with torch.no_grad():
            cache_remain = model_run(coco_remain[j])
            feature_acts_r = sae.encode(cache_remain.reshape(-1, 768))
        top_features_r = feature_acts_r[1:].max(0)[0].topk(128, dim=-1).indices.cpu().tolist()
        global_top_features.update(top_features_r)
    top_features_set = set()
    for i in range(len(erase_data)):
        print(f"Concept: {i}")
        with torch.no_grad():
            cache_erased = model_run(erase_data[i])
            feature_acts = sae.encode(cache_erased.reshape(-1, 768))
        top_features = feature_acts[1:].max(0)[0].topk(128, dim=-1).indices.cpu().tolist()
        target_features = [idx for idx in top_features if idx not in global_top_features]
        top_features_set.update(target_features)

    top_features_set = list(top_features_set)

    threshold = None
    
    return threshold, top_features_set



def find_the_threshold_for_celebrity(sae, tokenizer,text_encoder,erase_data, remain_data,datasets_threshold, blocks, device):
    kwargs = {
        'positions_to_cache': blocks,
        'save_input': False,
        'save_output': True,
    }
    def get_mse(cache, feature_idx, value):
        with torch.no_grad():
            activated = sae.encode(cache)
            activated[..., 1:, feature_idx] = activated[..., 1:, feature_idx] * value
            to_add = sae.decoder(activated) + sae.pre_bias
            mse_loss = F.mse_loss(to_add, cache, reduction='none') 
        return  mse_loss.mean(dim=(1, 3)).topk(1)[0].mean(1).tolist() 
    
    def model_run(batch,):
        input_ids = tokenizer(batch, 
                             return_tensors='pt', 
                             padding=True,  
                             truncation=True, 
                             max_length=tokenizer.model_max_length)['input_ids'].to(text_encoder.device)
        
        with torch.no_grad():
            kwargs['input_ids'] = input_ids
            _, cache = text_encoder.run_with_cache(**kwargs)
        return cache['output'][blocks[0]]
    
    def batch_process(data_dict, key, features_idx, batch_size=100, scale_factor=-8):
        results = []
        if key==None and isinstance(data_dict, list):
            data = data_dict
        else:
            data = data_dict[key]
        num_batches = len(data) // batch_size + 1
        for i in range(num_batches):
            batch_data = data[i * batch_size: (i + 1) * batch_size]
            if not batch_data: 
                continue
            batch_cache = model_run(batch_data)
            results.extend(get_mse(batch_cache, features_idx, scale_factor))
        return results
         
        
    global_top_features = set()
    for j in range(len(remain_data)):
        with torch.no_grad():
            cache_remain = model_run(remain_data[j])
            feature_acts_r = sae.encode(cache_remain.reshape(-1, 768))
        top_features_r = feature_acts_r[1:].max(0)[0].topk(128, dim=-1).indices.cpu().tolist()
        global_top_features.update(top_features_r)
    top_features_set = set()
    for i in range(len(erase_data)):
        print(f"Concept: {i}")
        with torch.no_grad():
            cache_erased = model_run(erase_data[i])
            feature_acts = sae.encode(cache_erased.reshape(-1, 768))
        top_features = feature_acts[1:].max(0)[0].topk(128, dim=-1).indices.cpu().tolist()
        target_features = [idx for idx in top_features if idx not in global_top_features]
        top_features_set.update(target_features)
    top_features_set = list(top_features_set)

    
    target = batch_process(
        data_dict=datasets_threshold,
        key="prompt",
        features_idx=top_features_set
    )
    threshold = (1-0.01) * (min(target))
    
    return threshold, top_features_set

def find_the_threshold_for_style(sae, tokenizer,text_encoder,erase_data, remain_data,datasets_threshold, blocks, device):
    kwargs = {
        'positions_to_cache': blocks,
        'save_input': False,
        'save_output': True,
    }
    def get_mse(cache, feature_idx, value):
        with torch.no_grad():
            activated = sae.encode(cache)
            activated[..., 1:, feature_idx] = activated[..., 1:, feature_idx] * value
            to_add = sae.decoder(activated) + sae.pre_bias
            mse_loss = F.mse_loss(to_add, cache, reduction='none') 
        return  mse_loss.mean(dim=(1, 2, 3)).tolist() # 
    
    
    def model_run(batch,):
        input_ids = tokenizer(batch, 
                             return_tensors='pt', 
                             padding=True,  
                             truncation=True, 
                             max_length=tokenizer.model_max_length)['input_ids'].to(text_encoder.device)
        
        
        with torch.no_grad():
            kwargs['input_ids'] = input_ids
            _, cache = text_encoder.run_with_cache(**kwargs)
        return cache['output'][blocks[0]]
    
    def batch_process(data_dict, key, features_idx, batch_size=100, scale_factor=-6):
        results = []
        if key==None and isinstance(data_dict, list):
            data = data_dict
        else:
            data = data_dict[key]
        num_batches = len(data) // batch_size + 1
        for i in range(num_batches):
            batch_data = data[i * batch_size: (i + 1) * batch_size]
            if not batch_data: 
                continue
            batch_cache = model_run(batch_data)
            results.extend(get_mse(batch_cache, features_idx, scale_factor))
        return results
         
        
    global_top_features = set()
    for j in range(len(remain_data)):
        with torch.no_grad():
            cache_remain = model_run(remain_data[j])
            feature_acts_r = sae.encode(cache_remain.reshape(-1, 768))
        top_features_r = feature_acts_r[1:].max(0)[0].topk(256, dim=-1).indices.cpu().tolist()
        global_top_features.update(top_features_r)
    top_features_set = set()
    for i in range(len(erase_data)):
        print(f"Concept: {i}")
        with torch.no_grad():
            cache_erased = model_run(erase_data[i])
            feature_acts = sae.encode(cache_erased.reshape(-1, 768))
        top_features = feature_acts[1:].max(0)[0].topk(256, dim=-1).indices.cpu().tolist()
        target_features = [idx for idx in top_features if idx not in global_top_features]
        top_features_set.update(target_features)
        # top_features_set.update(top_features)
    top_features_set = list(top_features_set)

    
    target = batch_process(
        data_dict=datasets_threshold,
        key="prompt",
        features_idx=top_features_set
    )
    
    threshold = (1-0.01) * (min(target))
    
    
    return threshold, top_features_set




def extract_text_encoder_ckpt(ckpt_path):
    full_ckpt = torch.load(ckpt_path)
    new_ckpt = {}
    for key in full_ckpt.keys():
        if 'text_encoder.text_model' in key:
            new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
    return new_ckpt

def generate_images(path_to_checkpoints,prompts_domain,num_samples_per_prompt, batch_size_pipe, pretrained_path,save_folder,model_name, device='cuda:1'):
    
    dataset_files = {
        "coco": "prompt/coco_30k.csv",
        "i2p": "prompts/i2p.csv",
        "celebrity":"prompt/celebrity/celebrity.csv",
        "celebrity_remain":"prompt/celebrity/celebrity_remain.csv",
        "celebrity_target":"prompt/celebrity/celebrity_target.csv",
        "imagenet":"prompt/imagenet1k-id2label.csv",
        "style":"prompt/style/style.csv",
        "style_target":"prompt/style/style_target.csv",
        "style_remain":"prompt/style/style_remain.csv",
        "character":"prompt/character.csv",
        "diffusiondb":"prompt/diffusiondb/diffusiondb10K_modified.csv",
        "diffusiondb_nsfw":"prompt/diffusiondb/diffusiondb10K_nsfw.csv",
    }

    
    if model_name == "SD":
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_path,
            torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
    
    elif "ItD" in model_name:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
        text_encoder = HookedTextEncoder.from_pretrained(pretrained_path, subfolder="text_encoder")
        text_encoder.to(device)
        for param in text_encoder.parameters():
            param.requires_grad = False
        

        if "nudity" in model_name:   
            cfg = Config(json.load(open('scripts/config/config_text_nsfw.json')), wandb_name='sae_text_nsfw')    
            blocks = cfg.block_name 
            erase_data = ["nudity", "naked", "erotic", "sexual"]
            datasets_coco = load_dataset("csv", split="train", data_files=dataset_files["coco"])
            coco_remain = datasets_coco.shuffle()["prompt"][:50]
            
            sae = SparseAutoencoder.load_from_disk(
                os.path.join(path_to_checkpoints, "4000"),
                ).to(device)
            for param in sae.parameters():
                param.requires_grad = False
            threshold, top_features_set = find_the_threshold_for_nudity(sae,tokenizer,text_encoder,erase_data, coco_remain, datasets_coco, blocks, device)

        elif "celeb" in model_name:
            cfg = Config(json.load(open('scripts/config/config_text.json')), wandb_name='sae_text')    
            blocks = cfg.block_name 
            datasets_celeb_erase = load_dataset("csv", split="train", data_files=dataset_files["celebrity_target"])
            datasets_celeb_remain = load_dataset("csv", split="train", data_files=dataset_files["celebrity_remain"])
            datasets_style_erase = load_dataset("csv", split="train", data_files=dataset_files["style_target"])
            datasets_style_remain = load_dataset("csv", split="train", data_files=dataset_files["style_remain"])
            celeb_erased = list(set(datasets_celeb_erase['prompt']))
            celeb_remain = list(set(datasets_celeb_remain['prompt']))\
                + list(set(datasets_style_erase['prompt']))\
                    + list(set(datasets_style_remain['prompt']))
            

            sae = SparseAutoencoder.load_from_disk(
                os.path.join(path_to_checkpoints, "6000"),
                ).to(device)
            for param in sae.parameters():
                param.requires_grad = False
        
            threshold, top_features_set = find_the_threshold_for_celebrity(sae,tokenizer,text_encoder,celeb_erased, celeb_remain, datasets_celeb_erase, blocks, device)
            
        elif "style" in model_name:
            cfg = Config(json.load(open('scripts/config/config_text.json')), wandb_name='sae_text')    
            blocks = cfg.block_name 
            datasets_celeb_erase = load_dataset("csv", split="train", data_files=dataset_files["celebrity_target"])
            datasets_celeb_remain = load_dataset("csv", split="train", data_files=dataset_files["celebrity_remain"])
            datasets_style_erase = load_dataset("csv", split="train", data_files=dataset_files["style_target"])
            datasets_style_remain = load_dataset("csv", split="train", data_files=dataset_files["style_remain"])

            
            style_erased = list(set(datasets_style_erase['prompt']))
            style_remain = list(set(datasets_style_remain['prompt']))

            path_to_checkpoints = 'models/text_model.encoder.layers.8_k64_hidden524288_auxk256_bs50_lr0.0001_datasetcelebrity_style_coco'

            sae = SparseAutoencoder.load_from_disk(
                os.path.join(path_to_checkpoints, "6000"),
                ).to(device)
            for param in sae.parameters():
                param.requires_grad = False
        
            threshold, top_features_set = find_the_threshold_for_style(sae,tokenizer,text_encoder,style_erased, style_remain, datasets_style_erase, blocks, device)

        @torch.no_grad()
        def strength_with_feature_text_batch(sae, feature_idx, value, module, input, output):
            original = output[0] 
            activated = sae.encode(output[0])
            activated[..., 1:, feature_idx] = activated[..., 1:, feature_idx] * value
            to_add = sae.decoder(activated) + sae.pre_bias
            # activated @ sae.decoder.weight.T
            mse_loss = F.mse_loss(to_add, output[0], reduction='none')
            mean_mse = mse_loss.mean(dim=(1, 2)) 
            filtered_output = []
            for i in range(original.shape[0]):
                if mean_mse[i] >= threshold:
                    filtered_output.append(to_add[i])
                else:
                    filtered_output.append(original[i])
            filtered_output = torch.stack(filtered_output, dim=0)
            return (filtered_output.to(original.device),)

        
        kwargs = {
            'positions_to_cache': blocks,
            'save_input': False,
            'save_output': True,
        }
        def activation_modulation_batch(sae, prompt, block, feature_idx, strength):
            input_ids = tokenizer(prompt, return_tensors='pt', padding=True,  truncation=True, max_length=tokenizer.model_max_length)['input_ids'].to(text_encoder.device)
            output = text_encoder.run_with_hooks(
                input_ids,
                position_hook_dict={
                    block: lambda *args, **kwargs: strength_with_feature_text_batch(
                        sae,
                        feature_idx,
                        strength,
                        # threshold,
                        *args, **kwargs
                    ) 
                }
            )
            return output[0]
        

        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_path,
            torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

    
    batch_size = batch_size_pipe  # can be changed

    if ',' in prompts_domain:
        csv_names = prompts_domain.split(',')
    else:
        csv_names = [prompts_domain] # if prompts_domain else [] 

    base_folder = save_folder
    for target_csv in csv_names:
        print(f"now generating {target_csv}")
        save_folder = os.path.join(base_folder, target_csv)
        os.makedirs(save_folder, exist_ok=True)
        
        dataset_prompts = load_dataset("csv", split="train", data_files=dataset_files[target_csv])
        print("Rows in dataset_prompts (from CSV):", len(dataset_prompts))
        
        
        if target_csv == "coco":
            name_list = dataset_prompts["coco_id"] 
            seed_list = dataset_prompts["evaluation_seed"]
            class_idx_list = dataset_prompts["case_number"]
        elif target_csv=="i2p":
            name_list = dataset_prompts["categories"] 
            seed_list = dataset_prompts["evaluation_seed"]
            class_idx_list = dataset_prompts["case_number"]
        else:
            name_list = dataset_prompts["class"] 
            class_idx_list = dataset_prompts["class_idx"]

        prompt_list = dataset_prompts["prompt"] 
        case_number_list = dataset_prompts["case_number"]
        # We will iterate over prompt_list in chunks
        n = len(prompt_list)
        num_batches = (n + batch_size - 1) // batch_size        
        # for batch_index in range(num_batches):
        for batch_index in tqdm(range(num_batches), desc="Processing Batches"):
            
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, n)
            current_batch_size= end_idx-start_idx
            if start_idx >= end_idx:
                break
            
            current_prompts = prompt_list[start_idx:end_idx]  # sub-list
            current_class_idx = class_idx_list[start_idx:end_idx]
            current_name = name_list[start_idx:end_idx]
            current_case_number = case_number_list[start_idx:end_idx]
            negative_prompt=""
            if target_csv=="coco" or target_csv=="i2p":
                current_seeds = seed_list[start_idx:end_idx]
                generators_per_prompt = [
                    [torch.Generator(device='cuda:0').manual_seed(seed) for seed in current_seeds]
                ]
            elif "style" in target_csv:
                fixed_seeds = [x + 1 for x in range(num_samples_per_prompt)]
                current_seeds = fixed_seeds
                generators_per_prompt = [
                    [torch.Generator(device='cuda:0').manual_seed(seed) for seed in current_seeds]
                    for _ in range(current_batch_size)
                ]
            elif "celeb" in target_csv:
                negative_prompt="bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
                # fixed_seeds = [1, 2, 3, 4, 5]
                fixed_seeds = [x + 1 for x in range(num_samples_per_prompt)]
                current_seeds = fixed_seeds
                generators_per_prompt = [
                    [torch.Generator(device='cuda:0').manual_seed(seed) for seed in current_seeds]
                    for _ in range(current_batch_size)
                ]
            else:
                fixed_seeds = [x + 1 for x in range(num_samples_per_prompt)]
                current_seeds = fixed_seeds
                generators_per_prompt = [
                    [torch.Generator(device='cuda:0').manual_seed(seed) for seed in current_seeds]
                    for _ in range(current_batch_size)
                ]
            generators = [gen for prompt_generators in generators_per_prompt for gen in prompt_generators]

            # generate images
            if "ItD" in model_name:
                current_prompt_embeds=activation_modulation_batch(sae, current_prompts, blocks[0], feature_idx=top_features_set, strength=-8)  
                outputs = pipe( 
                    prompt_embeds=current_prompt_embeds,
                    num_inference_steps=50, 
                    negative_prompt=[negative_prompt]*current_batch_size,
                    num_images_per_prompt=num_samples_per_prompt,
                    generator=generators)
            else:
                outputs = pipe(
                    prompt = current_prompts,
                    num_inference_steps=50, 
                    negative_prompt=[negative_prompt]*current_batch_size,
                    num_images_per_prompt=num_samples_per_prompt,
                    generator=generators)
            
            
            images = outputs.images
            sub_batch_size = len(current_prompts)
            for i in range(sub_batch_size):

                c_idx = current_class_idx[i]
                c_name = current_name[i]
                c_case_number = current_case_number[i]
                
                for j in range(num_samples_per_prompt):
                    image_idx = i * num_samples_per_prompt + j
                    image = images[image_idx]
                    
                    if isinstance(image, torch.Tensor):
                        image = Image.fromarray(image.cpu().numpy()).convert("RGB")
                    elif not isinstance(image, Image.Image):
                        image = Image.fromarray(image).convert("RGB")
                    
                    # {save_folder}/{class_idx}_{name}_{case_number}_{jth of this case}.png
                    save_path = os.path.join(
                        save_folder, f"{c_name}",
                        f"{c_idx}_{c_name}_{c_case_number}_{j}.png"
                    )
                    if target_csv=="coco":
                        save_path = os.path.join(
                            save_folder, f"{c_name}",
                            f"{c_idx}_{c_name}_{c_case_number}_{j}.jpg"
                        )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)
                    print(f"save at {save_path}")
                    # import pdb; pdb.set_trace()


def main():
    parser = argparse.ArgumentParser(description="Main program to generate images using diffuser pipe.")
    
    parser.add_argument('--pretrained_path', type=str, required=False,default="/home/guest/zsf/Pretrained_model_files/sd_v1-4/",
                        help='Path to the pretrained model')
    parser.add_argument('--path_to_checkpoints', type=str, required=False,
                        help='Path to the SAE models')
    parser.add_argument('--model_name', type=str, required=False, default="SD", choices=['SD', 'ItD-celeb', 'ItD-style', 'ItD-nudity'],
                        help='name of model or path of model')
    parser.add_argument('--num_samples_per_prompt', type=int, default=10,
                        help='Number of samples to generate for each prompt')
    parser.add_argument('--batch_size_pipe', type=int, default=5,
                        help='The number of prompts to be input into the pipe at once.')
    parser.add_argument('--prompts_domain', type=str, required=False, default="coco",
                        help='Path to the file containing prompts you want to gengerate')
    parser.add_argument('--save_folder', type=str, default='results',
                        help='Directory to save generated images')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device type to use ("cuda" or "cpu") (default: "cuda" if available else "cpu")')
    
    args = parser.parse_args()
    
    prompts_domain=args.prompts_domain
    num_samples_per_prompt=args.num_samples_per_prompt
    batch_size_pipe=args.batch_size_pipe
    pretrained_path=args.pretrained_path
    save_folder=args.save_folder
    device=args.device
    model_name = args.model_name
    path_to_checkpoints = args.path_to_checkpoints
    
    generate_images(path_to_checkpoints, prompts_domain,num_samples_per_prompt, batch_size_pipe, pretrained_path,save_folder,model_name, device)
    

if __name__ == "__main__":
    main()