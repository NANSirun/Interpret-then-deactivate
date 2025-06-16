#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""
Textual inversion, observe the ||epsilon-noise||^2 of remaining data
"""

import argparse
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import sys
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import pandas as pd

import sys
import re
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPTokenizer
from SDLens import HookedTextEncoder
from SAE.sae import SparseAutoencoder
from SAE.sae_utils import SAETrainingConfig, Config
import torch.nn.functional as F
import json
import os
import argparse
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import pickle

os.environ["WANDB_API_KEY"] = 'd4f35ec79b5ab6c9312a43d806ebfd4b3772ddce'

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(text_encoder, tokenizer, unet, vae, prompt_embeds, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    with torch.autocast("cuda"):
        images = pipeline(prompt_embeds=prompt_embeds, num_inference_steps=50, generator=generator, num_images_per_prompt=args.num_validation_images
                          ).images

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(selected_features, save_path, safe_serialization=False):
    features = torch.tensor(selected_features, dtype=torch.int32)
    features_dict = {"selected_features": features}

    if safe_serialization:
        safetensors.torch.save_file(features_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(features_dict, save_path)
    
    print(f"Saved selected features to {save_path}")
    # data = torch.load("selected_features.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    # )
    # parser.add_argument(
    #     "--remain_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    # )
    # parser.add_argument("--image_base_dir", type=str, default=None, help="Base directory for the actual images")
    parser.add_argument("--train_data_dir_json", type=str, default=None, help="Path to TRAIN_DIR.json")
    parser.add_argument("--remain_data_dir_json", type=str, default=None, help="Path to REMAIN_DIR.json")
    parser.add_argument("--feature_update_frequency", type=int, default=100, help="Frequency of selected features update")

    # parser.add_argument(
    #     "--placeholder_token",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="A token to use as a placeholder for the concept.",
    # )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--restart_steps", type=int, default=500, help="How many steps to restart the training.")
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Validation prompts, one for each target concept.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["nudity", "naked", "erotic", "sexual"],
        help="List of concepts to learn embeddings for",
    )
    
    parser.add_argument(
        "--vectors_per_concept",
        type=int,
        default=16,
        help="Number of embedding vectors to learn for each concept",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir_json is None:
        raise ValueError("You must specify a train data directory.")

    # if args.train_data_dir is None:
    #     raise ValueError("You must specify a train data directory.")

    
    return args

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        placeholder_token_map,
        json_path,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
        set="train",
    ):
        self.tokenizer = tokenizer
        self.placeholder_token_map = placeholder_token_map
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        # 直接读取json
        # with open(json_path, 'r', encoding='utf-8') as f:
        self.data = load_dataset("json", split="train", data_files=json_path)

        self.num_images = len(self.data)
        self._length = self.num_images * repeats if set == "train" else self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i % self.num_images]
        image_path = item["path"]
        prompt = item["prompt"]
        cls = item["class"]
        
        placeholder_string = " ".join(self.placeholder_token_map[cls])
        text = f"{prompt} {placeholder_string}"   #change to front
        # text = f"{placeholder_string} {prompt}" 

        example = {}
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["input_ids"] = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main():
    code_to_blocks = ["text_model.encoder.layers.8"]

    kwargs = {
        'positions_to_cache': code_to_blocks,
        'save_input': False,
        'save_output': True,
    }

    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    text_encoder_hooked = HookedTextEncoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    
    # change to a fintuned unet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    logger.info(f"Loaded all modules")

    # Add the placeholder tokens in tokenizer for each concept
    placeholder_tokens = []
    for concept in args.concepts:
        base_token = re.sub(r"[<>]", "", concept)
        # Add main token for the concept
        placeholder_tokens.append(f"<{base_token}>")
        # Add additional vectors for the concept
        for i in range(1, args.vectors_per_concept):
            placeholder_tokens.append(f"<{base_token}_{i}>")
    
    args.placeholder_tokens = placeholder_tokens
    
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    expected_tokens = len(args.concepts) * args.vectors_per_concept
    if num_added_tokens != expected_tokens:
        raise ValueError(
            f"The tokenizer already contains some of the tokens. Please pass different concepts."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder_hooked.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder tokens with the embeddings of the initializer token
    # token_embeds = text_encoder_hooked.get_input_embeddings().weight.data
    # with torch.no_grad():
    #     for token_id in placeholder_token_ids:
    #         token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder_hooked.text_model.encoder.requires_grad_(False)
    text_encoder_hooked.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_hooked.text_model.embeddings.position_embedding.requires_grad_(False)
    
    # Ensure the input embeddings are trainable
    text_encoder_hooked.get_input_embeddings().requires_grad_(True)
    
    if args.gradient_checkpointing:
        unet.train()
        text_encoder_hooked.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        text_encoder_hooked.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    placeholder_token_map = {}
    for concept in args.concepts:
        base_token = re.sub(r"[<>]", "", concept)
        tokens = [f"<{base_token}>"] + [f"<{base_token}_{i}>" for i in range(1, args.vectors_per_concept)]
        placeholder_token_map[concept] = tokens

    train_dataset = TextualInversionDataset(
        tokenizer=tokenizer,
        placeholder_token_map=placeholder_token_map,
        json_path=args.train_data_dir_json,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        set="train",
    )

    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    # Prepare everything with our `accelerator`.
    text_encoder_hooked, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder_hooked, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_hooked.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def model_run(batch, tokenizer, text_encoder, kwargs, code_to_block, device):
        input_ids = tokenizer(batch, 
                            return_tensors='pt', 
                            padding=True,
                            truncation=True,
                            max_length=tokenizer.model_max_length)['input_ids'].to(text_encoder.device)
        
        with torch.no_grad():
            kwargs['input_ids'] = input_ids
            _, cache = text_encoder.run_with_cache(**kwargs)
        return cache['output'][code_to_block[0]]

    def find_features_threshold(sae, erase_data, contrast_data, device, code_to_blocks):
        dataset_files = {
            "coco": "prompt/coco/coco_30k.csv",
            "i2p": "prompt/i2p.csv"
        }

        dataset_contrast = load_dataset("csv", split="train", data_files=dataset_files[contrast_data])
        

        kwargs = {
            'positions_to_cache': code_to_blocks,
            'save_input': False,
            'save_output': True,
        }

        dataset_remain = list(set(dataset_contrast['prompt'][:5000]))
        global_top_features = set()

        coco_file_path = os.path.join(args.output_dir, 'coco.pkl')

        if os.path.exists(coco_file_path):
            # Load the existing coco.pkl if it exists
            with open(coco_file_path, 'rb') as f:
                global_top_features = pickle.load(f)  # Assuming it's saved as a set or a list
        else:
            # Generate top_features_r if coco.pkl does not exist
            for i in range(0, len(dataset_remain), 100):
                batch = dataset_remain[i : i + 100]
                if not batch:
                    break
                cache = model_run(batch, tokenizer, text_encoder_hooked, kwargs, code_to_blocks, device)
                feature_acts = sae.encode(cache)
                suffix = feature_acts[:, 0, 1:, :]
                mask = suffix.ne(0).any(dim=1)  
                active = mask.any(dim=0)  # [features]
                indices = torch.nonzero(active, as_tuple=True)[0]
                global_top_features |= set(indices.tolist())
            
            # for j in range(len(dataset_remain)):
            #     with torch.no_grad():
            #         cache_remain = model_run(dataset_remain[j], tokenizer, text_encoder_hooked, kwargs, code_to_blocks, device)
            #         feature_acts_r = sae.encode(cache_remain.reshape(-1, 768))
            #     top_features_r = feature_acts_r[1:].max(0)[0].topk(512, dim=-1).indices.cpu().tolist()
            #     global_top_features.update(top_features_r)
            with open(coco_file_path, 'wb') as f:
                pickle.dump(global_top_features, f)

        features_set = set()
        for i in range(len(erase_data)):
            with torch.no_grad():
                cache_erased = model_run(erase_data[i], tokenizer, text_encoder_hooked, kwargs, code_to_blocks, device)
                feature_acts = sae.encode(cache_erased.reshape(-1, 768))
            top_features = feature_acts[1:].max(0)[0].topk(512, dim=-1).indices.cpu().tolist() # nonzero().squeeze().tolist()
            # topk(512, dim=-1).indices.cpu().tolist()
            target_features = [idx for idx in top_features if idx not in global_top_features]
            features_set.update(target_features)
        
        return list(features_set)


    @torch.enable_grad()
    def strength_with_feature_text_batch(sae, feature_idx, value, module, input, output):
        original = output[0] 
        activated = sae.encode(output[0])
        activated[..., 1:, feature_idx] = activated[..., 1:, feature_idx] * value
        to_add = sae.decoder(activated) + sae.pre_bias
        return (to_add.to(original.device),)


    def activation_modulation_batch(sae, input_ids, block, feature_idx, strength):
        output = text_encoder_hooked.run_with_hooks(
            input_ids=input_ids,
            position_hook_dict={
                block: lambda *args, **kwargs: strength_with_feature_text_batch(
                    sae,
                    feature_idx,
                    strength,
                    *args, **kwargs
                )
            }
        )
        return output[0]

    def train_iteration(text_encoder_hooked, unet, vae, noise_scheduler, sae,
                        code_to_block, top_features_set, strength,
                       batch, optimizer, lr_scheduler, accelerator, weight_dtype):
        """Single training iteration with feature filtering"""
        with accelerator.accumulate(text_encoder_hooked.text_encoder):
            # Get filtered embeddings
            sae_filtered_prompt_embeds = activation_modulation_batch(
                sae, 
                batch['input_ids'], 
                code_to_block[0], 
                feature_idx=top_features_set, 
                strength=strength
            )

            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings for conditioning
            encoder_hidden_states = text_encoder_hooked(batch["input_ids"])[0].to(dtype=weight_dtype)

            # Predict noise residual
            model_pred = unet(noisy_latents, timesteps, sae_filtered_prompt_embeds).sample

            # Calculate target based on prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            return sae_filtered_prompt_embeds, loss
        
    
    
    # Initialize SAE and get initial features to filter
    path_to_checkpoints = 'Interpret-then-deactivate/models'
    sae = SparseAutoencoder.load_from_disk(os.path.join(path_to_checkpoints, 
        f"{code_to_blocks[0]}_k64_hidden524288_auxk256_bs50_lr0.0001_datasetdiffusiondb_nsfw_coco", "4000")).to(accelerator.device)
    for param in sae.parameters():
        param.requires_grad = False

    # Initial feature filtering
    top_features_set = find_features_threshold(
        sae=sae,
        erase_data=["nudity", "naked", "erotic", "sexual"], 
        contrast_data="coco",
        device=accelerator.device, 
        code_to_blocks=code_to_blocks, 
    )

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder_hooked).get_input_embeddings().weight.data.clone()
    min_value = orig_embeds_params.min(0)[0]
    max_value = orig_embeds_params.max(0)[0]
    mvn = torch.distributions.MultivariateNormal(orig_embeds_params.mean(dim=0), covariance_matrix= torch.cov(orig_embeds_params.T))

    # args.validation_prompt = f"{args.validation_prompt} {' '.join(placeholder_token_map['nudity'])}"

    for epoch in range(first_epoch, args.num_train_epochs):
        # text_encoder_hooked.text_encoder.train()
            
        # Training loop with feature filtering
        for step, batch in enumerate(train_dataloader):
            sae_filtered_prompt_embeds, loss = train_iteration(
                text_encoder_hooked, unet, vae, noise_scheduler,
                sae, code_to_blocks, top_features_set, -4,
                batch, optimizer, lr_scheduler, accelerator, weight_dtype
            )

            # Update embeddings while preserving other tokens
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            for token_id in placeholder_token_ids:
                index_no_updates[token_id] = False

            with torch.no_grad():
                accelerator.unwrap_model(text_encoder_hooked).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
                accelerator.unwrap_model(text_encoder_hooked).get_input_embeddings().weight[placeholder_token_ids].clamp_(min_value, max_value)

                embedding = text_encoder.get_input_embeddings().weight
                old_num_tokens = embedding.shape[0] - num_added_tokens
                new_token_ids = list(range(old_num_tokens, old_num_tokens + num_added_tokens))
                                            
                if (global_step +1) % args.restart_steps == 0:
                    # reinitialize learned embeddings
                    accelerator.unwrap_model(text_encoder_hooked).get_input_embeddings().weight[placeholder_token_ids] = mvn.sample((num_added_tokens,)).to(embedding.device)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"selected_features-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"selected_features-steps-{global_step}.safetensors"
                    )
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        # text_encoder_hooked,
                        # placeholder_token_ids,
                        # accelerator,
                        # args,
                        top_features_set,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Periodically update discovered features
                    if global_step % args.feature_update_frequency == 0:
                        nudity_csv = load_dataset("csv", split="train", data_files="Interpret-then-deactivate/prompt/nsfw/nudity_cases.csv")
                        prompts = nudity_csv['prompt']
                        classes = nudity_csv['class']

                        new_tokens = {key:" ".join(token_list)
                            for key, token_list in placeholder_token_map.items()
                        }
                        new_sentences = []
                        for prompt, cls in zip(prompts, classes):
                            new_sentences.append(prompt+ " " + new_tokens[cls]) # 跟Dataset里保持一致
                            # new_sentences.append(new_tokens[cls] + " " + prompt) # 跟Dataset里保持一致

                        # Get new features from current embeddings
                        new_features_set = find_features_threshold(
                            sae=sae,
                            erase_data=new_sentences, 
                            contrast_data="coco",
                            device=accelerator.device, 
                            code_to_blocks=code_to_blocks, 
                        )
                        top_features_set = list(set(top_features_set + new_features_set))

                    # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    if global_step % args.validation_steps == 0 or (global_step-1) % args.validation_steps == 0:
                        with torch.no_grad():
                            text_encoder.get_input_embeddings().weight[placeholder_token_ids] = text_encoder_hooked.get_input_embeddings().weight[placeholder_token_ids]

                        images = log_validation(
                            text_encoder, tokenizer, unet, vae,  sae_filtered_prompt_embeds[:3], args, accelerator, weight_dtype, epoch
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "number of features": len(top_features_set)}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
                

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[placeholder_token_ids] = text_encoder_hooked.get_input_embeddings().weight[placeholder_token_ids]
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            # text_encoder_hooked,
            # placeholder_token_ids,
            # accelerator,
            top_features_set,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    main()