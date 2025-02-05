# Interpret-then-Deactivate

## Setup
Creating a Conda Environment
```sh
git clone
pip install -r requirements.txt
```

## Training SAE
The pretrained models are available here: https://huggingface.co/ItD-anonymous/SAE

You can train your own SAE by changing the config files at `scripts/config`.

### For Artistic Styles Erasure and Celebrity Erasure
```sh
python scripts/train_sae_text.py --config_path "scripts/config/config_text.json"
```
### For Explicit Content Erasure

```sh
python scripts/train_sae_text.py --config_path "scripts/config/config_text_nsfw.json"
```

## Generating Samples

+ The deactivated model can be simply tested by running the following command to generate several images: 

For example, when unlearning celebrities
```sh

PRETRAINED_PATH="CompVis/stable-diffusion-v1-4"
PATH_TO_CHECKPOINTS="path/to/pretrained/sae/models"
MODEL_NAME="ItD-celeb"
NUM_SAMPLES=5
BATCH_SIZE=10
PROMPTS_DOMAIN="celebrity_remain" # can change to other prompts
BASE_SAVE_FOLDER="results"
SAVE_FOLDER="${BASE_SAVE_FOLDER}${MODEL_NAME}"
DEVICE="cuda:0"

python generate_images.py \
    --path_to_checkpoints "$PATH_TO_CHECKPOINTS" \
    --pretrained_path "$PRETRAINED_PATH" \
    --model_name "$MODEL_NAME" \
    --num_samples_per_prompt $NUM_SAMPLES \
    --batch_size_pipe $BATCH_SIZE \
    --prompts_domain "$PROMPTS_DOMAIN" \
    --save_folder "$SAVE_FOLDER" \
    --device "$DEVICE"
```

## Metrics Evaluation
+ Evaluate FID and KID
```sh
python run_fid.py \
    --dir1 "path/to/generated/images" \
    --dir2 "path/to/coco/dataset/images" \
    --fid
```
change the paras when calculate KID
```sh
python run_fid.py \
    --dir1 "path/to/generated/images" \
    --dir2 "path/to/SD/generated/images" \
    --kid
```

+ Evaluate CLIP score

Note that we locate images and their corresponding text using the **case number**.
```sh
python run_clipscore.py \
    --image_dir "path/to/generated/images" \
    --prompts_path "path/to/prompt/coco_30k.csv"
```

+ Evaluate GCD accuracy
You may need to set up an appropriate environment first. Refer to this [GCD](https://github.com/Shilin-LU/MACE/tree/main/metrics) repository for guidance. 
```sh
# conda activate GCD
cd eval_scripts/metrics/GCD/celeb-detection-oss/examples
sh run_GCD.sh
```
+ Evaluate NudeNet detection
```sh
python run_nudenet.py \
    --folder path/to/generated/images \
    --save_folder /path/to/save \
    --method ItD
```