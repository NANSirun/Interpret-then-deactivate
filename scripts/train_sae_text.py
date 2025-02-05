import sys

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from transformers import CLIPTokenizer
from diffusers.optimization import get_scheduler
from src.SDLens import HookedTextEncoder

from src.SAE.sae import SparseAutoencoder, unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
from src.SAE.sae_utils import SAETrainingConfig, Config

from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional, List
import json
import wandb
import argparse

# change to your own key
# os.environ["WANDB_API_KEY"] = ''


def weighted_average(points: torch.Tensor, weights: torch.Tensor):
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)


@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore

    return (norms * weights).sum()


def compute_geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
    do_log: bool = False,
):
    """
    :param points: ``torch.Tensor`` of shape ``(n, d)``
    :param weights: Optional ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
        Equivalently, this is a smoothing parameter. Default 1e-6.
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :param do_log: If true will return a log of function values encountered through the course of the algorithm
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list (None if do_log is false).
    """
    with torch.no_grad():

        if weights is None:
            weights = torch.ones((points.shape[0],), device=points.device)
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        if do_log:
            logs = [objective_value]
        else:
            logs = None

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if logs is not None:
                logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=(
            "function value converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
        logs=logs,
    )

def maybe_transpose(x):
    return x.T if not x.is_contiguous() and x.T.is_contiguous() else x


RANK = 0

class Logger:
    def __init__(self, sae_name, **kws):
        self.vals = {}
        self.enabled = (RANK == 0) and not kws.pop("dummy", False)
        self.sae_name = sae_name

    def logkv(self, k, v):
        if self.enabled:
            self.vals[f'{self.sae_name}/{k}'] = v.detach() if isinstance(v, torch.Tensor) else v
        return v

    def dumpkvs(self, step):
        if self.enabled:
            wandb.log(self.vals, step=step)
            self.vals = {}
    

class FeaturesStats:
    def __init__(self, dim, logger, device):
        self.dim = dim
        self.logger = logger
        self.device = device
        self.reinit()
        

    def reinit(self):
        self.n_activated = torch.zeros(self.dim, dtype=torch.long, device=self.device)
        self.n = 0
    
    def update(self, inds):
        self.n += inds.shape[0]
        inds = inds.flatten().detach()
        self.n_activated.scatter_add_(0, inds, torch.ones_like(inds))

    def log(self):
        self.logger.logkv('activated', (self.n_activated / self.n + 1e-9).log10().cpu().numpy())

def training_loop_(
    dataloader,
    model,
    aes, 
    loss_fn, 
    log_interval, 
    save_interval,
    device,
    loggers,
    sae_cfgs,
    epochs,
):  
    sae_packs = []
    for ae, cfg, logger in zip(aes, sae_cfgs, loggers):
        pbar = tqdm(unit=" steps", desc="Training Loss: ")
        fstats = FeaturesStats(ae.n_dirs, logger, device)
        opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr, eps=cfg.eps, fused=True)
        lr_scheduler =  get_scheduler(
            "constant_with_warmup",
            optimizer=opt,
            num_warmup_steps=1000,
            num_training_steps=len(dataloader) * epochs
        )
        sae_packs.append((ae, cfg, logger, pbar, fstats, opt, lr_scheduler))
                
    step = 0
    for _ in range(epochs):
        for batch in dataloader:
            # train_acts_iter = model(batch).reshape(-1, 768)
            cache = model(batch)

            for ae, cfg, logger, pbar, fstats, opt, lr_scheduler in sae_packs:
                train_acts_iter = cache['output'][cfg.block_name].float() # - cache['input'][cfg.block_name].float()
                train_acts_iter = train_acts_iter.reshape(-1, cfg.d_model)
                
                recons, info = ae(train_acts_iter)
                loss = loss_fn(ae, cfg, train_acts_iter, recons, info, logger)

                fstats.update(info['inds'])
                
                bs = train_acts_iter.shape[0]
                logger.logkv('not-activated 1e4', (ae.stats_last_nonzero > 1e4 / bs).mean(dtype=float).item())
                logger.logkv('not-activated 1e6', (ae.stats_last_nonzero > 1e6 / bs).mean(dtype=float).item())
                logger.logkv('not-activated 1e7', (ae.stats_last_nonzero > 1e7 / bs).mean(dtype=float).item())

                logger.logkv('explained variance', explained_variance(recons, train_acts_iter))
                logger.logkv('l2_div', (torch.linalg.norm(recons, dim=1) / torch.linalg.norm(train_acts_iter, dim=1)).mean())

                if (step + 1) % log_interval == 0:
                    fstats.log()
                    fstats.reinit()
                
                if (step + 1) % save_interval == 0:
                    ae.save_to_disk(f"{cfg.save_path}/{step + 1}")

                loss.backward()

                unit_norm_decoder_(ae)
                unit_norm_decoder_grad_adjustment_(ae)

                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
                logger.dumpkvs(step)

                pbar.set_description(f"Training Loss {loss.item():.4f}")
                pbar.update(1)
                
            step = step + 1


    for ae, cfg, logger, pbar, fstats, opt, lr_scheduler in sae_packs:
        pbar.close()
        ae.save_to_disk(f"{cfg.save_path}/final")


def init_from_data_(ae, stats_acts_sample, device):
    ae.pre_bias.data = (
        compute_geometric_median(stats_acts_sample[:32768].float().cpu()).median.float().to(device)
    )


def mse(recons, x):
    return ((recons - x) ** 2).mean()

def normalized_mse(recon: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    # only used for auxk
    xs_mu = xs.mean(dim=0)

    loss = mse(recon, xs) / mse(
        xs_mu[None, :].broadcast_to(xs.shape), xs
    )

    return loss

def explained_variance(recons, x):
    # Compute the variance of the difference
    diff = x - recons
    diff_var = torch.var(diff, dim=0, unbiased=False)

    # Compute the variance of the original tensor
    x_var = torch.var(x, dim=0, unbiased=False)

    # Avoid division by zero
    explained_var = 1 - diff_var / (x_var + 1e-8)

    return explained_var.mean()

# def main(dataset_name=["file_coco", "file_i2p", "file_celebrity", "file_imagenet"], device='cuda:3'):
def train_sae(config_path, device='cuda:0'):
    def model_run(batch,):
        input_ids = tokenizer(batch['prompt'], return_tensors='pt', padding=True,  truncation=True, max_length=tokenizer.model_max_length)['input_ids'].to(text_encoder.device)
        with torch.no_grad():
            kwargs['input_ids'] = input_ids
            _, cache = text_encoder.run_with_cache(**kwargs)
        return cache

    cfg = Config(json.load(open(config_path)), wandb_name='sae_text_nsfw')    
    blocks = cfg.block_name

    
    dataset_files = {

        "imagenet":"prompt/imagenet1k-id2label.csv",
        "celebrity":"prompt/celebrity/celebrity_100_concepts.csv", # 5000 prompts
        "style":"prompt/style/art_100_concepts.csv", # 5000 prompts
        "coco": "prompt/coco_30k.csv",
        "i2p": "prompts/i2p.csv",
        "diffusiondb_nsfw":"prompt/diffusiondb/diffusiondb10K_nsfw.csv",        
        
    }

    datasets = []
    for name in cfg.datasets:
        file_dataset = load_dataset("csv", split="train", data_files=dataset_files[name])
        file_dataset = file_dataset.remove_columns([col for col in file_dataset.column_names if col != "prompt"])
        datasets.append(file_dataset)
        
    datasets = concatenate_datasets(datasets)
    dataloader = DataLoader(datasets, batch_size=cfg.bs, shuffle=True, num_workers=5,)
    
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = HookedTextEncoder.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    text_encoder.to(device)

    kwargs = {
        'positions_to_cache': blocks,
        'save_input': False,
        'save_output': True,
    }

    aes = [
        SparseAutoencoder(
            n_dirs_local=sae.n_dirs,
            d_model=sae.d_model,
            k=sae.k,
            auxk=sae.auxk,
            dead_steps_threshold=sae.dead_toks_threshold // cfg.bs,
        ).to(device)
        for sae in cfg.saes
    ]
    
    
    stats_acts_sample = {block: [] for block in blocks}
    
    prompt = {'prompt': datasets['prompt'][:500]}
    cache = model_run(prompt)
        
    for block in blocks:
        stats_acts_sample[block] = cache['output'][block].float()  #- cache['input'][block].float()
    
    for ae, cfg_sae in zip(aes, cfg.saes):
        assert ae.d_model == cfg_sae.d_model
        
        stats_acts_sample_block = stats_acts_sample[cfg_sae.block_name].reshape(-1, ae.d_model)
        init_from_data_(ae, stats_acts_sample_block, device)
    
        mse_scale = (
            1 / ((stats_acts_sample_block.float().mean(dim=0) - stats_acts_sample_block.float()) ** 2).mean()
        )
        mse_scale = mse_scale.item()
        cfg_sae.mse_scale = mse_scale
        
    
    del stats_acts_sample
    del stats_acts_sample_block

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        # mode="offline"
    )

    loggers = [Logger(
        sae_name=cfg_sae.sae_name,
        dummy=False,
    ) for cfg_sae in cfg.saes]

    

    training_loop_(
        dataloader,
        model_run,
        aes,
        lambda ae, cfg_sae, flat_acts_train_batch, recons, info, logger: (
            # MSE
            logger.logkv("train_recons", mse_scale * mse(recons, flat_acts_train_batch))
            # AuxK
            + logger.logkv(
                "train_maxk_recons",
                cfg_sae.auxk_coef
                * normalized_mse(
                    ae.decode_sparse(
                        info["auxk_inds"],
                        info["auxk_vals"],
                    ),
                    flat_acts_train_batch - recons.detach() + ae.pre_bias.detach(),
                ).nan_to_num(0),
            )
        ),
        device=device,
        sae_cfgs = cfg.saes,
        loggers=loggers,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        epochs=cfg.epochs
    )
            
def main():
    parser = argparse.ArgumentParser(description="Main program to generate images using diffuser pipe.")
    
    parser.add_argument('--config_path', type=str,default="scripts/config/config_text_nsfw.json",
                        help='Path to the pretrained model')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device type to use ("cuda" or "cpu") (default: "cuda" if available else "cpu")')
    
    args = parser.parse_args()
    
    device=args.device
    config_path = args.config_path
    
    train_sae(config_path, device)
    

if __name__ == "__main__":
    main()