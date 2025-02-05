import torch
from dataclasses import dataclass, field

@dataclass
class SAETrainingConfig:
    d_model: int
    n_dirs: int
    k: int
    block_name: str
    bs: int
    save_path_base: str
    datasets: str
    auxk: int = 256
    lr: float = 1e-4
    eps: float = 6.25e-10
    dead_toks_threshold: int = 10_000_000
    auxk_coef: float = 1/32
    
    @property
    def sae_name(self):
        return f'{self.block_name}_k{self.k}_hidden{self.n_dirs}_auxk{self.auxk}_bs{self.bs}_lr{self.lr}_dataset{self.datasets}'
    
    @property
    def save_path(self):
        return f'{self.save_path_base}/{self.block_name}_k{self.k}_hidden{self.n_dirs}_auxk{self.auxk}_bs{self.bs}_lr{self.lr}_dataset{self.datasets}'


@dataclass
class Config:
    saes: list[SAETrainingConfig]
    paths_to_latents: list[str]
    log_interval: int
    save_interval: int
    bs: int
    block_name: str
    wandb_project: str = 'sdxl_sae_train'
     
    # wandb_name: str = 'multiple_sae_text'

    def __init__(self, cfg_json, wandb_name):
        
        block_names = (
            [cfg_json['block_name'][0]] * len(cfg_json['sae_configs']) 
            if len(cfg_json['block_name']) == 1 
            else cfg_json['block_name']
        )
        
        self.saes = [
            SAETrainingConfig(
                **sae_cfg,
                block_name=block_names[i],
                bs=cfg_json['bs'],
                save_path_base=cfg_json['save_path_base'],
                datasets='_'.join(cfg_json["datasets"])
            )
            for i, sae_cfg in enumerate(cfg_json['sae_configs'])
        ]

        self.save_path_base = cfg_json['save_path_base']
        # self.paths_to_latents = cfg_json['paths_to_latents']
        self.log_interval = cfg_json['log_interval']
        self.save_interval = cfg_json['save_interval']
        self.bs = cfg_json['bs']
        self.block_name = cfg_json['block_name']
        self.output_or_diff = cfg_json['output_or_diff']
        self.epochs = cfg_json['epochs']
        
        self.wandb_name = wandb_name
        self.datasets = cfg_json['datasets']