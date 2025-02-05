from diffusers import DDPMScheduler
import torch

class HookedNoiseScheduler:
    scheduler: DDPMScheduler
    pre_hooks: list
    post_hooks: list

    def __init__(self, scheduler):
        object.__setattr__(self, 'scheduler', scheduler)
        object.__setattr__(self, 'pre_hooks', [])
        object.__setattr__(self, 'post_hooks', [])
    
    def step(
        self,
        model_output, timestep, sample, generator, return_dict
    ):
        assert return_dict == False, "return_dict == True is not implemented"
        for hook in self.pre_hooks:
            hook_output = hook(model_output, timestep, sample, generator)
            if hook_output is not None:
                model_output, timestep, sample, generator = hook_output

        (pred_prev_sample, ) = self.scheduler.step(model_output, timestep, sample, generator, return_dict)
        
        for hook in self.post_hooks:
            hook_output = hook(pred_prev_sample)
            if hook_output is not None:
                pred_prev_sample = hook_output

        return (pred_prev_sample, ) 

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def __setattr__(self, name, value):
        if name in {'scheduler', 'pre_hooks', 'post_hooks'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.scheduler, name, value)