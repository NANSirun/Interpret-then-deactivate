import einops
from diffusers import StableDiffusionXLPipeline, IFPipeline, StableDiffusionPipeline
from typing import List, Dict, Callable, Union
import torch
from .hooked_scheduler import HookedNoiseScheduler

def retrieve(io):
    if isinstance(io, tuple):
        # if len(io) == 1:
        return io[0]
        # else:
            # raise ValueError("A tuple should have length of 1")
    elif isinstance(io, torch.Tensor):
        return io
    else:
        raise ValueError("Input/Output must be a tensor, or 1-element tuple")


class HookedDiffusionAbstractPipeline:
    parent_cls = None
    pipe = None  
    
    def __init__(self, pipe: parent_cls, use_hooked_scheduler: bool = False):
        if use_hooked_scheduler:
            pipe.scheduler = HookedNoiseScheduler(pipe.scheduler)
        self.__dict__['pipe'] = pipe
        self.use_hooked_scheduler = use_hooked_scheduler

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(cls.parent_cls.from_pretrained(*args, **kwargs))


    def run_with_hooks(self, 
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]], 
        **kwargs
    ):
        '''
        Run the pipeline with hooks at specified positions.
        Returns the final output.

        Args:
            *args: Arguments to pass to the pipeline.
            position_hook_dict: A dictionary mapping positions to hooks.
                The keys are positions in the pipeline where the hooks should be registered.
                The values are either a single hook or a list of hooks to be registered at the specified position.
                Each hook should be a callable that takes three arguments: (module, input, output).
            **kwargs: Keyword arguments to pass to the pipeline.
        '''
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            output = self.pipe(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []
        
        return output

    def run_with_cache(self, 
        *args,
        positions_to_cache: List[str],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs
    ):
        '''
        Run the pipeline with caching at specified positions.

        This method allows you to cache the intermediate inputs and/or outputs of the pipeline 
        at certain positions. The final output of the pipeline and a dictionary of cached values 
        are returned.

        Args:
            *args: Arguments to pass to the pipeline.
            positions_to_cache (List[str]): A list of positions in the pipeline where intermediate 
                inputs/outputs should be cached.
            save_input (bool, optional): If True, caches the input at each specified position. 
                Defaults to False.
            save_output (bool, optional): If True, caches the output at each specified position. 
                Defaults to True.
            **kwargs: Keyword arguments to pass to the pipeline.

        Returns:
            final_output: The final output of the pipeline after execution.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys are the specified positions 
                and values are dictionaries containing the cached 'input' and/or 'output' at each position, 
                depending on the flags `save_input` and `save_output`.
        '''
        cache_input, cache_output = dict() if save_input else None, dict() if save_output else None
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output) for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict['input'] = cache_input
        
        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict['output'] = cache_output
        return output, cache_dict

    def run_with_hooks_and_cache(self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        positions_to_cache: List[str] = [],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs
    ):
        '''
        Run the pipeline with hooks and caching at specified positions.

        This method allows you to register hooks at certain positions in the pipeline and 
        cache intermediate inputs and/or outputs at specified positions. Hooks can be used 
        for inspecting or modifying the pipeline's execution, and caching stores intermediate 
        values for later inspection or use.

        Args:
            *args: Arguments to pass to the pipeline.
            position_hook_dict Dict[str, Union[Callable, List[Callable]]]: 
                A dictionary where the keys are the positions in the pipeline, and the values 
                are hooks (either a single hook or a list of hooks) to be registered at those positions.
                Each hook should be a callable that accepts three arguments: (module, input, output).
            positions_to_cache (List[str], optional): A list of positions in the pipeline where 
                intermediate inputs/outputs should be cached. Defaults to an empty list.
            save_input (bool, optional): If True, caches the input at each specified position. 
                Defaults to False.
            save_output (bool, optional): If True, caches the output at each specified position. 
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to the pipeline.

        Returns:
            final_output: The final output of the pipeline after execution.
            cache_dict (Dict[str, Dict[str, Any]]): A dictionary where keys are the specified positions 
                and values are dictionaries containing the cached 'input' and/or 'output' at each position, 
                depending on the flags `save_input` and `save_output`.
        '''
        cache_input, cache_output = dict() if save_input else None, dict() if save_output else None
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output) for position in positions_to_cache
        ]
        
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict['input'] = cache_input

        if save_output:
            for position, block in cache_output.items():
                cache_output[position] = torch.stack(block, dim=1)
            cache_dict['output'] = cache_output
        
        return output, cache_dict

    
    def _locate_block(self, position: str):
        '''
        Locate the block at the specified position in the pipeline.
        '''
        block = self.pipe
        for step in position.split('.'):
            if step.isdigit():
                step = int(step)
                block = block[step]
            else:
                block = getattr(block, step)
        return block
    

    def _register_cache_hook(self, position: str, cache_input: Dict, cache_output: Dict):

        if position.endswith('$self_attention') or position.endswith('$cross_attention'):
            return self._register_cache_attention_hook(position, cache_output)

        if position == 'noise':
            def hook(model_output, timestep, sample, generator):
                if position not in cache_output:
                    cache_output[position] = []
                cache_output[position].append(sample)
            
            if self.use_hooked_scheduler:
                self.pipe.scheduler.post_hooks.append(hook)
            else:
                raise ValueError('Cannot cache noise without using hooked scheduler')
            return

        block = self._locate_block(position)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                cache_input[position].append(retrieve(input))
            
            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                cache_output[position].append(retrieve(output))

        return block.register_forward_hook(hook, with_kwargs=True)

    def _register_cache_attention_hook(self, position, cache):
        attn_block = self._locate_block(position.split('$')[0])
        if position.endswith('$self_attention'):
            attn_block = attn_block.attn1
        elif position.endswith('$cross_attention'):
            attn_block = attn_block.attn2
        else:
            raise ValueError('Wrong attention type')

        def hook(module, args, kwargs, output):
            hidden_states = args[0]
            encoder_hidden_states = kwargs['encoder_hidden_states']
            attention_mask = kwargs['attention_mask']
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn_block.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = attn_block.to_q(hidden_states)


            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn_block.norm_cross is not None:
                encoder_hidden_states = attn_block.norm_cross(encoder_hidden_states)

            key = attn_block.to_k(encoder_hidden_states)
            value = attn_block.to_v(encoder_hidden_states)

            query = attn_block.head_to_batch_dim(query)
            key = attn_block.head_to_batch_dim(key)
            value = attn_block.head_to_batch_dim(value)

            attention_probs = attn_block.get_attention_scores(query, key, attention_mask)
            attention_probs = attention_probs.view(
                batch_size, 
                attention_probs.shape[0] // batch_size,
                attention_probs.shape[1],
                attention_probs.shape[2]
            )
            if position not in cache:
                cache[position] = []
            cache[position].append(attention_probs)
        
        return attn_block.register_forward_hook(hook, with_kwargs=True) 

    def _register_general_hook(self, position, hook):
        if position == 'scheduler_pre':
            if not self.use_hooked_scheduler:
                raise ValueError('Cannot register hooks on scheduler without using hooked scheduler')
            self.pipe.scheduler.pre_hooks.append(hook)
            return
        elif position == 'scheduler_post':
            if not self.use_hooked_scheduler:
                raise ValueError('Cannot register hooks on scheduler without using hooked scheduler')
            self.pipe.scheduler.post_hooks.append(hook)
            return

        block = self._locate_block(position)
        return block.register_forward_hook(hook)

    def to(self, *args, **kwargs):
        self.pipe = self.pipe.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def __setattr__(self, name, value):
        return setattr(self.pipe, name, value)

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


class HookedStableDiffusionXLPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = StableDiffusionXLPipeline
    
class HookedStableDiffusionPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = StableDiffusionPipeline

class HookedIFPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = IFPipeline
