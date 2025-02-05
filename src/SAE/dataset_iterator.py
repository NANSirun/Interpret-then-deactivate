import webdataset as wds
import os
import torch

class ActivationsDataloader:
    def __init__(self, paths_to_datasets, block_name, batch_size, output_or_diff='diff', num_in_buffer=50):
        assert output_or_diff in ['diff', 'output'], "Provide 'output' or 'diff'"

        self.dataset = wds.WebDataset(
            [os.path.join(path_to_dataset, f"{block_name}.tar")
             for path_to_dataset in paths_to_datasets]
        ).decode("torch")
        self.iter = iter(self.dataset)
        self.buffer = None
        self.pointer = 0
        self.num_in_buffer = num_in_buffer
        self.output_or_diff = output_or_diff
        self.batch_size = batch_size
        self.one_size = None

    def renew_buffer(self, to_retrieve):
        to_merge = []
        if self.buffer is not None and self.buffer.shape[0] > self.pointer:
            to_merge = [self.buffer[self.pointer:].clone()]
        del self.buffer
        for _ in range(to_retrieve):
            sample = next(self.iter)
            latents = sample['output.pth'] if self.output_or_diff == 'output' else sample['diff.pth']
            latents = latents.permute((0, 1, 3, 4, 2)) if self.output_or_diff == 'diff' else latents
            latents = latents.reshape((-1, latents.shape[-1]))
            to_merge.append(latents.to('cuda'))
            self.one_size = latents.shape[0]
            
        if to_merge:
            self.buffer = torch.cat(to_merge, dim=0)
            shuffled_indices = torch.randperm(self.buffer.shape[0])
            self.buffer = self.buffer[shuffled_indices]
        else:
            self.buffer = None
        self.pointer = 0

    def iterate(self):
        while True:
            if self.buffer == None or self.buffer.shape[0] - self.pointer < self.num_in_buffer * self.one_size * 4 // 5:
                try:
                    to_retrieve = self.num_in_buffer if self.buffer is None else self.num_in_buffer // 5
                    self.renew_buffer(to_retrieve)
                except:
                    # self.iter = iter(self.dataset)
                    self.buffer = None
                    # self.pointer = 0
                    # break

            if self.buffer is not None and self.buffer.shape[0] - self.pointer >= self.batch_size:
                batch = self.buffer[self.pointer:self.pointer + self.batch_size]
                self.pointer += self.batch_size
                yield batch
            else:
                self.iter = iter(self.dataset)
                self.pointer = 0  # Drop the last epoch and renew buffer


