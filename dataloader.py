import os

import torch


class DataLoader:
    def __init__(self,
                 B,
                 process_rank,
                 num_processes,
                 split, device,
                 master_process):
        self.B = B  # Number of examples per batch
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device
        self.master_process = master_process
        
        self.audio_file_path = r"data/audio"
        self.labels_file_path = r"data/labels"
        assert os.path.exists(self.audio_file_path), (
            f"File{self.audio_file_path} not found")
        assert os.path.exists(self.labels_file_path), (
            f"File {self.labels_file_path} not found")

        self.audio_shards = sorted(
            [s for s in os.listdir(self.audio_file_path) if split in s])
        self.label_shards = sorted(
            [s for s in os.listdir(self.labels_file_path) if split in s])
        self.audio_shards = [os.path.join(self.audio_file_path, s)
                             for s in self.audio_shards]
        self.label_shards = [os.path.join(self.labels_file_path, s)
                             for s in self.label_shards]

        assert len(self.audio_shards) > 0, (
            f"no audio shards found for split {split}")
        assert len(self.label_shards) > 0, (
            f"no label shards found for split {split}")
        assert len(self.audio_shards) == len(self.label_shards), (
            f"missing one or more shards for {split} split")
        
        if master_process:
            print(f"found {len(self.audio_shards)} "
                  f"audio shards for {split} split")
            print(f"found {len(self.label_shards)} "
                  f"label shards for {split} split")
        
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.spectrograms = torch.load(self.audio_shards[self.current_shard],
                                       map_location=self.device, 
                                       weights_only=True)
        self.labels = torch.load(self.label_shards[self.current_shard], 
                                 map_location=self.device, 
                                 weights_only=True).to(dtype=torch.long)
        self.current_position = self.B * self.process_rank

    def next_batch(self):
        # Spectrograms
        encoder_x = self.spectrograms[
            self.current_position : self.current_position+self.B
            ].to(self.device) # (64, 80, 101)
        
        # Labels
        y = self.labels[
            self.current_position : self.current_position+self.B
            ].to(self.device)

       # advance the current position in the spectrograms and labels tensors
        self.current_position += self.B * self.num_processes

        # if loading the next batch would be out of bounds, move to next shard
        if (
            self.current_position
            + (self.B * self.num_processes + 1)
            > len(self.spectrograms)
        ):

            self.current_shard = (
                (self.current_shard + 1) % len(self.audio_shards)
            )
            self.spectrograms = torch.load(
                self.audio_shards[self.current_shard], 
                map_location=self.device, 
                weights_only=True)
            self.labels = torch.load(
                self.label_shards[self.current_shard],
                  map_location=self.device, 
                  weights_only=True).to(dtype=torch.long)
            self.current_position = self.B * self.process_rank

        return encoder_x, y