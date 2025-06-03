# ---------- music_dataset.py ----------
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class ChordMelodyDataset(Dataset):
    def __init__(self, data_path, split='train', val_ratio=0.1, seed=42):
        data = torch.load(data_path)
        self.chords = data['chord_sequences']  # [N, L]
        self.melodies = data['melody_sequences']  # [N, L]

        assert len(self.chords) == len(self.melodies)

        # Set split
        N = len(self.chords)
        indices = list(range(N))
        random.seed(seed)
        random.shuffle(indices)

        val_size = int(N * val_ratio)
        if split == 'train':
            split_ids = indices[val_size:]
        elif split == 'val':
            split_ids = indices[:val_size]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.chord = self.chords[split_ids]
        self.melody = self.melodies[split_ids]

    def __len__(self):
        return len(self.chord)

    def __getitem__(self, idx):
        return self.chord[idx], self.melody[idx]
