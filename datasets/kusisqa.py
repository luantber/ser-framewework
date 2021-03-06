import os 
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from datasets import utils

class Kusisqa(Dataset):

    def __init__(self,annotations_file, audio_dir,  transform=None ):
        self.annotation_file = annotations_file
        self.audio_labels = pd.read_csv(self.annotation_file)
        self.audio_dir = audio_dir
        self.transform = transform

        # Cache Preload
        self.spec_cache = {}

        # Number of Classes
        self.number_classes = 8
        

    def set_transform(self,transform):
        self.transform = transform

    def __len__(self):
        return len(self.audio_labels)

    def get_sr(self):
        audio_path = os.path.join(self.audio_dir,self.audio_labels.iloc[0,0])
        wave, sr = torchaudio.load(audio_path)
        return sr

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir,self.audio_labels.iloc[idx,0])

        wave, sr = torchaudio.load(audio_path)
        wave = torch.mean(wave, dim=0, keepdim=True)[0]
        
        # only supporting over Spectrogram
        if self.transform:
            if audio_path in self.spec_cache:
                wave = self.spec_cache[audio_path]
            else:
                spec = utils.melspectrogram(wave,sr)
                wave = utils.powertodb(spec,sr)
                self.spec_cache[audio_path] = wave


        label = self.audio_labels.iloc[idx, 1]

        if self.transform:
            for t in self.transform:
                wave = t(wave,sr)

        return wave, label

