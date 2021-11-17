import os 
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from datasets import utils

class Ravdess(Dataset):
    def __init__(self,annotations_file, audio_dir,  transform=None ):
        self.annotation_file = annotations_file
        self.audio_labels = pd.read_csv(self.annotation_file)
        self.audio_dir = audio_dir
        self.transform = transform

        # Cache Reasons Workers > 0 

        # if shared_audios_cache!=None and shared_spec_cache!=None:
        #     self.audio_cache = shared_audios_cache
        #     self.spec_cache = shared_spec_cache
        #     self.using_cache = True
        # else:
        #     self.using_cache = False

        self.audio_cache = {}
        self.spec_cache = {}

    def __len__(self):
         return len(self.audio_labels)

    def get_sr(self):
        audio_path = os.path.join(self.audio_dir,self.audio_labels.iloc[0,0])
        wave, sr = torchaudio.load(audio_path)
        return sr


    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir,self.audio_labels.iloc[idx,0])

        if audio_path in self.audio_cache:
            wave,sr = self.audio_cache[audio_path]
            wave = torch.mean(wave, dim=0, keepdim=True)[0]
        else:
            wave, sr = torchaudio.load(audio_path)
            wave = torch.mean(wave, dim=0, keepdim=True)[0]
            self.audio_cache[audio_path] = wave, sr
        
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

