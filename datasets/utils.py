import torch
from torchaudio import transforms
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import RandomCrop
import matplotlib.pyplot as plt
import librosa 

""""
  Transformations
"""


def melspectrogram(wave,sr):
  transform = MelSpectrogram(sample_rate=sr, n_fft=1024, n_mels= 64)
  return transform(wave)

def powertodb(spec,sr):
  return torch.tensor(librosa.power_to_db(spec))

def randomcrop(spec,sr):
  r = RandomCrop((64,128))
  return r(spec)

def d_melspectrogram(wave,label):
  return melspectrogram(wave,48000),label

def d_powertodb(wave,label):
  return powertodb(wave,48000) , label

"""
  Collate , etc 
"""
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [ torch.tensor(item).T for item in batch]
    print(batch[0].shape)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    # return batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for wave, label in batch:
        tensors += [wave]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.tensor(targets)

    return tensors, targets


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

