import os
import glob
import sys
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm


def file_list_generator(data_dir, machine, mode, domain='source', ext='wav'):
    """ Generate a list of audio files
    Args:
        data_dir (str): path to the dev/eval dataset directory
        machine (str): machine type
        mode (str): 'train' or 'test'
        domain (str): 'source' or 'target'. Invalid if mode == 'train'
        ext (str): audio file extension

    Returns:
        normal_files ([str]): list of paths to normal files
        anomaly_files ([str]): list of paths to anomaly files
    """
    assert machine in ['ToyCar', 'ToyTrain', 'fan', 'gearbox', 'pump', 'slider', 'valve']
    assert mode in ['train', 'test']

    data_dir = os.path.abspath(data_dir)
    prefix_normal = 'normal'
    prefix_anomaly = 'anomaly'

    if mode == 'train':
        data_subdir = os.path.join(data_dir, machine, mode)
    elif mode == 'test':
        assert domain in ['source', 'target']
        data_subdir = os.path.join(data_dir, machine, f'{domain}_{mode}')

    print(f'Generating file lists from: {data_subdir} ...')

    normal_query = os.path.abspath(f'{data_subdir}/*{prefix_normal}_*.{ext}')
    anomaly_query = os.path.abspath(f'{data_subdir}/*{prefix_anomaly}_*.{ext}')
    normal_files = sorted(glob.glob(normal_query))
    anomaly_files = sorted(glob.glob(anomaly_query))

    return normal_files, anomaly_files


def file_to_logmel(filename, machine, config):
    """Convert an audio file to log-mel spectrogram
    Args:
        filename (str): path to the audio file
        machine (str): ex. ToyCar (in the case where each machine has different processing configs)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_fft = config['machine_config'][machine]['n_fft']
    hop_length = config['machine_config'][machine]['hop_length']
    n_mels = config['machine_config'][machine]['n_mels']
    power = config['machine_config'][machine]['power']
    resample_rate = config['machine_config'][machine]['sample_rate']

    audio, sample_rate = torchaudio.load(filename)
    if sample_rate != resample_rate:
        resampler = torchaudio.transforms.Resample(
            sample_rate, resample_rate, dtype=audio.dtype
        )
        audio = resampler(audio)
    
    melspectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power,
    ).to(device)
    audio = audio.to(device)

    mel = melspectrogram(audio)
    log_mel = 20.0 / power * torch.log10(mel + sys.float_info.epsilon)

    return log_mel.cpu()


def normalize_time(x):
    d, h, w = x.shape
    x_mean = np.tile(np.mean(x, axis=(0,2)), (w, d)).T
    x_var = np.tile(np.std(x, axis=(0,2)), (w, d)).T
    x_norm = (x - x_mean) / x_var
    return x_norm

class DcaseDataset(Dataset):
    def __init__(self, files, machine, config, dim=1, dim_split=None, transform=None):
        self.machine = machine
        self.config = config
        self.transform = transform
        self.dim = dim              # the number of spectrogram slices
        self.slice_w = dim_split  # the width of a sliced spectrogram
        self.input_shape = config['machine_config'][machine]['input_shape']

        assert len(files) > 0, 'Empty file list'

        for i, filename in enumerate(tqdm(files, total=100)):
            log_mel = file_to_logmel(filename, self.machine, self.config)
            if self.slice_w is None:
                self.slice_w = log_mel.shape[-1] // self.dim

            if i == 0:
                features = np.zeros(  # zero-pad if the last slice width < slice_w
                    (len(files), self.dim, self.input_shape[0], self.slice_w),
                    np.float32,
                )
                sections = np.zeros(len(files), dtype=int)
            
            last_slice_w = self.input_shape[1]
            for d in range(self.dim - 1):
                features[i, d:d+1, :, :] = log_mel[:, :, d*self.slice_w:(d+1)*self.slice_w]
                last_slice_w -= self.slice_w

            last_split = (self.dim - 1) * self.slice_w
            features[i, self.dim-1:self.dim, :, :last_slice_w] = log_mel[:, :, last_split:last_split+last_slice_w]
            section_id = int(os.path.basename(filename)[8:10])  # section_00 -> 0
            sections[i] = section_id

        self.features = features
        self.sections = sections

        print(f'Input Feature Shape:', self.features.shape[1:])


    def __len__(self):
        return self.features.shape[0]  # the number of samples

    def __getitem__(self, idx):
        log_mel = self.features[idx, :]
        if self.config['machine_config'][self.machine]['normalize'] == 'time':
            log_mel = normalize_time(log_mel)
            
        if self.transform:
            log_mel = self.transform(log_mel)
        section_id = self.sections[idx]
        return log_mel, section_id


def get_dataloader(dataset, batch_size=32, val=False, shuffle=False):
    if not val:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        valid_dataloader = None
    else:
        train_size = int(len(dataset) * 0.9)
        file_indices = np.arange(len(dataset))
        random.shuffle(file_indices)  # shuffle sections

        dataloader = DataLoader(
            Subset(dataset, file_indices[:train_size]),
            batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        valid_dataloader = DataLoader(
            Subset(dataset, file_indices[train_size:]),
            batch_size=batch_size, shuffle=False, drop_last=False
        )
        print(f'train : validation = {len(dataloader)} : {len(valid_dataloader)}')

    print(f'Dataset size: {len(dataset)}, Batch size: {batch_size}, # iters: {len(dataloader)}')
    print('='*20)
    
    return dataloader, valid_dataloader