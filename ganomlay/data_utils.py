import os
import sys
import glob
import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import itertools
import re


def select_dirs(config, mode):
    """Get directory paths according to mode.
    
    Args:
        config (dict):  baseline.yaml data

    Returns :
        A list of base directories of dev_data or eval_data
    """
    assert mode in ['dev', 'eval']

    if mode == 'dev':
        print("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=config["dev_directory"]))
    elif mode == 'eval':
        print("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=config["eval_directory"]))

    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]

    return dirs


def get_section_names(target_dir, dir_name, ext="wav"):
    """
    Get section name (almost equivalent to machine ID).

    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath(
        "{target_dir}/{dir_name}/*.{ext}".format(
            target_dir=target_dir, dir_name=dir_name, ext=ext
        )
    )
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(
        list(
            set(
                itertools.chain.from_iterable(
                    [re.findall("section_[0-9][0-9]", ext_id) for ext_id in file_paths]
                )
            )
        )
    )
    return section_names


def file_list_generator(target_dir, section_name, dir_name, mode, ext='wav'):
    """Get list of audio file paths
    Args:
        target_dir (str): base directory path
        section_name (str): section name of audio file in <<dir_name>> directory
        dir_name (str): sub directory name
        mode (str): 'dev' or 'eval'
        ext (str): audio file extension (default: 'wav')
    
    Returns:
        files (list[str]): list of audio file paths
        labels (list[bool] or None):
                if mode == 'dev': normal 0, anomlay 1
                if mode == 'eval': None
    """
    assert mode in ['dev', 'eval']

    print("target_dir : %s" % (target_dir + "_" + section_name))

    prefix_normal = "normal"    # normal directory name
    prefix_anomaly = "anomaly"  # anomaly directory name

    # development
    if mode == 'dev':
        query = os.path.abspath(
            # example:
            # dev_data/ToyCar/train/section_00_source_train_normal_0025_A1Spd28VMic1.wav
            f'{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}'
        )
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath(
            f'{target_dir}/{dir_name}/{section_name}_*_{prefix_anomaly}_*.{ext}'
        )
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        print(f'Number of audio files : {len(files)}')
        if len(files) == 0:
            print(f'No {ext} files!!')

    elif mode == 'eval':
        query = os.path.abspath(
            f'{target_dir}/{dir_name}/{section_name}_*.{ext}'
            )
        files = sorted(glob.glob(query))
        labels = None

        print(f'Number of audio files : {len(files)}')
        if len(files) == 0:
            print(f'No {ext} files!!')

    return files, labels


def extract_feature(file_name, machine_config):
    """Extract feature vectors
    Args:
        file_name (str): target audio file path
        machine_config (dict): machine-specific configs for getting mel spectrograms
    
    Returns:
        vector array of shape (n_vectors, dims)
    """
    n_mels = machine_config['n_mels']
    n_frames = machine_config['n_frames']
    n_fft = machine_config['n_fft']
    hop_length = machine_config['hop_length']
    power = machine_config['power']

    # calculate the number of dimensions
    dims = n_mels * n_frames

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # generate melspectrogram
    audio, sample_rate = torchaudio.load(file_name)

    # TODO: change sampling rate
    if sample_rate != machine_config['input_samples']:
        resampler = torchaudio.transforms.Resample(sample_rate,
                                                   machine_config['input_samples'],
                                                   dtype=audio.dtype)
        audio = resampler(audio)

    melspectrogram = torchaudio.transforms.MelSpectrogram(
                                                        #   sample_rate=sample_rate,
                                                          n_fft=n_fft,
                                                          hop_length=hop_length,
                                                          n_mels=n_mels,
                                                          power=power).to(device)
    audio = audio.to(device)
    mel_spectrogram = melspectrogram(audio)  # (channel, n_mels, n_frames)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = (
        20.0 / power * torch.log10(
            torch.maximum(
                mel_spectrogram, torch.ones_like(mel_spectrogram) * sys.float_info.epsilon
                )
            )
    )

    return log_mel_spectrogram.cpu()

    # # calculate total vector size
    # n_vectors = log_mel_spectrogram.shape[-1] - n_frames + 1

    # # skip too short clips
    # if n_vectors < 1:
    #     return torch.empty((0, 0, dims))

    # # generate feature vectors by concatenating multiframes
    # vectors = torch.zeros((n_vectors, dims),
    #                        dtype=log_mel_spectrogram.dtype,
    #                        device=log_mel_spectrogram.device)
    # for frame in range(n_frames):
    #     vectors[:, n_mels * frame : n_mels * (frame + 1)] = log_mel_spectrogram[..., frame : frame + n_vectors].transpose(-1,-2).squeeze(0)

    # return vectors.cpu()


class DcaseDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, config, machine_config, transform=None):
        self.transform = transform
        self.config = config
        self.machine_config = machine_config

        for file_id, (file_name, label) in tqdm(enumerate(zip(files, labels))):
            features = extract_feature(file_name, self.machine_config)
            # features = features[:: self.machine_config['n_hop_frames'], :]

            sid = int(os.path.basename(file_name)[8:10])  # section id

            if file_id == 0:
                # dataset = np.zeros(
                #     (
                #         features.shape[0] * len(files),
                #         self.machine_config['n_mels'] * self.machine_config['n_frames'],
                #     ),
                #     np.float32,
                # )
                # section_ids = np.zeros((features.shape[0] * len(files)), dtype=int)
                # anomaly_label = np.zeros((features.shape[0] * len(files)), dtype=int)

                dataset = np.zeros(
                    (
                        len(files),
                        features.shape[0],  # channel
                        self.machine_config['n_mels'],
                        self.machine_config['n_frames'],
                    ),
                    np.float32,
                )
                section_ids = np.zeros(len(files), dtype=int)
                anomaly_label = np.zeros(len(files), dtype=int)

            # dataset[
            #     features.shape[0] * file_id : features.shape[0] * (file_id + 1), :
            # ] = features
            
            # section_ids[
            #     features.shape[0] * file_id : features.shape[0] * (file_id + 1)
            # ] = sid

            # anomaly_label[
            #     features.shape[0] * file_id : features.shape[0] * (file_id + 1)
            # ] = label

            dataset[file_id : (file_id + 1), :] = features
            section_ids[file_id : (file_id + 1)] = sid
            anomaly_label[file_id : (file_id + 1)] = label

        self.feat_data = dataset.reshape(
            (
                dataset.shape[0],
                1,  # number of channels
                self.machine_config['n_mels'],
                self.machine_config['n_frames'],
            )
        )
        self.section_ids = section_ids
        self.anomaly_label = anomaly_label

        # train_size = int(len(dataset) * (1.0 - self.config['training']['validation_split']))
        # val_size = int(len(dataset) * self.config['training']['validation_split'])
        # print(f'train size: {train_size}, val_size: {val_size}')
        print(f'Feature Shape:', self.feat_data.shape)

    def __len__(self):
        return self.feat_data.shape[0]  # the number of samples

    def __getitem__(self, idx):
        sample = self.feat_data[idx, :]
        if self.transform:
            sample = self.transform(sample)
        section_id = self.section_ids[idx]
        anomaly = self.anomaly_label[idx]
        return sample, section_id, anomaly


def get_dataloader(dataset, config, machine_type):
    """Make dataloader from dataset for training"""
    train_size = int(len(dataset) * (1.0 - config['training']['validation_split']))
    val_size = int(len(dataset) * config['training']['validation_split'])
    
    data_loader_train = torch.utils.data.DataLoader(
        Subset(dataset, list(range(0, train_size))),
        batch_size=config[machine_type]['batch_size'],
        shuffle=config['training']['shuffle'],
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        Subset(dataset, list(range(train_size, len(dataset)))),
        batch_size=config[machine_type]['batch_size'],
        shuffle=False,
        drop_last=False,
    )

    print(f'train size: {train_size}, val_size: {val_size}')

    return data_loader_train, data_loader_val


##### Dataset for eval mode ####
class DcaseEvalDataset(torch.utils.data.Dataset):
    def __init__(self, files, config, machine_config, transform=None):
        self.transform = transform
        self.config = config
        self.machine_config = machine_config

        for file_id, file_name in tqdm(enumerate(files)):
            features = extract_feature(file_name, self.machine_config)
            features = features[:: self.machine_config['n_hop_frames'], :]

            if file_id == 0:
                dataset = np.zeros(
                    (
                        features.shape[0] * len(files),
                        self.machine_config['n_mels'] * self.machine_config['n_frames'],
                    ),
                    np.float32,
                )
            dataset[
                features.shape[0] * file_id : features.shape[0] * (file_id + 1), :
            ] = features
            

        self.feat_data = dataset.reshape(
            (
                dataset.shape[0],
                # 1,  # number of channels
                self.machine_config['n_frames'],
                self.machine_config['n_mels'],
            )
        )
        print(f'test size:', len(dataset))

    def __len__(self):
        return self.feat_data.shape[0]  # the number of samples

    def __getitem__(self, idx):
        sample = self.feat_data[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_eval_dataloader(dataset, config, machine_type):
    data_loader_test = torch.utils.data.DataLoader(
        Subset(dataset, list(range(len(dataset)))),
        batch_size=config[machine_type]['batch_size'],
        shuffle=False,
        drop_last=False,
    )

    print(f'test size: {int(len(dataset))}')
    
    return data_loader_test