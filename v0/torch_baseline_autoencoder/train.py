"""
PyTorch script for model training (Autoencoder).

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard library imports.
import os
import sys

# Related third party imports.
import numpy
import scipy.stats
import torch
import torch.utils.data
from torch import optim
from torch.utils.data.dataset import Subset
from torchinfo import summary

import joblib

# Local application/library specific imports.
import util
from pytorch_model import AutoEncoder

# Load configuration from YAML file.
CONFIG = util.load_yaml("./config.yaml")

# String constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DcaseDataset(torch.utils.data.Dataset):
    """
    Prepare dataset.
    """

    def __init__(self, files, transform=None):

        self.transform = transform

        for file_id, file_name in enumerate(files):

            # shape = (#frames, #dims)
            features = util.extract_feature(file_name, config=CONFIG["feature"])
            features = features[:: CONFIG["feature"]["n_hop_frames"], :]

            if file_id == 0:
                # shape = (#total frames over all audio files, #dim. of feature vector)
                dataset = numpy.zeros(
                    (
                        features.shape[0] * len(files),
                        CONFIG["feature"]["n_mels"] * CONFIG["feature"]["n_frames"],
                    ),
                    numpy.float32,
                )

            dataset[
                features.shape[0] * file_id : features.shape[0] * (file_id + 1), :
            ] = features

        self.feat_data = dataset

        train_size = int(len(dataset) * (1.0 - CONFIG["training"]["validation_split"]))
        print(
            "train_size: %d, val_size: %d"
            % (
                train_size,
                int(len(dataset) * CONFIG["training"]["validation_split"]),
            )
        )

    def __len__(self):
        return self.feat_data.shape[0]  # return num of samples

    def __getitem__(self, index):
        sample = self.feat_data[index, :]  # return vector

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader(dataset):
    """
    Make dataloader from dataset for training.
    """
    train_size = int(len(dataset) * (1.0 - CONFIG["training"]["validation_split"]))
    data_loader_train = torch.utils.data.DataLoader(
        Subset(dataset, list(range(0, train_size))),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=CONFIG["training"]["shuffle"],
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        Subset(dataset, list(range(train_size, len(dataset)))),
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    return data_loader_train, data_loader_val


def get_model():
    """
    Instantiate AutoEncoder.
    """
    model = AutoEncoder(
        x_dim=CONFIG["feature"]["n_mels"] * CONFIG["feature"]["n_frames"],
        h_dim=CONFIG["model"]["hidden_dim"],
        z_dim=CONFIG["model"]["latent_dim"],
        n_hidden=CONFIG["model"]["n_hidden"],
    )
    return model


def get_optimizer(model):
    """
    Instantiate optimizer.
    """

    optimizer = optim.Adam(
        params=model.parameters(),
        weight_decay=CONFIG["training"]["weight_decay"],
        lr=CONFIG["training"]["learning_rate"],
    )

    # optional
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG["training"]["lr_step_size"],
        gamma=CONFIG["training"]["lr_gamma"],
    )

    return optimizer, scheduler


def training(model, data_loader, optimizer, scheduler=None):
    """
    Perform training
    """
    model.train()  # training mode
    train_loss = 0.0
    for data in data_loader:
        data = data.to(DEVICE)  # send data to GPU
        data = data.float()  # workaround
        optimizer.zero_grad()  # reset gradient
        loss = model.get_loss(data)
        loss.backward()  # backpropagation
        train_loss += loss.item()
        optimizer.step()  # update paramerters

    if scheduler is not None:
        scheduler.step()  # update learning rate

    print("train_loss: {:.6f} ".format(train_loss / len(data_loader)), end="")
    return train_loss / len(data_loader)


def validation(model, data_loader):
    """
    Perform validation
    """
    model.eval()  # validation mode
    val_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(DEVICE)  # send data to GPU
            data = data.float()  # workaround
            loss = model.get_loss(data)
            val_loss += loss.item()

    print("val_loss: {:.6f} ".format(val_loss / len(data_loader)))
    return val_loss / len(data_loader)


def calc_anomaly_score(model, data_loader):
    """
    - Calculate anomaly scores over all feature vectors
    - Fit gamma distribution for anomaly scores
    - Returns parameters of gamma distribution
    """
    anomaly_score = []
    model.eval()  # validation mode
    with torch.no_grad():
        for data in data_loader:
            data = data.to(DEVICE)  # send data to GPU
            data = data.float()  # workaround
            loss = model.get_loss(data)
            anomaly_score.append(loss)

    anomaly_score = numpy.array(anomaly_score, dtype=float)
    gamma_params = scipy.stats.gamma.fit(anomaly_score)
    gamma_params = list(gamma_params)

    return gamma_params


def fit_gamma_dist(dataset, model, target_dir):
    """
    - Fit gamma distribution for anomaly scores.
    - Save the parameters of the distribution.
    """

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    # fit gamma distribution for anomaly scores
    gamma_params = calc_anomaly_score(model=model, data_loader=data_loader)
    score_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=CONFIG["model_directory"], machine_type=os.path.split(target_dir)[1]
    )
    # save the parameters of the distribution
    joblib.dump(gamma_params, score_file_path)


def save_model(model, model_dir, machine_type):
    """
    Save PyTorch model.
    """

    model_file_path = "{model}/model_{machine_type}.hdf5".format(
        model=model_dir, machine_type=machine_type
    )
    # if os.path.exists(model_file_path):
    #     print("Model already exists!")
    #     continue
    torch.save(model.state_dict(), model_file_path)
    print("save_model -> %s" % (model_file_path))


def main():
    """
    Perform model training and validation.
    """

    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = util.command_line_chk()  # constant: True or False
    if mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(CONFIG["model_directory"], exist_ok=True)

    # load base_directory list
    dir_list = util.select_dirs(config=CONFIG, mode=mode)
    for idx, target_dir in enumerate(dir_list):
        print("===============================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

        print("\n============== DATASET_GENERATOR ==============")
        # generate file list under "target_dir" directory.
        files, _ = util.file_list_generator(
            target_dir=target_dir,
            section_name="*",
            dir_name="train",
            mode=mode,
        )
        dcase_dataset = DcaseDataset(files)  # generate dataset from file list.
        print("===============================================")

        print("\n=========== DATALOADER_GENERATOR ==============")
        data_loader = {"train": None, "val": None}
        data_loader["train"], data_loader["val"] = get_dataloader(dcase_dataset)
        print("===============================================")

        print("\n================ MODEL TRAINING ===============")
        model = get_model().to(DEVICE)
        optimizer, _ = get_optimizer(model)
        # optimizer, scheduler = get_optimizer(model)  # optional

        # display summary of model through torchinfo
        summary(
            model,
            input_size=(
                CONFIG["training"]["batch_size"],
                CONFIG["feature"]["n_mels"] * CONFIG["feature"]["n_frames"],
            ),
        )

        # training loop
        for epoch in range(1, CONFIG["training"]["epochs"] + 1):
            print("Epoch {:2d}: ".format(epoch), end="")
            training(
                model=model,
                data_loader=data_loader["train"],
                optimizer=optimizer,
                # scheduler=scheduler  # optional
            )
            validation(model=model, data_loader=data_loader["val"])

        del data_loader  # delete the dataset for training.

        # fit gamma distribution for anomaly scores
        # and save the parameters of the distribution
        fit_gamma_dist(dcase_dataset, model, target_dir)

        print("============== SAVE MODEL ==============")
        save_model(
            model,
            model_dir=CONFIG["model_directory"],
            machine_type=os.path.split(target_dir)[1],
        )

        print("============== END TRAINING ==============")


if __name__ == "__main__":
    main()
