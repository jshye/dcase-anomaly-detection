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
from torchinfo import summary

import joblib
from tqdm import tqdm

# Local application/library specific imports.
import util
# from models import AutoEncoder
from models import Lopez2020
from data_utils import *

# Load configuration from YAML file.
config = util.load_yaml("./config.yaml")

# String constant: "cuda:0" or "cpu"
DEVICE = util.get_device()

def get_model():
    """
    Instantiate AutoEncoder.
    """
    # model = AutoEncoder(
    #     x_dim=config["feature"]["n_mels"] * config["feature"]["n_frames"],
    #     h_dim=config["model"]["hidden_dim"],
    #     z_dim=config["model"]["latent_dim"],
    #     n_hidden=config["model"]["n_hidden"],
    # )

    model = Lopez2020(num_sections=3,
                      n_frames=config["feature"]["n_frames"],
                      n_mels=config["feature"]["n_mels"])
    return model


def get_optimizer(model):
    """
    Instantiate optimizer.
    """

    optimizer = optim.Adam(
        params=model.parameters(),
        weight_decay=config["training"]["weight_decay"],
        lr=config["training"]["learning_rate"],
    )

    # optional
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"],
    )

    return optimizer, scheduler


def iterloop(config, epoch, model, data_loader, mode, optimizer=None, scheduler=None):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    losses = []
    with tqdm(data_loader) as pbar:
        # for data in pbar:
        #     data = data.to(DEVICE).float()
        #     if optimizer is not None:
        #         optimizer.zero_grad()
        #     loss = model.get_loss(data)
        for data, label in pbar:
            data = data.to(DEVICE).float()
            label = label.to(DEVICE).float()
            if optimizer is not None:
                optimizer.zero_grad()
            loss = model.get_loss(data, label)

            if mode == 'train':
                loss.backward()
                optimizer.step()
            losses.append(loss.item())

            pbar.set_postfix({'mode': mode, 'epoch': epoch, 'loss': np.mean(losses)})

        if scheduler is not None:
            scheduler.step()
    return np.mean(losses)


def calc_anomaly_score(model, data_loader):
    """
    - Calculate anomaly scores over all feature vectors
    - Fit gamma distribution for anomaly scores
    - Returns parameters of gamma distribution
    """
    anomaly_score = []
    model.eval()  # validation mode
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(DEVICE).float()  # send data to GPU
            label = label.to(DEVICE).float()  # workaround
            loss = model.get_loss(data, label).cpu().numpy()
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
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    # fit gamma distribution for anomaly scores
    gamma_params = calc_anomaly_score(model=model, data_loader=data_loader)
    score_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=config["model_directory"], machine_type=os.path.split(target_dir)[1]
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


def main(config):
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
    os.makedirs(config["model_directory"], exist_ok=True)

    # load base_directory list
    dir_list = util.select_dirs(config=config, mode=mode)
    for idx, target_dir in enumerate(dir_list):
        csv_logdir = os.path.join(config["model_directory"], f'{os.path.basename(target_dir)}_csv_log.csv')
        if os.path.exists(csv_logdir):
            os.remove(csv_logdir)

        with open(csv_logdir, 'a') as f:
            f.write('config\n')
            for k, v in config.items():
                f.write(k + ',' + str(v) + '\n')
            f.write('epoch,train_loss,val_loss\n')
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

        dcase_dataset = DcaseDataset(files, config=config, transform=None)  # generate dataset from file list.
        print("===============================================")

        print("\n=========== DATALOADER_GENERATOR ==============")
        data_loader = {"train": None, "val": None}
        data_loader["train"], data_loader["val"] = get_dataloader(dcase_dataset, config=config)
        print("===============================================")

        print("\n================ MODEL TRAINING ===============")
        model = get_model().to(DEVICE)
        optimizer, _ = get_optimizer(model)
        # optimizer, scheduler = get_optimizer(model)  # optional

        # display summary of model through torchinfo
        summary(
            model,
            input_size=(
                config["training"]["batch_size"],
                # config["feature"]["n_mels"] * config["feature"]["n_frames"],
                1,
                config["feature"]["n_frames"],
                config["feature"]["n_mels"],
            ),
        )

        # training loop
        for epoch in range(1, config["training"]["epochs"] + 1):
            print("Epoch {:2d}: ".format(epoch), end="")
            train_loss = iterloop(
                config=config,
                epoch=epoch,
                model=model,
                data_loader=data_loader["train"],
                mode='train',
                optimizer=optimizer,
                # scheduler=scheduler  # optional
            )
            with torch.no_grad():
                val_loss = iterloop(config=config, epoch=epoch, model=model, data_loader=data_loader["val"], mode='val')

            with open(csv_logdir, 'a') as f:
                f.write(f'{epoch},{train_loss},{val_loss}\n')
                

        del data_loader  # delete the dataset for training.

        # fit gamma distribution for anomaly scores
        # and save the parameters of the distribution
        fit_gamma_dist(dcase_dataset, model, target_dir)

        print("============== SAVE MODEL ==============")
        save_model(
            model,
            model_dir=config["model_directory"],
            machine_type=os.path.split(target_dir)[1],
        )

        print("============== END TRAINING ==============")


if __name__ == "__main__":
    main(config)
