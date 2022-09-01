import os
from argparse import ArgumentParser
import yaml
# from importlib.util import spec_from_file_location, module_from_spec
import random
import torch
# import torch.nn.functional as F
import numpy as np
import utils
import data_utils
# from itertools import zip_longest
from tqdm import tqdm
from logger import TrainLogger

from models.ganomaly.ganomaly import *


def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', help="Name of the model to train.")
    parser.add_argument('--machine', '-m', type=str, default='ToyCar',
                        #  nargs='*',
                         help="Machine Type <ToyCar, ToyTrain, fan, gearbox, pump, slider, valve>.")    
    parser.add_argument('--ood', type=str, default=None,
                        #  nargs='*',
                         help="Out-of-Distribution Machine Type")    
    parser.add_argument('--run_val', action='store_true', help="Run validation")
    # parser.add_argument('--logger', type=str, default='neptune', help='Logger Type <neptune, csv>.')
    parser.add_argument('--neptune', action='store_true', help='Log results in Neptune')
    parser.add_argument('--save_ckpt', action='store_true', help="Save checkpoint at the end of every epoch.")
    parser.add_argument('--run_name', type=str, required=True, help="Run name for saving checkpoints and logging")

    args = parser.parse_args()
    return args

def train(args, DEVICE):
    config = utils.get_config(args.model)

    if args.neptune:
        import neptune.new as neptune
        from getpass import getpass
        NEPTUNE_API_TOKEN = getpass('Enter your private Neptune API token: ')

        run = neptune.init(
            project='seonghye/dcase-anomaly',
            api_token=NEPTUNE_API_TOKEN,
            tags=[config['model']['task']],
            name=args.run_name,
            # description=input('Enter Neptune Description:'),
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
        )
        run[f'training/hyperparams'] = config['settings']
        run['model'] = config['model']['name']
        run[f'training/{args.machine}'] = config['machine_config'][args.machine]
        run[f'training/{args.machine}/ood'] = args.ood
        if config['settings']['label_smoothing'] > 0.0:
            run['sys/tags'].add(['label_smoothing'])
        if args.ood:
            run['sys/tags'].add(['outlier_exposure'])

    print('-' * 20)
    for k, v in config['settings'].items():
        print(f'{k}: {v}')
    print('-' * 20)

    if config['settings']['seed']:
        seed_everything(config['settings']['seed'])

    batch_size = config['settings']['train_batch_size']

    normal_files, _ = data_utils.file_list_generator(
        config['dataset']['dev_dir'], args.machine, 'train'
    )
    dataset = data_utils.DcaseDataset(normal_files, args.machine, config, dim=3, dim_split=128)
    dataloader, val_dataloader = data_utils.get_dataloader(dataset, batch_size, val=args.run_val, shuffle=True)

    
    model = Ganomaly(config) 
    model = model.to(DEVICE)
    optimizer_g = torch.optim.Adam(params=model.parameters(),
                                   weight_decay=config['settings']['weight_decay'],
                                   lr=float(config['settings']['lr_g']))
    optimizer_d = torch.optim.Adam(params=model.parameters(),
                                   weight_decay=config['settings']['weight_decay'],
                                   lr=float(config['settings']['lr_d']))
    scheduler = None

    criterion_g = GeneratorLoss(
        wadv=config['model']['wadv'], wcon=config['model']['wcon'], wenc=config['model']['wenc']
        )
    criterion_d = DiscriminatorLoss()

    train_losses_g = []  # train loss values for all epochs
    train_losses_d = []
    valid_losses_g = []  # validation loss values for all epochs
    valid_losses_d = []

    os.makedirs(f'./results/{args.run_name}', exist_ok=True)
    with open(f'./results/{args.run_name}/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    train_logger_g = TrainLogger(title=f'[{args.run_name}] {args.machine} - ood: {args.ood}',
                                 xlabel='Epoch', ylabel='Loss', i=1)
    train_logger_d = TrainLogger(title=f'[{args.run_name}] {args.machine} - ood: {args.ood}',
                                 xlabel='Epoch', ylabel='Loss', i=2)

    for epoch in range(1, config['settings']['max_epoch'] + 1):
        train_loss_g = []
        train_loss_d = []
        model.train()
        for train_data in tqdm(dataloader, leave=False, total=100):
            x, y = train_data
            x = x.to(DEVICE).float()
            # y = y.to(DEVICE).long()

            padded, fake, latent_i, latent_o = model(x)
            pred_real, _ = model.discriminator(padded)
            pred_fake, _ = model.discriminator(fake.detach())
            loss_g_step, error_enc, error_con, error_adv = criterion_g(latent_i, latent_o, padded, fake, pred_real, pred_fake)
            loss_d_step = criterion_d(pred_real, pred_fake)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            loss_g_step.backward(retain_graph=True)
            loss_d_step.backward()
            optimizer_g.step()
            optimizer_d.step()
            if scheduler is not None:
                scheduler.step()

            train_loss_g.append(loss_g_step.cpu().detach())
            train_loss_d.append(loss_d_step.cpu().detach())

        train_loss_g_epoch = np.mean(train_loss_g)
        train_losses_g.append(train_loss_g_epoch)
        train_loss_d_epoch = np.mean(train_loss_d)
        train_losses_d.append(train_loss_d_epoch)
        
        epoch_msg = f'[Epoch {epoch:3d}] Train Loss G: {train_loss_g_epoch:<12.4f} Train D: {train_loss_d_epoch:<12.4f}'
        train_logger_g.plot_learning_curve(train_losses_g, label='Train G')
        train_logger_d.plot_learning_curve(train_losses_d, label='Train D')

        if args.neptune:
            run[f'training/{args.machine}/train/epoch/loss_g'].log(train_loss_g_epoch)
            run[f'training/{args.machine}/train/epoch/loss_d'].log(train_loss_d_epoch)

        if args.run_val:  # validation loop
            valid_loss_g = []
            valid_loss_d = []
            model.eval()
            for valid_data in tqdm(val_dataloader, leave=False, total=100):
                x, y = valid_data
                x = x.to(DEVICE).float()
                # y = y.to(DEVICE).long()

                padded, fake, latent_i, latent_o = model(x)
                pred_real, _ = model.discriminator(padded)
                pred_fake, _ = model.discriminator(fake.detach())
                loss_g_step, error_enc, error_con, error_adv = criterion_g(latent_i, latent_o, padded, fake, pred_real, pred_fake)
                loss_d_step = criterion_d(pred_real, pred_fake)

                valid_loss_g.append(loss_g_step.cpu().detach())
                valid_loss_d.append(loss_d_step.cpu().detach())

            valid_loss_g_epoch = np.mean(valid_loss_g)
            valid_losses_g.append(valid_loss_g_epoch)
            valid_loss_d_epoch = np.mean(valid_loss_d)
            valid_losses_d.append(valid_loss_d_epoch)
            
            epoch_msg += f'Valid Loss G: {valid_loss_g_epoch:<12.4f} Valid D: {valid_loss_d_epoch:<12.4f}'
            train_logger_g.plot_learning_curve(valid_losses_g, label='valid G')
            train_logger_d.plot_learning_curve(valid_losses_d, label='valid D')

            if args.neptune:
                run[f'training/{args.machine}/valid/epoch/loss_g'].log(valid_loss_g_epoch)
                run[f'training/{args.machine}/valid/epoch/loss_d'].log(valid_loss_d_epoch)

        print(epoch_msg)
        train_logger_g.save_fig(f'./results/{args.run_name}/{args.machine}-learning_curve_g.png')
        train_logger_d.save_fig(f'./results/{args.run_name}/{args.machine}-learning_curve_d.png')

        if args.save_ckpt:
            ckpt_dir = f'./results/{args.run_name}/checkpoints'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'{args.machine}-epoch{epoch}.pt')
            torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(args, DEVICE)