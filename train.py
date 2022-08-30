import os
from argparse import ArgumentParser
import yaml
from importlib.util import spec_from_file_location, module_from_spec
import random
import torch
import torch.nn.functional as F
import numpy as np
import utils
import data_utils
from itertools import zip_longest
from tqdm import tqdm
from logger import TrainLogger


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
    dataset = data_utils.DcaseDataset(normal_files, args.machine, config)
    dataloader, val_dataloader = data_utils.get_dataloader(dataset, batch_size, val=args.run_val, shuffle=True)

    if args.ood:  # outlier exposure
        oe_batch_size = max(1, batch_size // (2 ** config['settings']['oe_div_power']))
        outlier_files, _ = data_utils.file_list_generator(
            config['dataset']['dev_dir'], args.ood, 'train'
        )
        if args.run_val:
            outlier_files = outlier_files[:oe_batch_size*(len(dataloader)+len(val_dataloader))]
        else:
            outlier_files = outlier_files[:oe_batch_size*len(dataloader)]
        outlier_dataset = data_utils.DcaseDataset(outlier_files, args.machine, config)
        outlier_dataloader, outlier_val_dataloader = data_utils.get_dataloader(outlier_dataset, oe_batch_size, val=args.run_val, shuffle=True)
        n_class = config['model']['n_class']
        oe_loss_coef = config['settings']['oe_loss_coef']

    else: 
        outlier_dataloader = []
        outlier_val_dataloader = []

    # model = utils.get_model(args.model, config)
    model_dict = {
        'resnet': 'ResNet',
        'liu2022': 'Liu2022',
        # 'ganomaly': 'Ganomaly',
    }

    if args.model in model_dict.keys():
        spec = spec_from_file_location(args.model, f'./models/{args.model}/{args.model}.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, model_dict[args.model])(config)
    else:
        raise ValueError(f"'{args.model}' Not Implemented!")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 weight_decay=config['settings']['weight_decay'],
                                 lr=float(config['settings']['lr']))
    scheduler = None

    train_losses = []  # train loss values for all epochs
    in_train_losses = []
    oe_train_losses = []
    valid_losses = []  # validation loss values for all epochs
    in_valid_losses = []
    oe_valid_losses = []

    os.makedirs(f'./results/{args.run_name}', exist_ok=True)
    with open(f'./results/{args.run_name}/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    train_logger = TrainLogger(title=f'[{args.run_name}] {args.machine} - ood: {args.ood}',
                               xlabel='Epoch', ylabel='Loss')

    for epoch in range(1, config['settings']['max_epoch'] + 1):
        train_loss = []
        in_train_loss = []
        oe_train_loss = []
        model.train()
        for indata, oedata in tqdm(zip_longest(dataloader, outlier_dataloader), leave=False, total=100):
            x, y = indata
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).long()

            logits = model(x)
            loss_step = utils.get_loss(config, x, y, logits, DEVICE)

            if oedata is not None:  # outlier exposure
                in_train_loss.append(loss_step.cpu().detach())
                
                oe_x, oe_y = oedata
                oe_x = oe_x.to(DEVICE).float()
                oe_y = oe_y.to(DEVICE).long()
                oe_p = torch.ones(oe_x.shape[0], n_class) / torch.tensor(n_class)  # uniform distribution
                oe_p = oe_p.float().to(DEVICE)

                oe_logits = model(oe_x)
                oe_logits = torch.softmax(oe_logits, dim=1)
                oe_loss_step = torch.nn.CrossEntropyLoss()(oe_logits, oe_p)
                loss_step += (oe_loss_coef * oe_loss_step)

                oe_train_loss.append(oe_loss_step.cpu().detach())
            
            train_loss.append(loss_step.cpu().detach())

            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        train_loss_epoch = np.mean(train_loss)
        train_losses.append(train_loss_epoch)
        epoch_msg = f'[Epoch {epoch:3d}] Train Loss: {train_loss_epoch:<12.4f}'
        train_logger.plot_learning_curve(train_losses, label='Train')

        if args.neptune:
            run[f'training/{args.machine}/train/epoch/loss'].log(train_loss_epoch)
        # if args.ood:
        #     in_train_loss_epoch = np.mean(in_train_loss)
        #     in_train_losses.append(in_train_loss_epoch)
        #     oe_train_loss_epoch = np.mean(oe_train_loss)
        #     oe_train_losses.append(oe_train_loss_epoch)

        #     train_logger.plot_learning_curve(in_train_losses, label='Train (indist)')
        #     train_logger.plot_learning_curve(oe_train_losses, label='Train (ood)')

        if args.run_val:  # validation loop
            valid_loss = []
            in_valid_loss = []
            oe_valid_loss = []
            model.eval()
            for indata, oedata in tqdm(zip_longest(val_dataloader, outlier_val_dataloader), leave=False, total=100):
                x, y = indata
                x = x.to(DEVICE).float()
                y = y.to(DEVICE).long()

                logits = model(x)
                loss_step = utils.get_loss(config, x, y, logits, DEVICE)

                if oedata is not None:  # outlier exposure
                    in_valid_loss.append(loss_step.cpu().detach())
                    
                    oe_x, oe_y = oedata
                    oe_x = oe_x.to(DEVICE).float()
                    oe_y = oe_y.to(DEVICE).long()
                    oe_p = torch.ones(oe_x.shape[0], n_class) / torch.tensor(n_class)  # uniform distribution
                    oe_p = oe_p.float().to(DEVICE)

                    oe_logits = model(oe_x)
                    oe_logits = torch.softmax(oe_logits, dim=1)
                    oe_loss_step = torch.nn.CrossEntropyLoss()(oe_logits, oe_p)
                    loss_step += (oe_loss_coef * oe_loss_step)

                    oe_valid_loss.append(oe_loss_step.cpu().detach())

                valid_loss.append(loss_step.cpu().detach())

            valid_loss_epoch = np.mean(valid_loss)
            valid_losses.append(valid_loss_epoch)

            epoch_msg += f'Valid Loss: {valid_loss_epoch:<12.4f}'
            train_logger.plot_learning_curve(valid_losses, label='Validation')

            if args.neptune:
                run[f'training/{args.machine}/valid/epoch/loss'].log(valid_loss_epoch)
                
            # if args.ood:
            #     in_valid_loss_epoch = np.mean(in_valid_loss)
            #     in_valid_losses.append(in_valid_loss_epoch)
            #     oe_valid_loss_epoch = np.mean(oe_valid_loss)
            #     oe_valid_losses.append(oe_valid_loss_epoch)
                
            #     train_logger.plot_learning_curve(in_valid_losses, label='Valid (indist)')
            #     train_logger.plot_learning_curve(oe_valid_losses, label='Valid (ood)')
        
        print(epoch_msg)
        train_logger.save_fig(f'./results/{args.run_name}/{args.machine}-learning_curve.png')

        if args.save_ckpt:
            ckpt_dir = f'./results/{args.run_name}/checkpoints'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'{args.machine}-epoch{epoch}.pt')
            torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(args, DEVICE)