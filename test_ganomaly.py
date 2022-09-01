import os
from argparse import ArgumentParser
from importlib.util import spec_from_file_location, module_from_spec
import random
import torch
import numpy as np
from tqdm import tqdm
import utils
import data_utils
from logger import ROCLogger, GANSampleLogger
from sklearn.metrics import roc_auc_score, roc_curve
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
    # parser.add_argument('--logger', type=str, default='neptune', help='Logger Type <neptune, csv>.')
    parser.add_argument('--neptune', action='store_true', help='Log results in Neptune')
    parser.add_argument('--run_name', type=str, required=True, help="Run name for saving checkpoints and logging")
    
    args = parser.parse_args()
    return args

def test(args, DEVICE):
    config = utils.get_test_config(args.run_name)

    if args.neptune:
        import neptune.new as neptune
        from getpass import getpass
        NEPTUNE_API_TOKEN = getpass('Enter your private Neptune API token: ')

        run = neptune.init(
            project='seonghye/dcase-anomaly',
            api_token=NEPTUNE_API_TOKEN,
            run=args.run_name,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
        )

        if config['settings']['temp_scaling'] != 1.0:
            run[f'test/{args.machine}/temperature_scaling'].log(config['settings']['temp_scaling'])
            run['sys/tags'].add(['temperature_scaling'])

    if config['settings']['seed']:
        seed_everything(config['settings']['seed'])
    
    batch_size = config['settings']['test_batch_size']
    input_dim = config['model']['input_dim']
    input_dim_split = config['model']['input_size'][1]

    ## source domain data
    source_normal_files, source_anomaly_files = data_utils.file_list_generator(
        config['dataset']['dev_dir'], args.machine, 'test', domain='source',
    )
    source_normal_dataset = data_utils.DcaseDataset(source_normal_files, args.machine, config, dim=input_dim, dim_split=input_dim_split)
    source_normal_dataloader, _ = data_utils.get_dataloader(source_normal_dataset, batch_size, val=False)
    source_anomaly_dataset = data_utils.DcaseDataset(source_anomaly_files, args.machine, config, dim=input_dim, dim_split=input_dim_split)
    source_anomaly_dataloader, _ = data_utils.get_dataloader(source_anomaly_dataset, batch_size, val=False)

    ## target domain data
    target_normal_files, target_anomaly_files = data_utils.file_list_generator(
        config['dataset']['dev_dir'], args.machine, 'test', domain='target',
    )
    target_normal_dataset = data_utils.DcaseDataset(target_normal_files, args.machine, config, dim=input_dim, dim_split=input_dim_split)
    target_normal_dataloader, _ = data_utils.get_dataloader(target_normal_dataset, batch_size, val=False)
    target_anomaly_dataset = data_utils.DcaseDataset(target_anomaly_files, args.machine, config, dim=input_dim, dim_split=input_dim_split)
    target_anomaly_dataloader, _ = data_utils.get_dataloader(target_anomaly_dataset, batch_size, val=False)

    # ckpt_dir = Path(Path(__file__).resolve().parents[0], config.logging.checkpoints)
    ckpt_dir = f'./results/{args.run_name}/checkpoints'
    ckpt_paths = sorted(os.listdir(ckpt_dir))

    os.makedirs(f'./results/{args.run_name}/roc_plots/', exist_ok=True)
    os.makedirs(f'./results/{args.run_name}/samples/', exist_ok=True)

    print('-' * 80)
    print(f"{'Epoch':<7}{'S-AUC':<10}{'T-AUC':<10}{'S-Normal':<12}{'S-Anomaly':<12}{'T-Normal':<12}{'T-Anomaly':<12}")
    print('-' * 80)

    top_source_epoch, top_target_epoch = 0, 0
    top_source_auc, top_target_auc = 0, 0

    for epoch, ckpt in enumerate(ckpt_paths):
        model = Ganomaly(config)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt)))
        model. eval()

        source_normal_pred = []
        source_anomaly_pred = []
        source_true = []

        for x, y in source_normal_dataloader:
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            padded, fake, latent_i, latent_o = model(x)
            if config['model']['task'] == 'classification':
                out /= config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, latent_i, latent_o)
            source_normal_pred.extend(score.cpu())
            source_true.extend(np.zeros(len(x)))  # normal samples = 0

        sample_logger = GANSampleLogger(f'Source Normal - epoch {epoch+1}', num_samples=3)
        for i in range(3):
            sample_src, sample_real, sample_fake = x[i].cpu(), padded[i].cpu(), fake[i].cpu().detach()
            sample_src, sample_real, sample_fake = utils.reshape_gan_output(sample_src, sample_real, sample_fake, input_dim)
            sample_logger.plot_sample(sample_src, sample_real, sample_fake, i+1)
        if args.neptune:
            run[f'test/{args.machine}/source_domain/normal_sample'].log(sample_logger.fig)
        sample_logger.save_fig(f'./results/{args.run_name}/samples/{args.machine}-epoch{epoch+1}-source_normal.png')

        for x, y in source_anomaly_dataloader:
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            padded, fake, latent_i, latent_o = model(x)
            if config['model']['task'] == 'classification':
                out /= config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, latent_i, latent_o)
            source_anomaly_pred.extend(score.cpu())
            source_true.extend(np.ones(len(x)))  # anomaly samples = 1

        sample_logger = GANSampleLogger(f'Source Anomaly - epoch {epoch+1}', num_samples=3)
        for i in range(3):
            sample_src, sample_real, sample_fake = x[i].cpu(), padded[i].cpu(), fake[i].cpu().detach()
            sample_src, sample_real, sample_fake = utils.reshape_gan_output(sample_src, sample_real, sample_fake, input_dim)
            sample_logger.plot_sample(sample_src, sample_real, sample_fake, i+1)
        if args.neptune:
            run[f'test/{args.machine}/source_domain/anomaly_sample'].log(sample_logger.fig)
        sample_logger.save_fig(f'./results/{args.run_name}/samples/{args.machine}-epoch{epoch+1}-source_anomaly.png')

        source_pred = [*source_normal_pred, *source_anomaly_pred]
        source_auc = roc_auc_score(source_true, source_pred)
        source_fpr, source_tpr, _ = roc_curve(source_true, source_pred)

        source_roc_logger = ROCLogger(f'Epoch {epoch+1}, Source Domain, AUC: {source_auc:.4f}')
        source_roc_logger.plot_roc(source_fpr, source_tpr, source_normal_pred, source_anomaly_pred)
        if args.neptune:
            run[f'test/{args.machine}/source_domain/roc'].log(source_roc_logger.fig)
        source_roc_logger.save_fig(f'./results/{args.run_name}/roc_plots/{args.machine}-epoch{epoch+1}-source.png')
        target_normal_pred = []
        target_anomaly_pred = []
        target_true = []

        for x, y in target_normal_dataloader:
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            padded, fake, latent_i, latent_o = model(x)
            if config['model']['task'] == 'classification':
                out /= config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, latent_i, latent_o)
            target_normal_pred.extend(score.cpu())
            target_true.extend(np.zeros(len(x)))  # normal samples = 0

        sample_logger = GANSampleLogger(f'Target Normal - epoch {epoch+1}', num_samples=3)
        for i in range(3):
            sample_src, sample_real, sample_fake = x[i].cpu(), padded[i].cpu(), fake[i].cpu().detach()
            sample_src, sample_real, sample_fake = utils.reshape_gan_output(sample_src, sample_real, sample_fake, input_dim)
            sample_logger.plot_sample(sample_src, sample_real, sample_fake, i+1)
        if args.neptune:
            run[f'test/{args.machine}/target_domain/normal_sample'].log(sample_logger.fig)
        sample_logger.save_fig(f'./results/{args.run_name}/samples/{args.machine}-epoch{epoch+1}-target_normal.png')

        for x, y in target_anomaly_dataloader:
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            padded, fake, latent_i, latent_o = model(x)
            if config['model']['task'] == 'classification':
                out /= config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, latent_i, latent_o)
            target_anomaly_pred.extend(score.cpu())
            target_true.extend(np.ones(len(x)))  # anomaly samples = 1

        sample_logger = GANSampleLogger(f'Target Anomaly - epoch {epoch+1}', num_samples=3)
        for i in range(3):
            sample_src, sample_real, sample_fake = x[i].cpu(), padded[i].cpu(), fake[i].cpu().detach()
            sample_src, sample_real, sample_fake = utils.reshape_gan_output(sample_src, sample_real, sample_fake, input_dim)
            sample_logger.plot_sample(sample_src, sample_real, sample_fake, i+1)
        if args.neptune:
            run[f'test/{args.machine}/target_domain/anomaly_sample'].log(sample_logger.fig)
        sample_logger.save_fig(f'./results/{args.run_name}/samples/{args.machine}-epoch{epoch+1}-target_anomaly.png')

        target_pred = [*target_normal_pred, *target_anomaly_pred]
        target_auc = roc_auc_score(target_true, target_pred)
        target_fpr, target_tpr, _ = roc_curve(target_true, target_pred)
        target_roc_logger = ROCLogger(f'Epoch {epoch+1}, Target Domain, AUC: {target_auc:.4f}')
        target_roc_logger.plot_roc(target_fpr, target_tpr, target_normal_pred, target_anomaly_pred)
        if args.neptune:
            run[f'test/{args.machine}/target_domain/roc'].log(target_roc_logger.fig)
        target_roc_logger.save_fig(f'./results/{args.run_name}/roc_plots/{args.machine}-epoch{epoch+1}-target.png')
        
        print(f"{epoch+1:<7}{source_auc:<10.4f}{target_auc:<10.4f}{np.mean(source_normal_pred):<12.4f}"+
        f"{np.mean(source_anomaly_pred):<12.4f}{np.mean(target_normal_pred):<12.4f}{np.mean(target_anomaly_pred):<12.4f}")

        if args.neptune:
            run[f'test/{args.machine}/source_domain/AUC'].log(source_auc)
            run[f'test/{args.machine}/source_domain/Normal_score'].log(np.mean(source_normal_pred))
            run[f'test/{args.machine}/source_domain/Anomaly_score'].log(np.mean(source_anomaly_pred))
            run[f'test/{args.machine}/target_domain/AUC'].log(target_auc)
            run[f'test/{args.machine}/target_domain/Normal_score'].log(np.mean(target_normal_pred))
            run[f'test/{args.machine}/target_domain/Anomaly_score'].log(np.mean(target_anomaly_pred))

        if top_source_auc <= source_auc:
            top_source_epoch = epoch + 1
            top_source_auc = source_auc
        if top_target_auc <= target_auc:
            top_target_epoch = epoch + 1
            top_target_auc = target_auc

    print('-' * 80)
    print(f'Best: [Source] Epoch {top_source_epoch} - {top_source_auc:.4f}'+
          f'  [Target] Epoch {top_target_epoch} - {top_target_auc:.4f}')
    print('-' * 80)


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test(args, DEVICE)