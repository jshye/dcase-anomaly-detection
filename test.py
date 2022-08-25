import os
# from pathlib import Path
from argparse import ArgumentParser
from importlib.util import spec_from_file_location, module_from_spec
import random
import torch
import numpy as np
from tqdm import tqdm
import utils
import data_utils
import logger
from sklearn.metrics import roc_auc_score, roc_curve


def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

    ## source domain data
    source_normal_files, source_anomaly_files = data_utils.file_list_generator(
        config['dataset']['dev_dir'], args.machine, 'test', domain='source',
    )
    source_normal_dataset = data_utils.DcaseDataset(source_normal_files, args.machine, config)
    source_normal_dataloader, _ = data_utils.get_dataloader(source_normal_dataset, batch_size, val=False)
    source_anomaly_dataset = data_utils.DcaseDataset(source_anomaly_files, args.machine, config)
    source_anomaly_dataloader, _ = data_utils.get_dataloader(source_anomaly_dataset, batch_size, val=False)

    ## target domain data
    target_normal_files, target_anomaly_files = data_utils.file_list_generator(
        config['dataset']['dev_dir'], args.machine, 'test', domain='target',
    )
    target_normal_dataset = data_utils.DcaseDataset(target_normal_files, args.machine, config)
    target_normal_dataloader, _ = data_utils.get_dataloader(target_normal_dataset, batch_size, val=False)
    target_anomaly_dataset = data_utils.DcaseDataset(target_anomaly_files, args.machine, config)
    target_anomaly_dataloader, _ = data_utils.get_dataloader(target_anomaly_dataset, batch_size, val=False)

    # ckpt_dir = Path(Path(__file__).resolve().parents[0], config.logging.checkpoints)
    ckpt_dir = f'./results/{args.run_name}/checkpoints'
    ckpt_paths = sorted(os.listdir(ckpt_dir))

    os.makedirs(f'./results/{args.run_name}/roc_plots/', exist_ok=True)

    print('-' * 80)
    print(f"{'Epoch':<7}{'S-AUC':<10}{'T-AUC':<10}{'S-Normal':<12}{'S-Anomaly':<12}{'T-Normal':<12}{'T-Anomaly':<12}")
    print('-' * 80)

    top_source_epoch, top_target_epoch = 0, 0
    top_source_auc, top_target_auc = 0, 0

    for epoch, ckpt in enumerate(ckpt_paths):
        model_dict = {
            'resnet': 'ResNet',
            # 'ganomaly': 'Ganomaly',
            'liu2022': 'Liu2022',
        }

        if args.model in model_dict.keys():
            spec = spec_from_file_location(args.model, f'./models/{args.model}/{args.model}.py')
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            model = getattr(module, model_dict[args.model])(config)
        else:
            raise ValueError(f"'{args.model}' Not Implemented!")

        model = model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt)))
        model. eval()

        source_normal_pred = []
        source_anomaly_pred = []
        source_true = []

        for x, y in tqdm(source_normal_dataloader, leave=False, total=100):
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            out = model(x)
            out = out / config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, x, y, out, DEVICE)
            source_normal_pred.extend(score.cpu())
            source_true.extend(np.zeros(len(x)))  # normal samples = 0
        
        for x, y in tqdm(source_anomaly_dataloader, leave=False, total=100):
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            out = model(x)
            out = out / config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, x, y, out, DEVICE)
            source_anomaly_pred.extend(score.cpu())
            source_true.extend(np.ones(len(x)))  # anomaly samples = 1

        source_pred = [*source_normal_pred, *source_anomaly_pred]
        source_auc = roc_auc_score(source_true, source_pred)
        source_fpr, source_tpr, _ = roc_curve(source_true, source_pred)

        source_roc_logger = logger.ROCLogger(f'Epoch {epoch+1}, Source Domain, AUC: {source_auc:.4f}')
        source_roc_logger.plot_roc(source_fpr, source_tpr, source_normal_pred, source_anomaly_pred)
        if args.neptune:
            run[f'test/{args.machine}/source_domain/roc'].log(source_roc_logger.fig)
        source_roc_logger.save_fig(f'./results/{args.run_name}/roc_plots/{args.machine}-epoch{epoch+1}-source.png')
        target_normal_pred = []
        target_anomaly_pred = []
        target_true = []

        for x, y in tqdm(target_normal_dataloader, leave=False, total=100):
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            out = model(x)
            out = out / config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, x, y, out, DEVICE)
            target_normal_pred.extend(score.cpu())
            target_true.extend(np.zeros(len(x)))  # normal samples = 0
        
        for x, y in tqdm(target_anomaly_dataloader, leave=False, total=100):
            x = x.to(DEVICE).float()  # log mel spectrogram
            y = y.to(DEVICE).long()   # section id

            out = model(x)
            out = out / config['settings']['temp_scaling']
            score = utils.get_anomaly_score(config, x, y, out, DEVICE)
            target_anomaly_pred.extend(score.cpu())
            target_true.extend(np.ones(len(x)))  # anomaly samples = 1
        
        target_pred = [*target_normal_pred, *target_anomaly_pred]
        target_auc = roc_auc_score(target_true, target_pred)
        target_fpr, target_tpr, _ = roc_curve(target_true, target_pred)
        target_roc_logger = logger.ROCLogger(f'Epoch {epoch+1}, Target Domain, AUC: {target_auc:.4f}')
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