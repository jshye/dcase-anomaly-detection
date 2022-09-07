import os
from argparse import ArgumentParser
from importlib.util import spec_from_file_location, module_from_spec
import random
import torch
import numpy as np
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

    domain_data = {'source': {}, 'target': {}}
    for domain in ['source', 'target']:
        normal_files, anomaly_files = data_utils.file_list_generator(
            config['dataset']['dev_dir'], args.machine, 'test', domain=domain,
        )
        normal_dataset = data_utils.DcaseDataset(normal_files, args.machine, config)
        normal_dataloader, _ = data_utils.get_dataloader(normal_dataset, batch_size, val=False)
        anomaly_dataset = data_utils.DcaseDataset(anomaly_files, args.machine, config)
        anomaly_dataloader, _ = data_utils.get_dataloader(anomaly_dataset, batch_size, val=False)

        domain_data[domain]['normal_dataloader'] = normal_dataloader
        domain_data[domain]['anomaly_dataloader'] = anomaly_dataloader


    # ckpt_dir = Path(Path(__file__).resolve().parents[0], config.logging.checkpoints)
    ckpt_dir = f'./results/{args.run_name}/checkpoints'
    ckpt_paths = sorted(os.listdir(ckpt_dir))

    os.makedirs(f'./results/{args.run_name}/roc_plots/', exist_ok=True)

    print('-' * 80)
    print(f"{'Epoch':<7}{'S-AUC':<10}{'S-Normal':<12}{'S-Anomaly':<12}{'T-AUC':<10}{'T-Normal':<12}{'T-Anomaly':<12}")
    print('-' * 80)

    best_epoch = {'source': 0, 'target': 0}
    best_auc = {'source': 0, 'target': 0}

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

        epoch_msg = f'{epoch+1:<7}'
        for domain in ['source', 'target']:
            pred_dict = {'normal': [], 'anomaly': []}
            true = []
            
            for key in ['normal', 'anomaly']:
                dataloader = domain_data[domain][f'{key}_dataloader']
                for x, y in dataloader:
                    x = x.to(DEVICE).float()  # log mel spectrogram
                    y = y.to(DEVICE).long()   # section id

                    out = model(x)
                    if config['model']['task'] == 'classification':
                        out /= config['settings']['temp_scaling']
                    score = utils.get_anomaly_score(config, x, out)
                    pred_dict[key].extend(score.cpu())

                    if key == 'normal':            
                        true.extend(np.zeros(len(x)))
                    elif key == 'anomaly':            
                        true.extend(np.ones(len(x)))

            pred = [*pred_dict['normal'], *pred_dict['anomaly']]
            auc = roc_auc_score(true, pred)
            fpr, tpr, _ = roc_curve(true, pred)

            roc_logger = logger.ROCLogger(f'Epoch {epoch+1}, {domain.capitalize()} Domain, AUC: {auc:.4f}')
            roc_logger.plot_roc(fpr, tpr, pred_dict['normal'], pred_dict['anomaly'])
            if args.neptune:
                run[f'test/{args.machine}/{domain}_domain/roc'].log(roc_logger.fig)
            roc_logger.save_fig(f'./results/{args.run_name}/roc_plots/{args.machine}-epoch{epoch+1}-{domain}.png')
            
            if best_auc[domain] <= auc:
                best_epoch[domain] = epoch + 1
                best_auc[domain] = auc

            if args.neptune:
                run[f'test/{args.machine}/{domain}_domain/AUC'].log(auc)
                run[f'test/{args.machine}/{domain}_domain/Normal_score'].log(np.mean(pred_dict['normal']))
                run[f'test/{args.machine}/{domain}_domain/Anomaly_score'].log(np.mean(pred_dict['anomaly']))
        
            epoch_msg += f"{auc:<10.4f}{np.mean(pred_dict['normal']):<12.4f}{np.mean(pred_dict['anomaly']):<12.4f}"

        print(epoch_msg)

    print('-' * 80)
    print(f"Best: [Source] Epoch {best_epoch['source']} - {best_auc['source']:.4f}"+
          f"  [Target] Epoch {best_epoch['target']} - {best_auc['target']:.4f}")
    print('-' * 80)


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test(args, DEVICE)