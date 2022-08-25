import os
import yaml
import argparse
import torch
import util
from data_utils import *
from ganomaly_model import *
import joblib
import csv

from sklearn import metrics
from scipy.stats import hmean

import neptune.new as neptune
from getpass import getpass


NEPTUNE_API_TOKEN = getpass('Enter your private Neptune API token: ')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine_type', nargs='*', type=str,
                     help='Choose 0 or more values from [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]')
parser.add_argument('--max_epoch', type=int, help='Maximum epoch index of saved checkpoints')
parser.add_argument('--resume_run', type=str, default=None, help='Neptune run name')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(config, epoch):
    assert args.resume_run is not None, 'Choose Neptune run ID'
    ####### ***** neptune ***** #######
    # NEPTUNE_API_TOKEN = getpass('Enter your private Neptune API token: ')
    if args.resume_run:
        run = neptune.init(
            project='seonghye/dcase-anomaly',
            api_token=NEPTUNE_API_TOKEN,
            run=args.resume_run,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
        )
    ####### ***** ******* ***** #######

    os.makedirs(config['result_dir'], exist_ok=True)
    
    performance = {"section": [], "all": []}
    # score_list = {"anomaly": None, "decision": None}

    dir_list = select_dirs(config=config, mode='dev')

    if args.machine_type is not None:
        dir_list = [d for d in dir_list if os.path.basename(d) in args.machine_type]

    for idx, target_dir in enumerate(dir_list):
        print("====================================================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

        machine_type = os.path.basename(target_dir)
        # machine_config = config[machine_type]

        csv_logdir = os.path.join(config['result_dir'], f'{machine_type}_epoch{epoch}_result.csv')
        with open(csv_logdir, 'a') as f:
            f.write(f'{machine_type}\n')
            f.write('section,domain,AUC,pAUC,precision,recall,F1_score\n')
        
        print("================ LOAD MODEL =================")
        # model_file = os.path.join(config['model_save_dir'], f'model_{machine_type}.pt')
        model_file = os.path.join(config['model_save_dir'], f'model_{machine_type}_epoch{epoch}.pt')
        section_names_file_path = os.path.join(config['model_save_dir'], f'section_names_{machine_type}.pkl')
        trained_section_names = joblib.load(section_names_file_path)
        # n_sections = trained_section_names.shape[0]

        model = GanomalyModel(input_size=(128,313),
                              latent_vec_size=100,
                              num_input_channels=1,
                              n_features=64,
                              extra_layers=0,
                              add_final_conv_layer=True).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(model_file))

        for dir_name in ["source_test", "target_test"]:
            domain = dir_name.split('_', 1)[0]
            domain_auc = []
            for section_name in get_section_names(target_dir, dir_name=dir_name):
                section = section_name.split('_', 1)[1]

                print(f'============== BEGIN TEST FOR A SECTION {section_name} OF {dir_name} ==============')

                print("============= DATASET GENERATOR ==============")
                test_files, labels = file_list_generator(target_dir=target_dir,
                                                         section_name=section_name,
                                                         dir_name=dir_name,
                                                         mode='dev')

                dcase_dataset = DcaseDataset(test_files,
                                             labels,
                                             config=config,
                                             machine_config=config[machine_type],
                                             transform=None)

                print("============ DATALOADER GENERATOR ============")
                data_loader = {'test': None}
                data_loader['test'] = get_eval_dataloader(dcase_dataset, config=config, machine_type=machine_type)

                print("================== RUN TEST ==================")
                anomaly_scores = []
                normal_scores = []    # anomaly scores of normal data
                abnormal_scores = []  # anomaly scores of abnormal data
                anomaly_true = []

                anomaly_plot_cnt = 0
                normal_plot_cnt = 0

                with tqdm(data_loader['test'], bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}') as pbar:
                    for x, s, y in pbar:  # data, section id, anomaly
                        x = x.to(DEVICE).float()
                        s = s.to(DEVICE).long()
                        y = y.to(DEVICE).long()

                        padded, fake, latent_i, latent_o = model(x)
                        anomaly_scores_batch = torch.mean(torch.pow((latent_i - latent_o), 2), dim=[1,2,3])
                        # anomaly_scores.extend(anomaly_scores_batch)
                        
                        for i in range(len(y)):
                            anomaly_scores.append(anomaly_scores_batch[i].detach().cpu())

                            # for verification
                            if y[i] == 0:
                                normal_scores.append(anomaly_scores_batch[i].detach().cpu())
                                if normal_plot_cnt < 3:
                                    recons_fig = plot_recons(x[i], padded[i], fake[i], anomaly=False, epoch=epoch, show=False)
                                    run[f'test/{machine_type}/{domain}_domain/section{section}/normal_recons/{i}'].log(recons_fig)
                                    plt.close()
                                    normal_plot_cnt += 1
                            else:
                                abnormal_scores.append(anomaly_scores_batch[i].detach().cpu())
                                if anomaly_plot_cnt < 3:
                                    recons_fig = plot_recons(x[i], padded[i], fake[i], anomaly=True, epoch=epoch, show=False)
                                    run[f'test/{machine_type}/{domain}_domain/section{section}/anomaly_recons/{i}'].log(recons_fig)
                                    plt.close()
                                    anomaly_plot_cnt += 1

                        anomaly_true.extend(y.cpu())

                auc = metrics.roc_auc_score(anomaly_true, anomaly_scores)
                fpr, tpr, thresholds = metrics.roc_curve(anomaly_true, anomaly_scores)
                domain_auc.append(auc)


                with open(csv_logdir, 'a') as f:
                    section = section_name.split('_', 1)[1]
                    domain = dir_name.split('_', 1)[0]
                    f.write(f'{section},{domain},{auc}\n')
    
                ####### ***** neptune ***** #######
                run[f'test/{machine_type}/{domain}_domain/section{section}/AUC'].log(auc)

                roc = util.plot_roc(fpr, tpr, auc, show=False)
                run[f'test/{machine_type}/{domain}_domain/section{section}/roc'].log(roc)
                plt.close()

                thresholds_str = ', '.join(str(t) for t in thresholds)
                run[f'test/{machine_type}/{domain}_domain/section{section}/thresholds'].log(thresholds_str)
                
                fig = util.plot_anomaly_score_distrib(normal_scores, abnormal_scores, epoch, decimals=2, show=False)
                run[f'test/{machine_type}/{domain}_domain/section{section}/anomaly_score'].log(fig)
                plt.close()

            auc_hmean = hmean(domain_auc)
            run[f'test/{machine_type}/{domain}_domain/auc_hmean'].log(auc_hmean)

        run.stop()
        ####### ***** ******* ***** #######


if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)  
    
    max_epoch = args.max_epoch
    for i in range(1, max_epoch+1):
        main(config, i)