import os
import yaml
import argparse
import torch
import util
from data_utils import *
from ganomaly_model import *
import joblib
import csv
import matplotlib.pyplot as plt

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


def main(config, epoch, decision_threshold=0.7):
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

        # score_distr_file_path = os.path.join(config['model_save_dir'], f'score_distr_{machine_type}_epoch{epoch}.pkl')
        # decision_threshold = util.calc_decision_threshold(score_distr_file_path, config)
        
        ####### ***** neptune ***** #######
        run[f'test/{machine_type}/decision_threshold'].log(decision_threshold)
        ####### ***** ******* ***** #######
        
        for dir_name in ["source_test", "target_test"]:
            for section_name in get_section_names(target_dir, dir_name=dir_name):

                temp_array = np.nonzero(trained_section_names == section_name)[0]  # search for section_name
                if temp_array.shape[0] == 0:
                    section_idx = -1
                else:
                    section_idx = temp_array[0]
                
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
                anomaly_scores_all = []
                anomaly_decision = []
                anomaly_true = []

                # max_score = torch.tensor(float('-inf'), dtype=torch.float32)
                # min_score = torch.tensor(float('inf'), dtype=torch.float32)

                normal_scores = []    # anomaly scores of normal data
                abnormal_scores = []  # anomaly scores of abnormal data
                
                with tqdm(data_loader['test'], bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}') as pbar:
                    for x, s, y in pbar:  # data, section id, anomaly
                        x = x.to(DEVICE).float()
                        s = s.to(DEVICE).long()
                        y = y.to(DEVICE).long()

                        padded, fake, latent_i, latent_o = model(x)
                        anomaly_scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=[1,2,3])
                        # anomaly_scores_all.extend(anomaly_scores)
                        
                        for i in range(len(y)):
                            anomaly_scores_all.append(anomaly_scores[i].detach().cpu())
                            if y[i] == 0:
                                normal_scores.append(anomaly_scores[i].detach().cpu())
                            else:
                                abnormal_scores.append(anomaly_scores[i].detach().cpu())

                        # max_score = max(max_score, torch.max(anomaly_scores))
                        # min_score = min(min_score, torch.min(anomaly_scores))

                        anomaly_true.extend(y.cpu())

                    min_score = np.min(anomaly_scores_all)
                    max_score = np.max(anomaly_scores_all)

                    if min_score < 0:
                        max_score -= min_score
                        min_score = 0

                    # anomaly_scores_all = torch.div(
                    #     torch.sub(anomaly_scores_all, min_score), torch.sub(max_score, min_score)
                    # )
                    # normal_scores = torch.div(
                    #     torch.sub(torch.as_tensor(normal_scores), min_score), torch.sub(max_score, min_score)
                    # )
                    # abnormal_scores = torch.div(
                    #     torch.sub(torch.as_tensor(abnormal_scores), min_score), torch.sub(max_score, min_score)
                    # )
                    anomaly_scores_all = np.divide(
                        np.subtract(anomaly_scores_all, min_score), np.subtract(max_score, min_score)
                    )
                    normal_scores = np.divide(
                        np.subtract(normal_scores, min_score), np.subtract(max_score, min_score)
                    )
                    abnormal_scores = np.divide(
                        np.subtract(abnormal_scores, min_score), np.subtract(max_score, min_score)
                    )

                    for score in anomaly_scores_all:
                        if score > decision_threshold:
                            anomaly_decision.append(1)
                        else:
                            anomaly_decision.append(0)


                eval_scores = util.calc_evaluation_scores(y_true=anomaly_true,
                                                          y_pred=anomaly_decision,
                                                          decision_threshold=decision_threshold,
                                                          config=config)
                auc, p_auc, precision, recall, f1_score = eval_scores

                with open(csv_logdir, 'a') as f:
                    section = section_name.split('_', 1)[1]
                    domain = dir_name.split('_', 1)[0]
                    f.write(f'{section},{domain},{auc},{p_auc},{precision},{recall},{f1_score}\n')
  
                ####### ***** neptune ***** #######
                run[f'test/{machine_type}/{domain}_domain/section{section}/AUC'].log(auc)
                run[f'test/{machine_type}/{domain}_domain/section{section}/pAUC'].log(p_auc)
                run[f'test/{machine_type}/{domain}_domain/section{section}/precision'].log(precision)
                run[f'test/{machine_type}/{domain}_domain/section{section}/recall'].log(recall)
                run[f'test/{machine_type}/{domain}_domain/section{section}/F1_score'].log(f1_score)
                
                fig = util.plot_anomaly_score_distrib(normal_scores, abnormal_scores, epoch, decimals=2, show=False)
                run[f'test/{machine_type}/{domain}_domain/section{section}/anomaly_score'].log(fig)
                plt.close()

                performance['section'].append(eval_scores)
                performance['all'].append(eval_scores)
            
            csv_lines = util.calc_performance_section(performance['section'])
            with open(csv_logdir, 'a') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(csv_lines)

        # del model, dcase_dataset, data_loader

        csv_lines = util.calc_performance_all(performance['all'])
        with open(csv_logdir, 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(csv_lines)

        ####### ***** neptune ***** #######
        # run[f'test/{machine_type}/epoch{epoch}'].upload(csv_logdir)
        run.stop()
        ####### ***** ******* ***** #######


if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)  
    
    max_epoch = args.max_epoch
    for i in range(1, max_epoch+1):
        main(config, i, decision_threshold=0.1)