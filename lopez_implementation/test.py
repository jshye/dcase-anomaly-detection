import os
import csv
from platform import machine
import yaml
import argparse
import joblib
import torch
import util
from data_utils import *
from models import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine_type', nargs='*', type=str,
                     help='Choose 0 or more values from [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]')
parser.add_argument('--ep', type=int, help='score_distr_file epoch', default=1)
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(config):
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

        csv_logdir = os.path.join(config['result_dir'], f'{machine_type}_result.csv')
        with open(csv_logdir, 'a') as f:
            f.write(f'{machine_type}\n')
            f.write('section,domain,AUC,pAUC,precision,recall,F1_score\n')
        
        print("================ LOAD MODEL =================")
        model_file = os.path.join(config['model_save_dir'], f'model_{machine_type}.pt')
        section_names_file_path = os.path.join(config['model_save_dir'], f'section_names_{machine_type}.pkl')
        trained_section_names = joblib.load(section_names_file_path)
        # n_sections = trained_section_names.shape[0]

        model = Lopez2021().to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(model_file))

        score_distr_file_path = os.path.join(config['model_save_dir'], f'score_distr_{machine_type}_epoch{args.ep}.pkl')
        decision_threshold = util.calc_decision_threshold(score_distr_file_path, config)
        
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
                anomaly_scores = []
                anomaly_decision = []
                anomaly_true = []
                with tqdm(data_loader['test']) as pbar:
                    for x, s, y in pbar:  # data, section id, anomaly
                        x = x.to(DEVICE).float()
                        s = s.to(DEVICE).long()
                        y = y.to(DEVICE).long()

                        out = model(x)

                        for i in range(len(out)):
                            score = util.calc_anomaly_score(torch.unsqueeze(out[i], axis=0).detach(), int(s[i].item()))
                            anomaly_scores.append(score)
                            if score > decision_threshold:
                                anomaly_decision.append(1)
                            else:
                                anomaly_decision.append(0)

                        anomaly_true.extend(y.cpu())

                eval_scores = util.calc_evaluation_scores(y_true=anomaly_true,
                                                          y_pred=anomaly_decision,
                                                          decision_threshold=decision_threshold,
                                                          config=config)
                auc, p_auc, precision, recall, f1_score = eval_scores

                with open(csv_logdir, 'a') as f:
                    section = section_name.split('_', 1)[1]
                    domain = dir_name.split('_', 1)[0]
                    f.write(f'{section},{domain},{auc},{p_auc},{precision},{recall},{f1_score}\n')

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


if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    main(config)