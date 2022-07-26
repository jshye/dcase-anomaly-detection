import os
import yaml
import argparse
import torch
import util
from data_utils import *
from models import *
import joblib

import neptune.new as neptune
from getpass import getpass


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine_type', nargs='*', type=str,
                     help='Choose 0 or more values from [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]')
parser.add_argument('--save_ckpt', action='store_true')
parser.add_argument('--resume_run', type=str, default=None, help='Neptune run name')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def main(config, machine_type=None, save_ckpt=True, resume_run=None):
def main(config):
    """Perform model training and validation"""

    ####### ***** neptune ***** #######
    NEPTUNE_API_TOKEN = getpass('Enter your private Neptune API token: ')
    if args.resume_run:
        run = neptune.init(
            project='seonghye/dcase-anomaly',
            api_token=NEPTUNE_API_TOKEN,
            run=args.resume_run,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
        )
    else:
        run = neptune.init(
            project='seonghye/dcase-anomaly',
            api_token=NEPTUNE_API_TOKEN,
            tags=['classification'],
            description=input('Enter Neptune Description:'),
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
        )

        run[f'training/hyperparams'] = config['training']
    ####### ***** ******* ***** #######

    ## make directory to save train logs
    os.makedirs(config['train_log_dir'], exist_ok=True)
    os.makedirs(config['model_save_dir'], exist_ok=True)

    ## load base directory list
    dir_list = select_dirs(config=config, mode='dev')
    if args.machine_type is not None:
        dir_list = [d for d in dir_list if os.path.basename(d) in args.machine_type]

    for idx, target_dir in enumerate(dir_list):
        print("====================================================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

        machine_type = os.path.basename(target_dir)
        csv_logdir = os.path.join(config['train_log_dir'], f'{machine_type}_train_log.csv')

        ####### ***** neptune ***** #######
        run[f'training/{machine_type}'] = config[machine_type]
        ####### ***** ******* ***** #######

        section_names_file_path = os.path.join(config['model_save_dir'], f'section_names_{machine_type}.pkl')
        unique_section_names = np.unique(get_section_names(target_dir, dir_name="train"))
        joblib.dump(unique_section_names, section_names_file_path)

        with open(csv_logdir, 'a') as f:
            f.write('config\n')
            for k, v in config.items():
                f.write(k + ',' + str(v) + '\n')
            f.write('epoch,train_loss,val_loss,val_acc\n')

        print("============= DATASET GENERATOR ==============")
        files, labels = file_list_generator(target_dir=target_dir,
                                            section_name='*',
                                            dir_name='train',
                                            mode='dev')

        dcase_dataset = DcaseDataset(files,
                                     labels,
                                     config=config,
                                     machine_config=config[machine_type],
                                     transform=None)

        print("============ DATALOADER GENERATOR ============")
        data_loader = {'train': None, 'val': None}
        data_loader['train'], data_loader['val'] = get_dataloader(dcase_dataset,
                                                                  config=config,
                                                                  machine_type=machine_type)
        print("================ BUILD MODEL =================")
        model = Lopez2021().to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     weight_decay=config['training']['weight_decay'],
                                     lr=config['training']['learning_rate'])
        
        # scheduler = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=1.5,
                                                               eta_min=0,
                                                               last_epoch=-1,
                                                               verbose=False)

        plot_logdir = os.path.join(config['train_log_dir'], f'{machine_type}_train.png')

        ####### ***** neptune ***** #######
        run[f'model'] = 'Lopez2021'
        run[f'training/hyperparams/optimizer'] = 'Adam'
        run[f'training/hyperparams/scheduler'] = 'CosineAnnealingLR'
        ####### ***** ******* ***** #######

        print("================= TRAINING ===================")
        train_losses = []  # train loss for all epochs
        val_losses = []    # validation loss for all epochs
        val_accs = []      # section id classification accuracy for all epochs

        for epoch in range(1, 1+config['training']['num_epochs']):
            # ========== train ==========
            model.train()
            train_loss = []
            with tqdm(data_loader['train'], ncols=100, leave=False, desc=f'Epoch {epoch:3d}') as pbar:
                for x, s, y in pbar:  # data, section id, anomaly
                    x = x.to(DEVICE).float()
                    s = s.to(DEVICE).long()
                    y = y.to(DEVICE).long()

                    optimizer.zero_grad()
                    
                    # train_loss_step = model.get_loss(x, y)
                    out = model(x)
                    train_loss_step = criterion(out, s)
                    train_loss_step.backward()
                    optimizer.step()

                    train_loss.append(train_loss_step.item())

                    ####### ***** neptune ***** #######
                    run[f'training/{machine_type}/train/batch/loss'].log(train_loss_step.item())
                    ####### ***** ******* ***** #######    

                if scheduler is not None:
                    scheduler.step()
                
                pbar.set_postfix({'epoch': epoch, 'train_loss': np.mean(train_loss)})

            train_loss = np.mean(train_loss)
            train_losses.append(train_loss)

            # ======== validation ======== 
            model.eval()
            val_loss = []
            correct = 0
            total = 0
            anomaly_scores = []
            anomaly_decision = []
            anomaly_true = []

            if epoch > 1:
                ## calculate decision threshold from gamma score_distr_*.pkl
                score_file_path = os.path.join(
                    config['model_save_dir'], f'score_distr_{machine_type}_epoch{epoch-1}.pkl'
                )
                decision_threshold = util.calc_decision_threshold(score_file_path, config)
            else:
                decision_threshold = 0.5

            with torch.no_grad():
                with tqdm(data_loader['val'], ncols=100, leave=False, desc=f'Epoch {epoch:3d}') as pbar:
                    for x, s, y in pbar:  # data, section id, anomaly
                        x = x.to(DEVICE).float()
                        s = s.to(DEVICE).long()
                        y = y.to(DEVICE).long()

                        # val_loss_step = model.get_loss(x, y)
                        out = model(x)
                        _, s_pred = torch.max(out, 1) 
                        val_loss_step = criterion(out, s)

                        val_loss.append(val_loss_step.item())
                        correct += (s == s_pred).sum().item()
                        total += s.size(0)
                        val_acc = 100 * (correct / total)

                        ####### ***** neptune ***** #######
                        run[f'training/{machine_type}/valid/batch/loss'].log(val_loss_step.item())
                        ####### ***** ******* ***** #######

                        # score = util.calc_anomaly_score(out, int(s[0]))
                        # anomaly_scores.append(score)
                        for i in range(len(out)):
                            score = util.calc_anomaly_score(torch.unsqueeze(out[i], axis=0).detach(), int(s[i].item()))
                            anomaly_scores.append(score)

                            if score > decision_threshold:
                                anomaly_decision.append(1)
                            else:
                                anomaly_decision.append(0)

                        anomaly_true.extend(y.cpu())

                    pbar.set_postfix({'epoch': epoch, 'val_loss': np.mean(val_loss), 'val_acc': val_acc})
                
                eval_scores = util.calc_evaluation_scores(y_true=anomaly_true,
                                                          y_pred=anomaly_decision,
                                                          decision_threshold=decision_threshold,
                                                          config=config)
                auc, p_auc, precision, recall, f1_score = eval_scores

                val_loss = np.mean(val_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            ## fit gamma distribution and save to score_distr_{machine_type}_epoch{epoch}.pkl
            gamma_params = util.fit_gamma_dist(anomaly_scores, machine_type, epoch, config)
            
            ## calculate decision threshold from gamma score_distr_*.pkl
            # decision_threshold = util.calc_decision_threshold(machine_type, epoch, config)
            
            print(f'[Epoch {epoch:3d}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc:{val_acc:.2f}%\n')

            with open(csv_logdir, 'a') as f:
                f.write(f'{epoch},{train_loss},{val_loss},{val_acc}\n')

            ####### ***** neptune ***** #######
            run[f'training/{machine_type}/train/epoch/loss'].log(train_loss)
            run[f'training/{machine_type}/valid/epoch/loss'].log(val_loss)
            run[f'training/{machine_type}/valid/epoch/accuracy'].log(val_acc)
            run[f'training/{machine_type}/valid/epoch/AUC'].log(auc)
            run[f'training/{machine_type}/valid/epoch/pAUC'].log(p_auc)
            run[f'training/{machine_type}/valid/epoch/precision'].log(precision)
            run[f'training/{machine_type}/valid/epoch/recall'].log(recall)
            run[f'training/{machine_type}/valid/epoch/F1_score'].log(f1_score)
            run[f'training/{machine_type}/valid/epoch/decision_threshold'].log(decision_threshold)
            ####### ***** ******* ***** #######    
            
            util.visualize(train_losses, val_losses, plot_logdir)

            if args.save_ckpt:
                ckpt_path = os.path.join(config['model_save_dir'], f'model_{machine_type}_epoch{epoch}.pt')
                torch.save(model.state_dict(), ckpt_path)

        del dcase_dataset, data_loader

        print("================ SAVE MODEL ================")
        util.save_model(model, model_save_dir=config['model_save_dir'], machine_type=machine_type)
        print("====================================================================")

        ####### ***** neptune ***** #######
        run.stop()
        ####### ***** ******* ***** #######  


if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    
    main(config)