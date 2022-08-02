import os
import yaml
import argparse
import torch
import util
from data_utils import *
from ganomaly_model import *
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
            f.write('epoch, train_loss_g, train_loss_d, val_loss_g, val_loss_d\n')
                # f.write(k + ',' + str(v) + '\n')
            # f.write('epoch,train_loss,val_loss,val_acc\n')

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
        model = GanomalyModel(
            input_size=(128,313),
            latent_vec_size=100,
            num_input_channels=1,
            n_features=64,
            extra_layers=0,
            add_final_conv_layer=True,   
        ).to(DEVICE)

        criterion_g = GeneratorLoss(wadv=1, wcon=50, wenc=1)
        criterion_d = DiscriminatorLoss()

        optimizer_g = torch.optim.Adam(params=model.generator.parameters(),
                                       lr=config['training']['learning_rate'],
                                       betas=(0.5, 0.999),
                                       )
        optimizer_d = torch.optim.Adam(params=model.discriminator.parameters(),
                                       lr=config['training']['learning_rate'],
                                       betas=(0.5, 0.999),
                                       )

        scheduler = None
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=1.5,
        #                                                        eta_min=0,
        #                                                        last_epoch=-1,
        #                                                        verbose=False)

        plot_logdir = os.path.join(config['train_log_dir'], f'{machine_type}_train.png')

        ####### ***** neptune ***** #######
        run[f'model'] = 'GANomaly'
        run[f'training/hyperparams/optimizer'] = 'Adam'
        run[f'training/hyperparams/scheduler'] = 'CosineAnnealingLR'
        ####### ***** ******* ***** #######

        print("================= TRAINING ===================")
        train_losses_g = []  # generator train loss for all epochs
        train_losses_d = []  # discriminator train loss for all epochs
        val_losses_g = []    # generator validation loss for all epochs
        val_losses_d = []    # discriminator validation loss for all epochs

        max_score = torch.tensor(float('-inf'), dtype=torch.float32)
        min_score = torch.tensor(float('inf'), dtype=torch.float32)

        for epoch in range(1, 1+config['training']['num_epochs']):
            # ========== train ==========
            model.train()
            train_loss_g = []
            train_loss_d = []
            with tqdm(data_loader['train'], ncols=100, leave=False, desc=f'Epoch {epoch:3d}') as pbar:
                for x, s, y in pbar:  # data, section id, anomaly
                    x = x.to(DEVICE).float()
                    # s = s.to(DEVICE).long()
                    # y = y.to(DEVICE).long()

                    optimizer_g.zero_grad()
                    optimizer_d.zero_grad()
                    
                    padded, fake, latent_i, latent_o = model(x)
                    pred_real, _ = model.discriminator(padded)
                    pred_fake, _ = model.discriminator(fake.detach())
                    loss_g_step, error_enc, error_con, error_adv = criterion_g(latent_i, latent_o, padded, fake, pred_real, pred_fake)
                    loss_d_step = criterion_d(pred_real, pred_fake)

                    loss_g_step.backward(retain_graph=True)
                    loss_d_step.backward()
                    optimizer_g.step()
                    optimizer_d.step()

                    train_loss_g.append(loss_g_step.item())
                    train_loss_d.append(loss_d_step.item())

                    ####### ***** neptune ***** #######
                    run[f'training/{machine_type}/train/batch/loss_g'].log(loss_g_step.item())
                    run[f'training/{machine_type}/train/batch/loss_d'].log(loss_d_step.item())

                    run[f'training/{machine_type}/train/batch/error_enc'].log(error_enc.item())
                    run[f'training/{machine_type}/train/batch/error_con'].log(error_con.item())
                    run[f'training/{machine_type}/train/batch/error_adv'].log(error_adv.item())
                    ####### ***** ******* ***** #######    

                if scheduler is not None:
                    scheduler.step()
                
                # pbar.set_postfix({'epoch': epoch, 'train_loss_g': np.mean(train_loss_g)})

            train_loss_g = np.mean(train_loss_g)
            train_losses_g.append(train_loss_g)
            train_loss_d = np.mean(train_loss_d)
            train_losses_d.append(train_loss_d)

            # ======== validation ======== 
            model.eval()
            val_loss_g = []
            val_loss_d = []
            anomaly_scores = []
            anomaly_decision = []
            anomaly_true = []

            # if epoch > 1:
            #     ## calculate decision threshold from gamma score_distr_*.pkl
            #     score_file_path = os.path.join(
            #         config['model_save_dir'], f'score_distr_{machine_type}_epoch{epoch-1}.pkl'
            #     )
            #     decision_threshold = util.calc_decision_threshold(score_file_path, config)
            # else:
            #     decision_threshold = 0.
            decision_threshold = 0.5

            with torch.no_grad():
                with tqdm(data_loader['val'], ncols=100, leave=False, desc=f'Epoch {epoch:3d}') as pbar:
                    for x, s, y in pbar:  # data, section id, anomaly
                        x = x.to(DEVICE).float()
                        s = s.to(DEVICE).long()
                        y = y.to(DEVICE).long()

                        padded, fake, latent_i, latent_o = model(x)
                        pred_real, _ = model.discriminator(padded)
                        pred_fake, _ = model.discriminator(fake.detach())
                        loss_g_step, error_enc, error_con, error_adv = criterion_g(latent_i, latent_o, padded, fake, pred_real, pred_fake)
                        loss_d_step = criterion_d(pred_real, pred_fake)

                        val_loss_g.append(loss_g_step.item())
                        val_loss_d.append(loss_d_step.item())

                        ####### ***** neptune ***** #######
                        run[f'training/{machine_type}/valid/batch/loss_g'].log(loss_g_step.item())
                        run[f'training/{machine_type}/valid/batch/loss_d'].log(loss_d_step.item())

                        run[f'training/{machine_type}/valid/batch/error_enc'].log(error_enc.item())
                        run[f'training/{machine_type}/valid/batch/error_con'].log(error_con.item())
                        run[f'training/{machine_type}/valid/batch/error_adv'].log(error_adv.item())
                        ####### ***** ******* ***** #######

                        # for i in range(len(x)):
                        #     # score = util.calc_anomaly_score(torch.unsqueeze(out[i], axis=0).detach(), int(s[i].item()))
                        #     score = torch.mean(torch.pow((latent_i[i] - latent_o[i]), 2), dim=1).view(-1)[:].cpu()
                        #     import pdb; pdb.set_trace()
                        #     anomaly_scores.append(score)

                        #     if score > decision_threshold:
                        #         anomaly_decision.append(1)
                        #     else:
                        #         anomaly_decision.append(0)

                        # anomaly_scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
                        anomaly_scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=[1,2,3])

                        # normalize anomaly scores to [0,1]
                        max_score = max(max_score, torch.max(anomaly_scores))
                        min_score = min(min_score, torch.min(anomaly_scores))
                        anomaly_scores = torch.div(
                            torch.sub(anomaly_scores, min_score), torch.sub(max_score, min_score)
                        )

                        for score in anomaly_scores:
                            if score > decision_threshold:
                                anomaly_decision.append(1)
                            else:
                                anomaly_decision.append(0)

                        anomaly_true.extend(y.cpu())

                    # pbar.set_postfix({'epoch': epoch, 'val_loss': np.mean(val_loss), 'val_acc': val_acc})
                
                eval_scores = util.calc_evaluation_scores(y_true=anomaly_true,
                                                          y_pred=anomaly_decision,
                                                          decision_threshold=decision_threshold,
                                                          config=config)
                auc, p_auc, precision, recall, f1_score = eval_scores

                val_loss_g = np.mean(val_loss_g)
                val_losses_g.append(val_loss_g)
                val_loss_d = np.mean(val_loss_d)
                val_losses_d.append(val_loss_d)

            ## fit gamma distribution and save to score_distr_{machine_type}_epoch{epoch}.pkl
            # gamma_params = util.fit_gamma_dist(anomaly_scores, machine_type, epoch, config)
            
            ## calculate decision threshold from gamma score_distr_*.pkl
            # decision_threshold = util.calc_decision_threshold(machine_type, epoch, config)
            
            # print(f'[Epoch {epoch:3d}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc:{val_acc:.2f}%\n')
            print(f'[Epoch {epoch:3d}] train_loss_g: {train_loss_g:.4f}, val_loss_g: {val_loss_g:.4f}')
            print(f'            train_loss_d: {train_loss_d:.4f}, val_loss_d: {val_loss_d:.4f}')


            with open(csv_logdir, 'a') as f:
                f.write(f'{epoch}, {train_loss_g}, {train_loss_d}, {val_loss_g}, {val_loss_d}\n')

            ####### ***** neptune ***** #######
            run[f'training/{machine_type}/train/epoch/loss_g'].log(train_loss_g)
            run[f'training/{machine_type}/valid/epoch/loss_g'].log(val_loss_g)
            run[f'training/{machine_type}/train/epoch/loss_d'].log(train_loss_d)
            run[f'training/{machine_type}/valid/epoch/loss_d'].log(val_loss_d)
            # run[f'training/{machine_type}/valid/epoch/accuracy'].log(val_acc)
            run[f'training/{machine_type}/valid/epoch/AUC'].log(auc)
            run[f'training/{machine_type}/valid/epoch/pAUC'].log(p_auc)
            run[f'training/{machine_type}/valid/epoch/precision'].log(precision)
            run[f'training/{machine_type}/valid/epoch/recall'].log(recall)
            run[f'training/{machine_type}/valid/epoch/F1_score'].log(f1_score)
            run[f'training/{machine_type}/valid/epoch/decision_threshold'].log(decision_threshold)
            ####### ***** ******* ***** #######    
            
            # util.visualize(train_losses, val_losses, plot_logdir)

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