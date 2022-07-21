import os
import yaml
import argparse
import torch
import util
from data_utils import *
from models import *
import joblib


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine_type', nargs='*', type=str,
                     help='Choose 0 or more values from [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(config):
    """Perform model training and validation"""
    
    # make directory to save train logs
    os.makedirs(config['train_log_dir'], exist_ok=True)
    os.makedirs(config['model_save_dir'], exist_ok=True)

    # load base directory list
    dir_list = select_dirs(config=config, mode='dev')
    if args.machine_type is not None:
        dir_list = [d for d in dir_list if os.path.basename(d) in args.machine_type]

    for idx, target_dir in enumerate(dir_list):
        print("====================================================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

        machine_type = os.path.basename(target_dir)
        csv_logdir = os.path.join(config['train_log_dir'], f'{machine_type}_train_log.csv')

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

        print("================= TRAINING ===================")
        train_losses = []  # train loss for all epochs
        val_losses = []    # validation loss for all epochs
        val_accs = []      # section id classification accuracy for all epochs

        for epoch in range(1, 1+config['training']['epochs']):
            # ========== train ==========
            model.train()
            train_loss = []
            with tqdm(data_loader['train'], ncols=100) as pbar:
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
            with torch.no_grad():
                with tqdm(data_loader['val'], ncols=100) as pbar:
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

                        score = util.calc_anomaly_score(out, int(s[0]))
                        anomaly_scores.append(score)

                    pbar.set_postfix({'epoch': epoch, 'val_loss': np.mean(val_loss), 'val_acc': val_acc})

                val_loss = np.mean(val_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            gamma_params = util.fit_gamma_dist(anomaly_scores, machine_type, epoch, config)
            # decision_threshold = util.calc_decision_threshold(machine_type, epoch, config)
            
            print(f'[Epoch {epoch:3d}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc:{val_acc:.2f}%\n')

            with open(csv_logdir, 'a') as f:
                f.write(f'{epoch},{train_loss},{val_loss},{val_acc}\n')

            util.visualize(train_losses, val_losses, plot_logdir)

            ckpt_path = os.path.join(config['model_save_dir'], f'model_{machine_type}_epoch{epoch}.pt')
            torch.save(model.state_dict(), ckpt_path)

        del dcase_dataset, data_loader

        print("================ SAVE MODEL ================")
        util.save_model(model, model_save_dir=config['model_save_dir'], machine_type=machine_type)
        print("====================================================================")


if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    
    main(config)