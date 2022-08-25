import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import utils
from data_utils import *
from models import *
import joblib

import neptune.new as neptune
from getpass import getpass


parser = argparse.ArgumentParser()
# parser.add_argument('--machine_type', '-m', nargs='*', type=str,
#                      help='Choose none or more values from [ToyCar, ToyTrain, fan, gearbox, pump, slider, valve]')
parser.add_argument('--save_ckpt', action='store_true', help='Save checkpoint every epoch')
parser.add_argument('--resume_run', type=str, default=None, help='Neptune run name')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    # dir_list = select_dirs(config=config, mode='dev')
    # if args.machine_type is not None:
    #     dir_list = [d for d in dir_list if os.path.basename(d) in args.machine_type]

    # for idx, target_dir in enumerate(dir_list):
    #     print("====================================================================")
    #     print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

    dir_list = select_dirs(config=config, mode='dev')
    in_dir_list = [d for d in dir_list if os.path.basename(d) in 'ToyCar']
    target_dir = in_dir_list[0]
    oe_dir_list = [d for d in dir_list if os.path.basename(d) in 'slider']
    outlier_dir = oe_dir_list[0]

    # target_dir = os.path.abspath("{base}/ToyCar".format(base=config["dev_directory"]))
    # outlier_dir = os.path.abspath("{base}/slider".format(base=config["dev_directory"]))

    machine_type = os.path.basename(target_dir)
    csv_logdir = os.path.join(config['train_log_dir'], f'{machine_type}_train_log.csv')

    outlier_machine_type = os.path.basename(outlier_dir)

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

    outlier_files, outlier_labels = file_list_generator(target_dir=outlier_dir,
                                                        section_name='*',
                                                        # section_name='section_00',
                                                        dir_name='train',
                                                        mode='dev')

    outlier_dataset = DcaseDataset(outlier_files,
                                   outlier_labels,
                                   config=config,
                                   machine_config=config[outlier_machine_type],
                                   transform=None)

    print("============ DATALOADER GENERATOR ============")
    dataloader = {'train': None, 'val': None}
    dataloader['train'], dataloader['val'] = get_dataloader(dcase_dataset,
                                                            config=config,
                                                            machine_type=machine_type)
    
    outlier_dataloader = {'train': None, 'val': None}
    outlier_dataloader['train'], outlier_dataloader['val'] = get_dataloader(outlier_dataset,
                                                                            config=config,
                                                                            machine_type=outlier_machine_type)

    print("================ BUILD MODEL =================")
    model = ResNet(n_class=3).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])

    optimizer = torch.optim.Adam(params=model.parameters(),
                                    weight_decay=config['training']['weight_decay'],
                                    lr=config['training']['learning_rate'])
    
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                        T_max=1.5,
    #                                                        eta_min=0,
    #                                                        last_epoch=-1,
    #                                                        verbose=False)

    plot_logdir = os.path.join(config['train_log_dir'], f'{machine_type}_train.png')

    ####### ***** neptune ***** #######
    run[f'model'] = 'ResNet'
    run[f'training/hyperparams/optimizer'] = 'Adam'
    run[f'training/hyperparams/scheduler'] = 'None'
    ####### ***** ******* ***** #######

    print("================= TRAINING ===================")

    train_losses = []  # train loss for all epochs
    in_losses = []     # in-distribution train loss for all epochs
    oe_losses = []     # out-of-distribution train loss for all epochs
    
    val_losses = []    # validation loss for all epochs
    val_in_losses = []     # in-distribution validation loss for all epochs
    val_oe_losses = []     # out-of-distribution validation loss for all epochs

    for epoch in range(1, config['training']['num_epochs']+1):
        # ========== train ==========
        model.train()
        train_loss = []
        in_loss = []
        oe_loss = []
        # with tqdm(dataloader['train'], ncols=100, leave=False, desc=f'Epoch {epoch:3d}') as pbar:
        for indist, oedist in tqdm(zip(dataloader['train'], outlier_dataloader['train']), total=len(dataloader['train']), leave=False):
            in_x, in_s, _ = indist  # spectrogram, section id, anomaly
            oe_x, _, _ = oedist
            
            in_x = in_x.to(DEVICE).float()
            in_s = in_s.to(DEVICE).long()
            oe_x = oe_x.to(DEVICE).float()

            ## in-distribution loss
            logits = model(in_x)
            p = F.one_hot(in_s, 3)
            p = p.float().to(DEVICE)
            in_loss_step = criterion(logits, p)
            in_loss.append(in_loss_step.cpu().detach())

            ## out-of-distribution loss
            outlier_logits = model(oe_x)
            p_oe = torch.ones(oe_x.shape[0], 3) / torch.tensor(3)  # uniform dist.
            p_oe = p_oe.float().to(DEVICE)
            oe_loss_step = criterion(outlier_logits, p_oe)
            oe_loss.append(oe_loss_step.cpu().detach())

            # joint training
            train_loss_step = in_loss_step + 0.5*oe_loss_step
            train_loss.append(train_loss_step.cpu().detach())
        
            optimizer.zero_grad()
            train_loss_step.backward()
            optimizer.step()

            ####### ***** neptune ***** #######
            run[f'training/{machine_type}/train/batch/loss'].log(train_loss_step.item())
            run[f'training/{machine_type}/train/batch/in_loss'].log(in_loss_step.item())
            run[f'training/{machine_type}/train/batch/oe_loss'].log(oe_loss_step.item())
            ####### ***** ******* ***** #######    

            if scheduler is not None:
                scheduler.step()
            
        train_loss_epoch = np.mean(train_loss)
        in_loss_epoch = np.mean(in_loss)
        oe_loss_epoch = np.mean(oe_loss)

        train_losses.append(train_loss_epoch)
        in_losses.append(in_loss_epoch)
        oe_losses.append(oe_loss_epoch)

        ####### ***** neptune ***** #######
        run[f'training/{machine_type}/train/epoch/loss'].log(train_loss_epoch)
        run[f'training/{machine_type}/train/epoch/in_loss'].log(in_loss_epoch)
        run[f'training/{machine_type}/train/epoch/oe_loss'].log(oe_loss_epoch)
        ####### ***** ******* ***** #######    


        # ======== validation ======== 

        model.eval()
        val_loss = []
        val_in_loss = []
        val_oe_loss = []
        val_accs = []      # in-distribution section id classification accuracy for all epochs
        correct = 0
        total = 0

        for indist, oedist in tqdm(zip(dataloader['val'], outlier_dataloader['val']), total=len(dataloader['val']), leave=False):
            in_x, in_s, _ = indist  # spectrogram, section id, anomaly
            oe_x, _, _ = oedist
            
            in_x = in_x.to(DEVICE).float()
            in_s = in_s.to(DEVICE).long()
            oe_x = oe_x.to(DEVICE).float()

            ## in-distribution loss
            logits = model(in_x)
            _, in_s_pred = torch.max(logits, 1) 
            p = F.one_hot(in_s, 3)
            p = p.float().to(DEVICE)
            val_in_loss_step = criterion(logits, p)
            val_in_loss.append(val_in_loss_step.cpu().detach())
            
            correct += (in_s == in_s_pred).sum().item()
            total += in_s.size(0)
            val_acc = 100 * (correct / total)

            ## out-of-distribution loss
            outlier_logits = model(oe_x)
            p_oe = torch.ones(oe_x.shape[0], 3) / torch.tensor(3)  # uniform dist.
            p_oe = p_oe.float().to(DEVICE)
            val_oe_loss_step = criterion(outlier_logits, p_oe)
            val_oe_loss.append(val_oe_loss_step.cpu().detach())

            # joint loss
            val_loss_step = val_in_loss_step + 0.5*val_oe_loss_step  # 0.5 for visual task
            val_loss.append(val_loss_step.cpu().detach())

            ####### ***** neptune ***** #######
            run[f'training/{machine_type}/valid/batch/loss'].log(val_loss_step.item())
            run[f'training/{machine_type}/valid/batch/in_loss'].log(val_in_loss_step.item())
            run[f'training/{machine_type}/valid/batch/oe_loss'].log(val_oe_loss_step.item())
            ####### ***** ******* ***** #######    

        val_accs.append(val_acc)
        val_loss_epoch = np.mean(val_loss)
        val_in_loss_epoch = np.mean(val_in_loss)
        val_oe_loss_epoch = np.mean(val_oe_loss)

        val_losses.append(val_loss_epoch)
        val_in_losses.append(val_in_loss_epoch)
        val_oe_losses.append(val_oe_loss_epoch)

        ####### ***** neptune ***** #######
        run[f'training/{machine_type}/valid/epoch/loss'].log(val_loss_epoch)
        run[f'training/{machine_type}/valid/epoch/in_loss'].log(val_in_loss_epoch)
        run[f'training/{machine_type}/valid/epoch/oe_loss'].log(val_oe_loss_epoch)
        run[f'training/{machine_type}/valid/epoch/accuracy'].log(val_acc)
        ####### ***** ******* ***** #######    

        print(f'[Epoch {epoch:3d}] train_loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}, val_acc:{val_acc:.2f}%')

        with open(csv_logdir, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss},{val_acc}\n')

        
        utils.visualize(train_losses, val_losses, plot_logdir)

        if args.save_ckpt:
            ckpt_path = os.path.join(config['model_save_dir'], f'model_{machine_type}_epoch{epoch}.pt')
            torch.save(model.state_dict(), ckpt_path)

    del dcase_dataset, dataloader

    print("================ SAVE MODEL ================")
    utils.save_model(model, model_save_dir=config['model_save_dir'], machine_type=machine_type)
    print("====================================================================")

    ####### ***** neptune ***** #######
    run.stop()
    ####### ***** ******* ***** #######  


if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    
    main(config)