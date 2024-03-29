from yaml import safe_load
import torch
import torch.nn.functional as F


def get_config(model_name):
    config_path = f'./models/{model_name}/config.yaml'
    with open(config_path) as f:
        config = safe_load(f)
    return config

def get_test_config(run_name):
    config_path = f'./results/{run_name}/config.yaml'
    with open(config_path) as f:
        config = safe_load(f)
    return config

def get_loss(config, x, y, output, device):
    """
    Args:
        x: input image
        y: true class
        output: predicted class or reconstructed image
    """
    if config['model']['task'] == 'classification':
        p = F.one_hot(y, config['model']['n_class'])
        p = p.float().to(device)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['settings']['label_smoothing'])
        loss = criterion(output, p)
    elif config['model']['task'] == 'reconstruction':
        criterion = torch.nn.MSELoss()
        loss = criterion(output, x)
    return loss

def get_anomaly_score(config, x, output):
    """
    Args:
        x: input image
        output: classification model output (i.e. class probability)
    """
    if config['model']['task'] == 'classification':
        softmaxprob = torch.softmax(output, dim=1)
        max_confidence = torch.max(softmaxprob, dim=1).values
        anomaly_score = torch.tensor(1) - max_confidence
    elif config['model']['task'] == 'reconstruction':
        anomaly_score = torch.mean(torch.pow((output - x), 2), dim=[1,2,3])
    return anomaly_score.detach()

def reshape_gan_output(src, real, fake, input_dim):
    if input_dim > 1:
        src = torch.cat(torch.unbind(src, dim=0), dim=-1).unsqueeze(0)
        real = torch.cat(torch.unbind(real, dim=0), dim=-1).unsqueeze(0)
        fake = torch.cat(torch.unbind(fake, dim=0), dim=-1).unsqueeze(0)
    
    src = src[0,:,:]
    real = real[0,:,:]
    fake = fake[0,:,:]
    
    return src, real, fake 