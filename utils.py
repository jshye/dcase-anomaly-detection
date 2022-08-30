import os
from pathlib import Path
from yaml import safe_load
# from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
import torch
import torch.nn.functional as F


def get_config(model_name):
    # root = Path(__file__).resolve().parents[1]
    # config_path = Path(root, 'models', model_name, 'config.yaml')
    config_path = f'./models/{model_name}/config.yaml'
    with open(config_path) as f:
        config = safe_load(f)
    return config

def get_test_config(run_name):
    config_path = f'./results/{run_name}/config.yaml'
    with open(config_path) as f:
        config = safe_load(f)
    return config

# def get_model(model_name, config):
#     model_dict = {
#         'resnet': 'ResNet',
#         'ganomaly': 'Ganomaly',
#     }
#     if model_name in model_dict.keys():
#         # root = Path(__file__).resolve().parents[1]
#         # module_path = Path(root, 'models', model_name)
#         # module_path = f'/models/{model_name}/{model_name}'
#         # module = import_module(module_path, package='./')
#         # model = getattr(module, model_dict[model_name])(config)
#         spec = spec_from_file_location(model_name, f'./models/{model_name}/{model_name}.py')
#         module = module_from_spec(spec)
#         spec.loader.exec_module(module)
#         init_model = f'module.{model_dict[model_name]}(config)'
#         return exec(init_model)
#     else:
#         raise ValueError(f"'{model_name}' Not Implemented!")


def get_loss(config, x, y, output, device):
    if config['model']['task'] == 'classification':
        p = F.one_hot(y, config['model']['n_class'])
        p = p.float().to(device)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['settings']['label_smoothing'])
        loss = criterion(output, p)
    elif config['model']['task'] == 'reconstruction':
        criterion = torch.nn.MSELoss()
        loss = criterion(output, x)
    return loss

# def get_criterion(config):
#     if config.model.task == 'classificatoin':
#         criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.settings.label_smoothing)
#     elif config.model.task == 'reconstruction':
#         criterion = torch.nn.MSELoss()
#     return criterion

def get_anomaly_score(config, x, y, output, device):
    if config['model']['task'] == 'classification':
        softmaxprob = torch.softmax(output, dim=1)
        max_confidence = torch.max(softmaxprob, dim=1).values
        anomaly_score = torch.tensor(1) - max_confidence
        # n_class = config['model']['n_class']
        # p = torch.ones(x.shape[0], n_class) / torch.tensor(n_class)  # uniform distribution
        # p = p.float().to(device)
        # ce = torch.nn.CrossEntropyLoss()(output, p)
        # anomaly_score = 1 - ce
    elif config['model']['task'] == 'reconstruction':
        anomaly_score = torch.nn.MSELoss(reduction='none')(output, x)
    return anomaly_score.detach()