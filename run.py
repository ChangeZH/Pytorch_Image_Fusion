import torch
import argparse
from core.model import *
from tools import train, test
from core.dataset import Fusion_Datasets
import torchvision.transforms as transforms
from core.util import load_config, count_parameters


def get_args():
    parser = argparse.ArgumentParser(description='run')

    parser.add_argument('--config', type=str, default='./config/DenseFuse.yaml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()
    return args


def runner(args):
    configs = load_config(args.config)
    project_configs = configs['PROJECT']
    model_configs = configs['MODEL']
    train_configs = configs['TRAIN']
    test_configs = configs['TEST']
    train_dataset_configs = configs['TRAIN_DATASET']
    test_dataset_configs = configs['TEST_DATASET']
    input_size = train_dataset_configs['input_size'] if args.train else test_dataset_configs['input_size']

    if train_dataset_configs['channels'] == 3:
        base_transforms = transforms.Compose(
            [transforms.Resize((input_size, input_size)),
             transforms.ToTensor()])  # ,
        # transforms.Normalize(mean=train_dataset_configs['mean'], std=train_dataset_configs['std'])])
    elif train_dataset_configs['channels'] == 1:
        base_transforms = transforms.Compose(
            [transforms.Resize((input_size, input_size)),
             transforms.ToTensor()])  # ,
        # transforms.Normalize(mean=[sum(train_dataset_configs['mean']) / len(train_dataset_configs['mean'])],
        #                      std=[sum(train_dataset_configs['std']) / len(train_dataset_configs['std'])])])

    train_datasets = Fusion_Datasets(train_dataset_configs, base_transforms)
    test_datasets = Fusion_Datasets(test_dataset_configs, base_transforms)

    model = eval(model_configs['model_name'])(model_configs)
    print('Model Para:', count_parameters(model))

    if train_configs['resume'] != 'None':
        checkpoint = torch.load(train_configs['resume'])
        model.load_state_dict(checkpoint['model'].state_dict())

    if args.train:
        train(model, train_datasets, test_datasets, configs)
    if args.test:
        test(model, test_datasets, configs, load_weight_path=True)


if __name__ == '__main__':
    args = get_args()
    runner(args)
