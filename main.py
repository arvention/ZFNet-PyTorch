import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import mkdir
from datetime import datetime
import zipfile
import torch
import numpy as np


def zipdir(path, ziph):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".py") or file.endswith("cfg"):
            ziph.write(os.path.join(path, file))
            if file.endswith("cfg"):
                os.remove(file)


def save_config(config):
    current_time = str(datetime.now()).replace(":", "_")
    save_name = "zfnet_files_{}.{}"
    with open(save_name.format(current_time, "cfg"), "w") as f:
        for k, v in sorted(args.items()):
            f.write('%s: %s\n' % (str(k), str(v)))

    zipf = zipfile.ZipFile(save_name.format(current_time, "zip"),
                           'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()

    return current_time


def str2bool(v):
    return v.lower() in ('true')


def main(version, config):
    # for fast training
    cudnn.benchmark = True

    # create directories if not exist
    mkdir(config.log_path)

    if config.mode == 'train':
        temp_save_path = os.path.join(config.model_save_path, version)
        mkdir(temp_save_path)

        data_loader = get_loader(config.data_path + config.train_data_path,
                                 config.train_x_key, config.train_y_key,
                                 config.batch_size, config.mode)
        solver = Solver(version, data_loader, vars(config))
        solver.train()
    elif config.mode == 'test':
        data_loader = get_loader(config.data_path + config.test_data_path,
                                 config.test_x_key, config.test_y_key,
                                 config.batch_size, config.mode)
        solver = Solver(version, data_loader, vars(config))
        solver.test()


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--class_count', type=int, default=256)

    # training settings
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pretrained_model', type=str,
                        default=None)

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # dataset
    parser.add_argument('--data_path', type=str, default='../../data/c256/')
    parser.add_argument('--train_data_path', type=str,
                        default='caltech_256_30_train_nobg_norm.hdf5')
    parser.add_argument('--train_x_key', type=str, default='train_x')
    parser.add_argument('--train_y_key', type=str, default='train_y')
    parser.add_argument('--test_data_path', type=str,
                        default='caltech_256_30_test_nobg_norm.hdf5')
    parser.add_argument('--test_x_key', type=str, default='test_x')
    parser.add_argument('--test_y_key', type=str, default='test_y')

    # path
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')

    # epoch step size
    parser.add_argument('--loss_log_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=1)
    parser.add_argument('--train_eval_step', type=int, default=1)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    version = save_config(config)
    main(version, config)
