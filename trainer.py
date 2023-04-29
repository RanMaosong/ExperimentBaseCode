import os
import time

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import torchvision
from loguru import logger

from model.minist_classifier import MinistClassifier
from util.setup import setup_seed
from options import get_formated_args


def main(args):
    setup_seed(args.seed)
    device = args.device
    batch_size = args.batch_size

    dir_name = "lr<{}>_seed<{}>_optim<{}>_criterion<{}>_dataset<{}>".format(
        args.lr, 
        args.seed,
        args.optimizer,
        args.criterion,
        args.dataset)
    checkpoint = os.path.join(args.project_root_path, "checkpoints")
    args.checkpoint = os.path.join(checkpoint, args.model_name, dir_name)
    logger.info("checkpoint path root: {}".format(args.checkpoint))

    logger.info(get_formated_args(args))
    time.sleep(5) # for checking wherther the args is right!

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    if args.log_to_file:
        path = os.path.join(args.checkpoint, "log.txt")
        logger.add(path)

        logger.info(get_formated_args(args))

    # ################################ setup dataset ##########################
    if args.dataset == "minist_dataset":
        data_path = os.path.join(args.project_root_path, "data")
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
        valset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    else:
        raise "Unrecognise the model name: {}".format(args.dataset)
    # ############################## 1.setup model ##############################
    if args.model_name == "minist_classifier":
        model = MinistClassifier(args)
    else:
        raise "Unrecognise the model name: {}".format(args.model_name)
    
    # ############################## 2.setup optimizer ##########################
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), args.lr)
        model.optimizer = optimizer
    else:
        raise "Unrecognise the optimizer: {}".format(args.optimizer)

    # ############################## 3.setup criterion ##########################
    if args.criterion == "cross_entropy":
        model.criterion = F.cross_entropy
    else:
        raise "Unrecognise the criterion: {}".format(args.criterion)

    # ############################## 4.setup metrics ##########################
    logger.info(model)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    model.metrics = [accuracy]

    # ############################## 5.initialize ##########################
    if args.param_path is not None:
        model.restore_state_dict(args.param_path)
    else:
        model.initialize()

    # ############################## 6. setup dataloader ##########################
    trainset_dataloader = DataLoader(trainset, batch_size)
    valset_dataloader = DataLoader(valset, batch_size)

    # ############################## 7.training ##########################
    model.to(device)
    model.fit(trainset_dataloader, valset_dataloader, args.epochs, args.print_by_step)
    


if __name__ == "__main__":
    pass