import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.init as init
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter

from util.tools import get_class_or_function_name, remove_file
from util.logger import Recoder

class BaseModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.metrics = None
        self._total_step = 0
        self._checkpoint = args.checkpoint

        path = os.path.join(self._checkpoint, "log")
        self._writer = SummaryWriter(path)
        self._recoder = Recoder()

        self._cur_epoch = 0
        self._best_val_loss = -1

        self._args = args
    
    def fit(self, trainset:DataLoader, valset:DataLoader=None, epochs=1, print_by_step=-1):
        r"""performance the optimization phase

        Args:
            trainset: the trainset dataloader
            valset: the valset dataloader
            epochs: total training epochs
            print_by_step: how many step to display training information
        """
        self._trainset = trainset
        self._valset = valset
        
        for epoch in range(epochs):
            self._cur_epoch = epoch
            self.train_one_epoch(epoch, print_by_step)
            val_metric_values = self.val_one_epoch(epoch)
            self.save_state_dict(val_metric_values)


    def train_one_epoch(self, epoch, print_by_step):
        r"""performance the optimization phase one epoch

        Args:
            epoch: the currented epoch
            print_by_step: how many step to display training information

        """
        if self.criterion is None:
            raise "You must call 'model.criterion = criterion' to set criterion"

        if self.optimizer is None:
            raise "You must call 'model.optimizer = optimizer' to set optimizer"

        self.train()
        for cur_step, data in enumerate(tqdm(self._trainset, desc="Epoch={}".format(epoch))):
            # optimize the model
            _, pred, target = self.step(data)
            loss = self.compute_loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute metrics on current batch data
            metric_values = self.compute_metrics(pred, target)
            metric_values["loss"] = loss.cpu().item()
            self._writer.add_scalars("train", metric_values, self._total_step)

            # log the training information
            if cur_step % print_by_step == 0:
                metric_info = self.get_metric_info(metric_values)
                logger.info("Train: epoch={}, step={}, {}".format(epoch, cur_step, metric_info))
            
            self._total_step += 1

    def val_one_epoch(self, epoch):
        r"""valid performance on val dataset

        Args:
            epoch: the currented epoch
        """
        self.eval()
        for _, data in enumerate(self._valset):
            _, pred, target = self.step(data)
            loss = self.compute_loss(pred, target)
            metric_values = self.compute_metrics(pred, target)

            self._recoder.record("loss", loss.cpu().item())
            for key, value in metric_values.items():
                self._recoder.record(key, value)
        
        metric_mean_value = self._recoder.summary()
        self._writer.add_scalars("val", metric_mean_value, epoch)

        metric_info = self.get_metric_info(metric_mean_value)
        logger.info("Val: epoch={}, {}".format(epoch, metric_info))

        return metric_mean_value

    def predict(self, x):
        r"""perform forward on x, and filter some unneeded the output

        Args:
            x: the input of model
        """
        input = Variable(x).to(self._args.device)
        pred = self(input)
        return pred

    def step(self, data):
        r"""forward on the current batch data

        Args:
            data: the data from dataloader
        """
        input = Variable(data[0]).to(self._args.device)
        target = Variable(data[1]).to(self._args.device)
        pred = self(input)
        return input, pred, target

    def compute_metrics(self, pred, target):
        r"""compute the loos between the output and target

        Args:
            pred: the output of the model
            target: the corresponding target
        
        """
        if self.metrics is None:
            return
        metrics_values = {}

        pred_for_metric = self.get_pred_for_metric(pred) # maybe some of the output of the model is not used to compute metric
        for metric in self.metrics:
            key = get_class_or_function_name(metric)
            value = metric(pred_for_metric, target).cpu().item()
            metrics_values[key] = value
        
        return metrics_values

    def compute_loss(self, pred, target):
        r"""compute the loos between the output and target

        Args:
            pred: the output of the model
            target: the corresponding target
        
        """
        pred_for_loss = self.get_pred_for_loss(pred)    # maybe some of the output of the model is not used to compute loss
        loss = self.criterion(pred_for_loss, target)
        return loss
    
    def save_state_dict(self, cur_val_metrics):
        r"""save the state dict

        Args:
            cur_val_metrics: the metric value on val dataset, used to compare with the previous bestval metric values
        
        """
        def save(path):
            state_dicts = {
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": self._cur_epoch
                }
            if self.lr_scheduler is not None:
                state_dicts["le_scheduler"] = self.lr_scheduler.state_dict()
            torch.save(state_dicts, path)
        
        if self._best_val_loss == -1 or self._best_val_loss > cur_val_metrics["loss"]:
            path = os.path.join(self._checkpoint, "best_*.pkl")
            remove_file(path)

            self._best_val_loss = cur_val_metrics["loss"]
            file_name = "best_loss<{:.4f}>_epoch<{}>.pkl".format(self._best_val_loss, self._cur_epoch)
            path = os.path.join(self._checkpoint, file_name)
            save(path)
        
        path = os.path.join(self._checkpoint, "finall_*.pkl")
        remove_file(path)

        self._best_val_loss = cur_val_metrics["loss"]
        file_name = "finall_loss<{:.4f}>_epoch<{}>.pkl".format(self._best_val_loss, self._cur_epoch)
        path = os.path.join(self._checkpoint, file_name)
        save(path)

    def restore_state_dict(self, path):
        r"""restore state_dict from checkpoint

        Args:
            path: the path of the checkpoint
        """
        state_dicts = torch.load(path)
        self.load_state_dict(state_dicts["model_state_dict"])
        self.optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
        self._cur_epoch = state_dicts["epoch"]
        if "lr_scheduler" in state_dicts:
            self.lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])
        logger.info("Restore state dict from {}".format(path))
    
    def get_pred_for_loss(self, pred):
        """
        Usually, the model return multiple outpus for experimental convenience, e.g. extract the intermediate feature map, but only partial pred is used to compute loss.

        Args:
            pred: the output of the model
        """
        return pred

    def get_pred_for_metric(self, pred):
        """
        Usually, the model return multiple outpus for experimental convenience, e.g. extract the intermediate feature map, but only partial pred is used to compute metric.

        Args:
            pred: the output of the model
        """
        return pred
    
    def get_pred_for_vis(self, pred):
        r"""Usually, the model return multiple outpus for experimental convenience, e.g. extract the intermediate feature map, but only one pred is used to visual.

        Args:
            pred: the output of the model
        """
        return pred

    def get_metric_info(self, metric_values:dict):
        r"""convert the metric values dict to a formated string

        Args:
            metric_values: the computeed metric values, the key is the metric name, the value is the corresponding value
        
        Example:
            metric_values = {"loss": 3.333, "accuracy": 0.98}
            return "loss=0.333, accuracy=0.98"
        
        """
        info = []

        for key, value in metric_values.items():
            info.append("{}={:.4f}".format(key, value))

        return ", ".join(info)
    
    def initialize(self):
        def weight_init(m):
            '''
            Usage:
                model = Model()
                model.apply(weight_init)
            '''
            if isinstance(m, nn.Conv1d):
                init.normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose1d):
                init.normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose3d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
            elif isinstance(m, nn.LSTMCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
            elif isinstance(m, nn.GRUCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
        self.apply(weight_init)
        logger.info("Initialize model parameter......")

                