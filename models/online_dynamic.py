import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, load_json, _set_device
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from utils.ops import Cutout
from utils.factory import get_model

class Online_Dynamic(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        t_args = argparse.Namespace(config="./exps/{}.json".format(args["teacher_method"]))
        param = load_json("./exps/{}.json".format(args["teacher_method"]))
        t_args = vars(t_args) 
        t_args.update(param)
        assert t_args["init_cls"] == args["init_cls"]
        assert t_args["increment"] == args["increment"]
        _set_device(t_args)
        self.t_model = get_model(args["teacher_method"], t_args)

        self.setup_dist_func(args)

    def after_task(self):
        self._known_classes = self._total_classes
        self.save_checkpoint("{}_{}_{}".format(self.args["model_name"], self.args["init_cls"], self.args["increment"]))
        self.t_model.after_task()
        
    def freeze(self):
        self.t_model._network.eval()
        for k, v in self.t_model._network.named_parameters():
            v.requires_grad = False
    
    def activate(self):
        self.t_model._network.train()
        for k, v in self.t_model._network.named_parameters():
            v.requires_grad = True

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self.data_augment()
        
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        self.t_model._cur_task = self._cur_task
        self.t_model._total_classes = self._total_classes
        self.t_model._network.update_fc(self._total_classes)
        # t_classes = self.t_model._total_classes if self._cur_task == 0 else self.t_model._known_classes
        # self.t_model._network.update_fc(t_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self.t_model._network = nn.DataParallel(self.t_model._network, self._multiple_gpus)

        if self._cur_task == 0:
            self.t_model._train(self.train_loader, self.test_loader)
            self.freeze()
        self._train(self.train_loader, self.test_loader)
        self.activate()

        if self._cur_task > 0:
            self.t_model._train(self.train_loader, self.test_loader)
            self.freeze()

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            self.t_model._network = self.t_model._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self.t_model._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=45, gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                t_output = self.t_model._network(inputs)["logits"]

                kd_loss = self.dist_fn(logits, t_output)
                cls_loss = F.cross_entropy(logits, targets)
                loss = cls_loss + self.args["coef_kd"] * kd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                cls_loss = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                reg_loss = self.dist_fn(
                    logits[:, : self._known_classes],
                    self.t_model._network(inputs)["logits"][:, : self._known_classes],
                )

                loss = self.args["coef_reg"] * reg_loss + cls_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def data_augment(self):
        if self.args["dataset"] == 'cifar100':
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                ]

