import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from sklearn.svm import LinearSVC
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from utils.ops import Cutout

EPSILON = 1e-8


class Online_Static(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self._means = []
        self._svm_accs = []
        self.t_means = []
        s_model_type = args["convnet_type"]
        args["convnet_type"] = args["teacher_model"]
        self.t_model = IncrementalNet(args, False)
        args["convnet_type"] = s_model_type

        self.setup_dist_func(args)

    def after_task(self):
        self._known_classes = self._total_classes
        self.save_checkpoint("{}_{}_{}".format(self.args["model_name"], self.args["init_cls"], self.args["increment"]))

    def freeze_teacher(self):
        self.t_model.eval()
        for k, v in self.t_model.named_parameters():
            v.requires_grad = False

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self.data_augment()

        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for p in self._network.convnet.parameters():
                p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"],
            pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task == 0:
            self.t_model.update_fc(self._total_classes)
            self.t_model.to(self._device)
            self._train_teacher(self.train_loader, self.test_loader)
        
        self.freeze_teacher()

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            self.t_model = self.t_model.module

    def _train_teacher(self, train_loader, test_loader):
        logging.info('Training teacher model')
        self.t_model.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        t_epochs = self.args["t_epochs"]
        optimizer = optim.SGD(self.t_model.parameters(), momentum=0.9, lr=self.args["t_lr"], weight_decay=self.args["t_weight_decay"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["t_milestones"], gamma=self.args["t_gamma"])
        prog_bar = tqdm(range(t_epochs))
        for _, epoch in enumerate(prog_bar):
            self.t_model.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                logits = self.t_model(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => cls_Loss {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self.t_model, test_loader)
                info = 'Task {}, Epoch {}/{} => cls_Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
        logging.info('Teacher model training finished')

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module

        if self._cur_task == 0:
            self._epoch_num = self.args["init_epochs"]
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["gamma"])
            self._train_base_function(train_loader, test_loader, optimizer, scheduler)
            self._compute_means(teacher=True)
            self._build_feature_set()
        elif self._cur_task == 1:
            self._epoch_num = self.args["epochs"]
            self._compute_means(teacher=True)
            self._compute_relations()
            self._build_feature_set(teacher=True)
            train_loader = DataLoader(self._feature_trainset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
            optimizer = optim.SGD(self._network_module_ptr.fc.parameters(), momentum=0.9, lr=self.args["lr"], weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
            self.task1_train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            self._epoch_num = self.args["epochs"]
            self._compute_means()
            self._compute_relations()
            self._build_feature_set()
            train_loader = DataLoader(self._feature_trainset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
            optimizer = optim.SGD(self._network_module_ptr.fc.parameters(), momentum=0.9, lr=self.args["lr"], weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args["epochs"])
            self.inc_train_function(train_loader, test_loader, optimizer, scheduler)
        self._train_svm(self._feature_trainset, self._feature_testset)

    def _compute_means(self, teacher=False):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)
                if teacher:
                    t_vectors, _ = self._extract_vectors_teacher(idx_loader)
                    t_mean = np.mean(t_vectors, axis=0)
                    self.t_means.append(t_mean)

    def _compute_relations(self):
        old_means = np.array(self._means[:self._known_classes])
        new_means = np.array(self._means[self._known_classes:])
        self._relations = np.argmax((old_means / np.linalg.norm(old_means, axis=1)[:, None]) @ (new_means / np.linalg.norm(new_means, axis=1)[:, None]).T, axis=1) + self._known_classes

    def _build_feature_set(self, teacher=False):
        self.vectors_train = []
        self.labels_train = []
        teacher_vectors_train = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                       source='train',
                                                                       mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            self.vectors_train.append(vectors)
            self.labels_train.append([class_idx] * len(vectors))
            if teacher:
                t_vectors, _ = self._extract_vectors_teacher(idx_loader)
                teacher_vectors_train.append(t_vectors)
        for class_idx in range(0, self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(self.vectors_train[new_idx - self._known_classes] - self._means[new_idx] + self._means[class_idx])
            self.labels_train.append([class_idx] * len(self.vectors_train[-1]))
            if teacher:
                teacher_vectors_train.append(teacher_vectors_train[new_idx - self._known_classes] - self.t_means[new_idx] + self.t_means[class_idx])

        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        if teacher:
            teacher_vectors_train = np.concatenate(teacher_vectors_train)
            self._feature_trainset = Mixed_FeatureDataset(self.vectors_train, self.labels_train, teacher_vectors_train)
        else:
            self._feature_trainset = Stu_FeatureDataset(self.vectors_train, self.labels_train)

        self.vectors_test = []
        self.labels_test = []
        for class_idx in range(0, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                       source='test',
                                                                       mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            self.vectors_test.append(vectors)
            self.labels_test.append([class_idx] * len(vectors))
        self.vectors_test = np.concatenate(self.vectors_test)
        self.labels_test = np.concatenate(self.labels_test)

        self._feature_testset = Stu_FeatureDataset(self.vectors_test, self.labels_test)

    def _train_base_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            if self._cur_task == 0:
                self._network.train()
            else:
                self._network.eval()
            cls_losses = 0.
            kd_losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                output = self._network(inputs)
                logits = output['logits']
                t_logits = self.t_model(inputs)["logits"]
                cls_loss = F.cross_entropy(logits, targets)
                kd_loss = self.dist_func(logits, t_logits)
                loss = cls_loss + self.args["kd_coef"] * kd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += cls_loss.item()
                kd_losses += kd_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => cls_Loss {:.3f}, kd_loss {:.5f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, cls_losses / len(train_loader), kd_losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => cls_Loss {:.3f}, kd_loss {:.5f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, cls_losses / len(train_loader), kd_losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def task1_train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            cls_losses = 0.
            reg_losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets, t_inputs) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                t_inputs = t_inputs.to(self._device, non_blocking=True)
                logits = self._network_module_ptr.fc(inputs)['logits']
                t_output = self.t_model.fc(t_inputs)['logits']
                cls_loss = F.cross_entropy(logits, targets)
                reg_loss = self.dist_func(logits[:, : self._known_classes], t_output)
                loss = cls_loss + self.args["kd_coef"] * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += loss.item()
                reg_losses += reg_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => cls_Loss {:.3f}, kd_loss {:.5f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, cls_losses / len(train_loader), reg_losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => cls_Loss {:.3f}, kd_loss {:.5f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self._epoch_num, cls_losses / len(train_loader), reg_losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def inc_train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                logits = self._network_module_ptr.fc(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct) * 100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch + 1, self._epoch_num, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _train_svm(self, train_set, test_set):
        train_features = train_set.features.numpy()
        train_labels = train_set.labels.numpy()
        test_features = test_set.features.numpy()
        test_labels = test_set.labels.numpy()
        train_features = train_features / np.linalg.norm(train_features, axis=1)[:, None]
        test_features = test_features / np.linalg.norm(test_features, axis=1)[:, None]
        svm_classifier = LinearSVC(random_state=42)
        svm_classifier.fit(train_features, train_labels)
        logging.info("svm train: acc: {}".format(
            np.around(svm_classifier.score(train_features, train_labels) * 100, decimals=2)))
        acc = svm_classifier.score(test_features, test_labels)
        self._svm_accs.append(np.around(acc * 100, decimals=2))
        logging.info("svm evaluation: acc_list: {}".format(self._svm_accs))

    @torch.no_grad()
    def _extract_vectors_teacher(self, loader):
        self.t_model.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self.t_model, nn.DataParallel):
                _vectors = tensor2numpy(
                    self.t_model.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self.t_model.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def data_augment(self):
        if self.args["dataset"] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.Pad(4),
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ]
            
        elif self.args["dataset"] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.Pad(4),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ]
           


class Mixed_FeatureDataset(Dataset):
    def __init__(self, features, labels, t_features):
        assert len(features) == len(labels) == len(t_features), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.t_features = torch.from_numpy(t_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        t_features = self.t_features[idx]

        return idx, feature, label, t_features

class Stu_FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label
