import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from .loss_kd import loss_fn_kd
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils


class XLAModelTrainer(object):
    @staticmethod
    def train(dataloader, model, loss_f, optimizer, device, epoch_id, max_epoch, num_cls, print_iter=10, log_file=None):
        def reduce_fn(vals):
            # take average
            return sum(vals) / len(vals)

        model.train()

        conf_mat = torch.zeros((num_cls, num_cls))
        loss_rec = []

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer)

            _, predictions = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                pred_i = predictions[j]
                cls_i = labels[j]
                conf_mat[cls_i, pred_i] += 1

            loss_reduced = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
            loss_rec.append(loss_reduced)
            avg_acc = conf_mat.trace() / conf_mat.sum()
            avg_acc_reduced = xm.mesh_reduce('acc_reduce', avg_acc, reduce_fn)

            if i % print_iter == 0:
                xm.master_print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch_id + 1, max_epoch, i + 1, len(dataloader),
                        torch.mean(torch.tensor(loss_rec)), avg_acc))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        xm.master_print(
                            "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                                epoch_id + 1, max_epoch, i + 1, len(dataloader),
                                torch.mean(torch.tensor(loss_rec)), avg_acc), fd=f)

        return torch.mean(torch.tensor(loss_rec)), avg_acc_reduced, conf_mat

    @staticmethod
    def valid(dataloader, model, loss_f, device, num_cls):
        def reduce_fn(vals):
            # take average
            return sum(vals) / len(vals)

        model.eval()

        conf_mat = torch.zeros((num_cls, num_cls))
        loss_rec = []

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                loss = loss_f(outputs, labels)
                _, predictions = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cls_i = labels[j]
                pred_i = predictions[j]
                conf_mat[cls_i, pred_i] += 1

            loss_reduced = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
            loss_rec.append(loss_reduced)
            avg_acc = conf_mat.trace() / conf_mat.sum()
            avg_acc_reduced = xm.mesh_reduce('acc_reduce', avg_acc, reduce_fn)

        return torch.mean(torch.tensor(loss_rec)), avg_acc_reduced, conf_mat

    @staticmethod
    def train_mc(dataloader, model, loss_f, optimizer, device, epoch_id, max_epoch, num_cls, print_iter=10,
                 mix_up=False, log_file=None):
        def reduce_fn(vals):
            # take average
            return sum(vals) / len(vals)

        model.train()

        correct_pred = torch.tensor(0)
        total_samples = torch.tensor(0)
        loss_rec = []
        f1_rec = []

        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if mix_up:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, device)
                outputs = model(inputs)
                loss = mixup_criterion(loss_f, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = loss_f(outputs, labels)

            loss.backward()
            xm.optimizer_step(optimizer)

            predictions = (torch.sigmoid(outputs.data) >= 0.5).to(torch.long)

            f1 = torch.tensor(f1_score(labels.cpu().detach(), predictions.cpu().detach(), average='samples'))
            f1_reduced = xm.mesh_reduce('f1_reduce', f1, reduce_fn)
            f1_rec.append(f1_reduced)

            for j, p in enumerate(predictions):
                if torch.sum(p == labels[j]) == num_cls:
                    correct_pred += 1

            total_samples += len(labels)

            avg_acc = correct_pred / total_samples
            avg_acc_reduced = xm.mesh_reduce('acc_reduce', avg_acc, reduce_fn)

            loss_reduced = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
            loss_rec.append(loss_reduced)

            if i % print_iter == 0:
                xm.master_print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%} F1:{:.4f}".format(
                        epoch_id + 1, max_epoch, i + 1, len(dataloader),
                        torch.mean(torch.tensor(loss_rec)), avg_acc, torch.mean(torch.tensor(f1_rec))))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        xm.master_print(
                            "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%} F1:{:.4f}".format(
                                epoch_id + 1, max_epoch, i + 1, len(dataloader),
                                torch.mean(torch.tensor(loss_rec)), avg_acc, torch.mean(torch.tensor(f1_rec))), fd=f)

        return torch.mean(torch.tensor(loss_rec)), avg_acc_reduced, torch.mean(torch.tensor(f1_rec))

    @staticmethod
    def valid_mc(dataloader, model, loss_f, device, epoch_id, max_epoch, num_cls, print_iter=10, log_file=None):
        def reduce_fn(vals):
            # take average
            return sum(vals) / len(vals)

        model.eval()

        correct_pred = torch.tensor(0)
        total_samples = torch.tensor(0)
        loss_rec = []
        f1_rec = []

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                loss = loss_f(outputs, labels)

            predictions = (torch.sigmoid(outputs.data) >= 0.5).to(torch.long)

            f1 = torch.tensor(f1_score(labels.cpu().detach(), predictions.cpu().detach(), average='samples'))
            f1_reduced = xm.mesh_reduce('f1_reduce', f1, reduce_fn)
            f1_rec.append(f1_reduced)

            for j, p in enumerate(predictions):
                if torch.sum(p == labels[j]) == num_cls:
                    correct_pred += 1

            total_samples += len(labels)

            avg_acc = correct_pred / total_samples
            avg_acc_reduced = xm.mesh_reduce('acc_reduce', avg_acc, reduce_fn)

            loss_reduced = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
            loss_rec.append(loss_reduced)

            if i % print_iter == 0:
                xm.master_print(
                    "Evaluating: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%} F1:{:.4f}".format(
                        epoch_id + 1, max_epoch, i + 1, len(dataloader),
                        torch.mean(torch.tensor(loss_rec)), avg_acc, torch.mean(torch.tensor(f1_rec))))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        xm.master_print(
                            "Evaluating: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%} F1:{:.4f}".format(
                                epoch_id + 1, max_epoch, i + 1, len(dataloader),
                                torch.mean(torch.tensor(loss_rec)), avg_acc, torch.mean(torch.tensor(f1_rec))), fd=f)

        return torch.mean(torch.tensor(loss_rec)), avg_acc_reduced, torch.mean(torch.tensor(f1_rec))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
