import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from .loss_kd import loss_fn_kd


class ModelTrainer(object):
    @staticmethod
    def train(dataloader, model, loss_f, optimizer, device, epoch_id, max_epoch, num_cls, print_iter=10, mix_up=False, log_file=None):
        model.train()

        conf_mat = np.zeros((num_cls, num_cls))
        loss_rec = []

        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if mix_up:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels)
                outputs = model(inputs)
                loss = mixup_criterion(loss_f, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = loss_f(outputs, labels)


            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                pred_i = predictions[j].cpu().numpy()
                cls_i = labels[j].cpu().numpy()
                conf_mat[cls_i, pred_i] += 1

            loss_rec.append(loss.item())
            avg_acc = conf_mat.trace() / conf_mat.sum()

            if i % print_iter == 0:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                            epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc), file=f)

        return np.mean(loss_rec), avg_acc, conf_mat

    @staticmethod
    def train_kd(dataloader, model, teacher_model, T, alpha, optimizer, device, epoch_id, max_epoch, num_cls,
                 label_smoothing=True, print_iter=10, log_file=None):
        model.train()

        conf_mat = np.zeros((num_cls, num_cls))
        loss_rec = []

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            teacher_outputs = teacher_model(inputs)

            optimizer.zero_grad()
            loss = loss_fn_kd(outputs, labels, teacher_outputs, T, alpha, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                pred_i = predictions[j].cpu().numpy()
                cls_i = labels[j].cpu().numpy()
                conf_mat[cls_i, pred_i] += 1

            loss_rec.append(loss.item())
            avg_acc = conf_mat.trace() / conf_mat.sum()

            if i % print_iter == 0:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                            epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc), file=f)

        return np.mean(loss_rec), avg_acc, conf_mat

    @staticmethod
    def valid(dataloader, model, loss_f, device, num_cls):
        model.eval()

        conf_mat = np.zeros((num_cls, num_cls))
        loss_rec = []

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                loss = loss_f(outputs, labels)
                _, predictions = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cls_i = labels[j].cpu().numpy()
                pred_i = predictions[j].cpu().numpy()
                conf_mat[cls_i, pred_i] += 1

            loss_rec.append(loss.item())
            avg_acc = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_rec), avg_acc, conf_mat

    @staticmethod
    def train_mc(dataloader, model, loss_f, optimizer, device, epoch_id, max_epoch, num_cls, print_iter=10, mix_up=False, log_file=None):
        model.train()

        correct_pred = 0
        total_samples = 0
        loss_rec = []
        f1_rec = []

        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if mix_up:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels)
                outputs = model(inputs)
                loss = mixup_criterion(loss_f, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = loss_f(outputs, labels)

            loss.backward()
            optimizer.step()

            predictions = np.array(torch.sigmoid(outputs.data).cpu().numpy() >= 0.5, dtype=int)

            f1_rec.append(f1_score(labels.cpu().numpy(), predictions, average='samples'))

            for j, p in enumerate(predictions):
                if sum(p == labels[j].cpu().numpy()) == num_cls:
                    correct_pred += 1

            total_samples += len(labels)

            loss_rec.append(loss.item())
            avg_acc = correct_pred / total_samples

            if i % print_iter == 0:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%} F1:{:.4f}".format(
                    epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc, np.mean(f1_rec)))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%} F1:{:.4f}".format(
                            epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc, np.mean(f1_rec)), file=f)

        return np.mean(loss_rec), avg_acc, np.mean(f1_rec)


    @staticmethod
    def valid_mc(dataloader, model, loss_f, device, epoch_id, max_epoch, num_cls, print_iter=10, log_file=None):
        model.eval()

        correct_pred = 0
        total_samples = 0
        loss_rec = []
        f1_rec = []

        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                loss = loss_f(outputs, labels)

            predictions = np.array(torch.sigmoid(outputs.data).cpu().numpy() >= 0.5, dtype=int)

            f1_rec.append(f1_score(labels.cpu().numpy(), predictions, average='samples'))

            for j, p in enumerate(predictions):
                if sum(p == labels[j].cpu().numpy()) == num_cls:
                    correct_pred += 1

            total_samples += len(labels)

            loss_rec.append(loss.item())
            avg_acc = correct_pred / total_samples

            if i % print_iter == 0:
                print("Evaluating: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} F1: {:.4f}".format(
                    epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc, np.mean(f1_rec)))
                if log_file is not None:
                    with open(log_file, 'a') as f:
                        print("Evaluating: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} F1: {:.4f}".format(
                            epoch_id + 1, max_epoch, i + 1, len(dataloader), np.mean(loss_rec), avg_acc, np.mean(f1_rec)), file=f)

        return np.mean(loss_rec), avg_acc, np.mean(f1_rec)


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


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_F1(conf_mat):
    F1 = []
    acc = []
    for i in range(len(conf_mat[0])):
        TP = conf_mat[i, i]
        FP = conf_mat[i, :].sum() - TP
        FN = conf_mat[:, i].sum() - TP
        TN = conf_mat.sum() - TP - FP - FN

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1.append(2 * R * P / (R + P))
        acc.append((TP + TN) / (TP + TN + FP + FN))

        return F1, acc, (np.mean(F1), np.mean(acc))