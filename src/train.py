
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from model import ClassifierModel
from data import load_train_data ,connect2vaule

from kafka import KafkaProducer
import configparser
config = configparser.ConfigParser()
config_path=os.path.join('config','config.ini')
config.read(config_path)
NUM_MODELS=config['default'].getint('num_models')
NUM_EPOCHS=config['default'].getint('num_epochs')
NUM_CLASSES=config['default'].getint('num_classes')
LOG_FREQ=config['default'].getint('log_freq')

LEARNING_RATE   = config['default'].getfloat('learning_rate')
WEIGHT_DECAY    = config['default'].getfloat('weight_decay')
LR_FACTOR       = config['default'].getfloat('lr_factor')
LR_PATIENCE     = config['default'].getfloat('lr_patience')
LR_MINIMUM      = config['default'].getfloat('lr_minimum')
LR_THRESHOLD    = config['default'].getfloat('lr_threshold')

class AverageMeter:
    ''' Computes and stores the average and current value. '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer: Any) -> float:
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        return lr

    assert False

def accuracy(predicts: Any, targets: Any) -> float:
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    if len(predicts.shape) == 2:
        predicts = np.argmax(predicts, axis=1)

    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis=1)

    if predicts.shape != targets.shape:
        print(predicts.shape)
        print(targets.shape)
        assert False

    return np.mean(predicts == targets)

def average_precision(actuals, predictions, k=None):
    num_positives = actuals.sum() + 1e-10

    sorted_idx = np.argsort(predictions)[::-1]
    if k is not None:
        sorted_idx = sorted_idx[:k]

    actuals = actuals[sorted_idx]
    precisions = np.cumsum(actuals) / np.arange(1, len(actuals) + 1)

    return (precisions * actuals).sum() / float(num_positives)

class MeanAveragePrecisionCalculator:
    ''' Classwise MAP@K - metric for Youtube-8M 2019 competition. '''

    def __init__(self, num_classes=NUM_CLASSES, k=10 ** 5):
        self._num_classes = num_classes
        self._k = k
        self._predictions = [[] for _ in range(num_classes)]
        self._actuals = [[] for _ in range(num_classes)]

    def accumulate(self, predictions, actuals, masks=None):
        if masks is None:
            masks = np.ones_like(actuals)

        for i in range(self._num_classes):
            mask = masks[:, i] > 0

            self._predictions[i].append(predictions[:, i][mask])
            self._actuals[i].append(actuals[:, i][mask])

    def __call__(self):
        aps = []
        positive_count = []
        total_count = []

        for i in range(self._num_classes):
            actuals = np.concatenate(self._actuals[i])
            predictions = np.concatenate(self._predictions[i])

            aps.append(average_precision(actuals, predictions, self._k))

            total_count.append(len(actuals))
            positive_count.append(actuals.sum())

        return np.mean(aps)


def train_epoch(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
                epoch: int, lr_scheduler: Any) -> float:
    print(f'epoch: {epoch}')
    print(f'learning rate: {get_lr(optimizer)}')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)

    print(f'total batches: {num_steps}')
    end = time.time()
    activation = nn.Softmax(dim=1)

    for i, (input_, target) in enumerate(train_loader):
        # input_ = input_.cuda()
        input_ = input_
        output = model(input_)

        # loss = criterion(output, target.cuda())
        loss = criterion(output, target)

        predict = torch.argmax(output.detach(), dim=-1)
        avg_score.update(accuracy(predict, target))

        losses.update(loss.data.item(), input_.size(0))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'acc {avg_score.val:.4f} ({avg_score.avg:.4f})')

    print(f' * average acc on train {avg_score.avg:.4f}')
    return avg_score.avg

def inference(data_loader: Any, model: Any) -> np.array:
    ''' Returns predictions array. '''
    model.eval()

    predicts_list = []
    activation = nn.Softmax(dim=1)

    with torch.no_grad():
        for input_, target in data_loader:
            # output = model(input_.cuda())
            output = model(input_)
            output = activation(output)
            predicts_list.append(output.detach().cpu().numpy())

    predicts = np.concatenate(predicts_list)
    # print('predicts', predicts.shape)
    return predicts

def validate(val_loader: Any, model: Any, epoch: int) -> float:
    ''' Infers predictions and calculates validation score. '''
    # print('validate()')
    val_pred = inference(val_loader, model)

    metric = MeanAveragePrecisionCalculator()

    val_true = val_loader.dataset.labels
    val_scores = val_loader.dataset.scores

    assert val_true.size == val_pred.shape[0]

    masks = np.eye(NUM_CLASSES)[val_true.astype('int64')]   # convert to one-hot encoding
    actuals = masks * np.expand_dims(val_scores, axis=-1)

    metric.accumulate(val_pred, actuals, masks)
    score = metric()

    print(f' * epoch {epoch} validation score: {score:.4f}')
    return score

# In my pipeline, there is a single inference function for both validation and test set prediction.
# But I had to copy-paste this function here to add some hacks to bypass
# Kaggle kernel memory restrictions.
def inference_for_testset(test_predicts: np.array, data_loader: Any, model: Any) -> np.array:
    ''' Returns predictions array. '''
    model.eval()

    ids_list: List[str] = []
    activation = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (input_, ids) in enumerate(data_loader):
            # output = model(input_.cuda())
            output = model(input_)
            output = activation(output)

            ids_list.extend(ids)
            pred = output.detach().cpu().numpy()
            bs = data_loader.batch_size
            test_predicts[i * bs : i * bs + pred.shape[0]] += pred

    ids = np.array(ids_list)
    print('ids', ids.shape)
    return ids

def get_model_path(output_path,fold_num: int) -> str:
    return os.path.join(output_path,f'best_model_fold_{fold_num}.pth')


def train_model(fold_num: int,features_path:str) -> float:
    print('=' * 80)

    model = ClassifierModel()
    # model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader = load_train_data(features_path)
    lr_scheduler = lr_sched.ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR,
                                      patience=LR_PATIENCE, threshold=LR_THRESHOLD,
                                      min_lr=LR_MINIMUM)

    last_epoch = -1
    print(f'training will start from epoch {last_epoch + 1}')

    best_score = 0.0
    best_epoch = 0

    last_lr = get_lr(optimizer)
    best_model_path = None

    for epoch in range(last_epoch + 1, NUM_EPOCHS):
        print('-' * 50)
        lr = get_lr(optimizer)

        # if we have just reduced LR, reload the best saved model
        if lr < last_lr - 1e-10 and best_model_path is not None:
            print(f'learning rate dropped: {lr}, reloading')
            last_checkpoint = torch.load(best_model_path)

            model.load_state_dict(last_checkpoint['state_dict'])
            optimizer.load_state_dict(last_checkpoint['optimizer'])
            print(f'checkpoint loaded: {best_model_path}')
            set_lr(optimizer, lr)
            last_lr = lr

        train_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        score = validate(val_loader, model, epoch)

        lr_scheduler.step(metrics=score)

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        if is_best:
            output_path='output'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            best_model_path = get_model_path(output_path,fold_num)


            data_to_save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(data_to_save, best_model_path)
            print(f'a snapshot was saved to {best_model_path}')

    hvac_response=connect2vaule('kafka')
    kafka_host=hvac_response['data']['data']['kafka_host']
    kafka_port=hvac_response['data']['data']['kafka_port']
    producer = KafkaProducer(bootstrap_servers=f"{kafka_host}:{kafka_port}", api_version=(0, 10, 2))
    p_value=f'{best_score:.04f}'.encode('utf-8')
    producer.send("kafka-pred", key=b'best score', value=p_value)
    producer.flush()
    producer.close()
    print(f'best score: {best_score:.04f}')
    return -best_score





if __name__ == '__main__':
    data_path='dataset'
    features_path=os.path.join('dataset','train_features.npy')
   
    for fold_idx in range(NUM_MODELS):
        train_model(fold_idx,features_path)
