import os
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union
from model import ClassifierModel
from data import SegmentsDataset,get_data_from_mysql
import configparser
config = configparser.ConfigParser()
config_path=os.path.join('config','config.ini')
config.read(config_path)



def inference_for_testset(data_loader: Any, model: Any) -> np.array:
    ''' Returns predictions array. '''
    model.eval()

    ids_list: List[str] = []
    preds: List[str] = []
    activation = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (input_, ids) in enumerate(data_loader):
            # output = model(input_.cuda())
            output = model(input_)
            output = activation(output)
            pred=np.argmax(output.detach().cpu().numpy())

            ids_list.extend(ids)
            preds.append(pred)

    ids = np.array(ids_list)
    print('ids', ids.shape)
    return ids,preds


def load_test_data(features_path) -> Any:
    all_ids, all_labels, all_scores,all_labels_index = get_data_from_mysql(mode='test')

    test_ids = np.array(all_ids)
    all_labels = np.array(all_labels)
    test_scores = np.array(all_scores)
    test_labels_index = np.array(all_labels_index)
    test_mask=None

    test_dataset = SegmentsDataset(test_ids, test_mask, test_labels_index, test_scores,
                                    features_path, mode='test')


    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, drop_last=True)

    return test_loader


def predict_with_model(model,test_loader) -> np.array:
    print(f'predicting on the test set')
    ids,test_predicts = inference_for_testset(test_loader, model)
    return ids,test_predicts

if __name__ == '__main__':
    model_path=os.path.join('output','best_model_fold_0.pth')
    model = ClassifierModel()
    # model.cuda()
    if os.path.exists(model_path):
        print(f'loading checkpoint: {model_path}')
        last_checkpoint = torch.load(model_path)
        model.load_state_dict(last_checkpoint['state_dict'])
        last_epoch = last_checkpoint['epoch']
        print(f'loaded the model from epoch {last_epoch}')

    data_path='dataset'
    features_path=os.path.join('dataset','test_features.npy')
    test_loader = load_test_data(features_path)
    test_ids,test_predicts = predict_with_model(model,test_loader)
