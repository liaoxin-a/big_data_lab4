import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import hvac
import mysql.connector
import configparser
config = configparser.ConfigParser()
config_path=os.path.join('config','config.ini')
config.read(config_path)


NUM_FOLDS=config['default'].getint('num_folds')
BATCH_SIZE=config['default'].getint('batch_size')


def dequantize(feat_vector: np.array, max_quantized_value=2, min_quantized_value=-2) -> np.array:
    ''' Dequantize the feature from the byte format to the float format. '''
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


# PyTorch dataset class for numpy arrays.
class SegmentsDataset(torch.utils.data.Dataset):
    def __init__(self, ids: np.array, dataset_mask: Optional[np.array], labels: Optional[np.array],
                 scores: Optional[np.array], features_path: str, mode: str) -> None:
        print(f'creating SegmentsDataset in mode {mode}')

        self.ids = ids
        self.scores = scores
        self.mode = mode
        self.labels = labels

        if self.mode != 'test':

            assert dataset_mask is not None and self.scores is not None
            self.features_indices = np.arange(dataset_mask.size)[dataset_mask]
            
            features_size = dataset_mask.size

            assert self.labels.shape[0] == self.scores.shape[0]
            assert self.features_indices.size == self.labels.shape[0]
            assert features_size >= self.scores.shape[0]

            if self.mode == 'train':
                self.labels *= (self.scores > 0.5).astype(int)
        else:
            features_size = self.ids.shape[0]

        self.features = np.load(features_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        features = self.features

        if self.mode != 'test':
            features_indices = self.features_indices
            labels = self.labels

            x = features[features_indices[index]]
        else:
            x = features[index]

        x = dequantize(x)
        x = torch.tensor(x, dtype=torch.float32)

        if self.mode == 'test':
            if self.labels[0]==None:
                return x, 0
            else:
                return x,torch.tensor(self.labels[index].item(),dtype=torch.long)
        else:
            y = labels[index].item()
            y=torch.tensor(y,dtype=torch.long)
            return x, y

    def __len__(self) -> int:
        return self.ids.shape[0]


def get_train_val_split(items: List[str], split_num: int) -> Tuple[np.array, np.array]:
    items = np.array(items)
    after_split=np.array_split(items, split_num)
    train_idx, val_idx = list(np.concatenate(after_split[:-1])),list(after_split[-1])
    return train_idx, val_idx

def connect2vaule(path:str):
    hvac_client_url=os.getenv("HVAC_CLIENT")
    hvac_token=os.getenv("HVAC_CLIENT_TOKEN")
    client = hvac.Client(url=hvac_client_url,token=hvac_token)
    print(client.is_authenticated())
    read_response = client.secrets.kv.read_secret_version(path)
    return read_response



def connect2mysql():
    user_value='root'
    hvac_response=connect2vaule('mysql')
    password_value=hvac_response['data']['data']['root']
    host_value='mysql'
    database_value= os.getenv("NAME_DATABASE")
    type_connect=False

    try:
        connection = mysql.connector.connect(
            user=user_value, password=password_value, host=host_value, port='3306', database=database_value)
        print("mysql connected")
        type_connect=True
    except:
        return type_connect,None
    return type_connect,connection



def get_data_from_mysql(mode='train'):
    type_r,connection = connect2mysql()
    assert type_r==True
    cursor = connection.cursor()
    cursor.execute("Select * FROM %s"%(mode))
    df = cursor.fetchall()
    all_ids, all_labels, all_scores,all_labels_index=zip(*df)
    cursor.close()
    connection.close()

    return all_ids, all_labels, all_scores,all_labels_index





def load_train_data(features_path) -> Any:

    all_ids, all_labels, all_scores,all_labels_index = get_data_from_mysql(mode='train')

    unique_ids = sorted(set(all_ids))
    # unique_train_ids, unique_val_ids = get_train_val_split(unique_ids, fold)
    unique_train_ids, unique_val_ids = get_train_val_split(unique_ids, NUM_FOLDS)

    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_labels_index = np.array(all_labels_index)
    print(all_ids.shape)
    print(all_labels.shape)

    train_mask = np.isin(all_ids, unique_train_ids)
    train_ids = all_ids[train_mask]
    train_labels = all_labels[train_mask]
    train_scores = all_scores[train_mask]
    train_labels_index=all_labels_index[train_mask]

    val_ids = all_ids[~train_mask]
    val_labels = all_labels[~train_mask]
    val_scores = all_scores[~train_mask]
    val_labels_index=all_labels_index[~train_mask]

    train_dataset = SegmentsDataset(train_ids, train_mask, train_labels_index, train_scores,
                                    features_path, mode='train')

    val_dataset = SegmentsDataset(val_ids, ~train_mask, val_labels_index, val_scores,
                                    features_path, mode='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False)

    return train_loader, val_loader