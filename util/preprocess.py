import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np


class BCI_IV_2a(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self, features, labels, domain=None, domain_label=False):
        super(BCI_IV_2a, self).__init__()
        self.domain_label = domain_label
        self.features = features
        self.labels = labels
        self.domain = domain
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.domain_label:
            return self.features[index], self.labels[index], self.domain[index]
        else:
            return self.features[index], self.labels[index]

def get_test_EEG_data(sub,data_path):
    '''
    Return one subject's test dataset.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
    @author:WenChao Liu 
    '''
    test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))
    test_data = sio.loadmat(test_path)
    test_x = test_data['x_data']
    test_y = test_data['y_data']
    test_x,test_y = torch.FloatTensor(test_x),torch.LongTensor(test_y).reshape(-1)
    test_dataset = BCI_IV_2a(test_x,test_y)
    return test_dataset


# need!
def get_HO_EEG_data(subject_id, data_path, validation_size=0.2, data_seed=20210902):
    
    '''
    Return one subject's training dataset,split training dataset and split validation dataset.
    Arg:
        subject_id:Subject number.
        data_path:The data path of all subjects.
        validation_size:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:WenChao Liu
    '''
    train_path = os.path.join(data_path, f'sub{subject_id}_train/Data.mat')
   
    train_data = sio.loadmat(train_path)
    train_features = train_data['x_data']
    train_labels = train_data['y_data'].reshape(-1)
    print(train_features.shape, train_labels.shape)
        
    split_train_features, split_train_labels, split_validation_features, split_validation_labels = train_validation_split(
        train_features, train_labels, validation_size, seed=data_seed
    )
    
    train_features, train_labels = torch.FloatTensor(train_features), torch.LongTensor(train_labels).reshape(-1)
    split_train_features, split_train_labels = torch.FloatTensor(split_train_features), torch.LongTensor(split_train_labels).reshape(-1)
    split_validation_features, split_validation_labels = torch.FloatTensor(split_validation_features), torch.LongTensor(split_validation_labels).reshape(-1)
   
    train_dataset = BCI_IV_2a(train_features, train_labels)
    split_train_dataset = BCI_IV_2a(split_train_features, split_train_labels)
    split_validation_dataset = BCI_IV_2a(split_validation_features, split_validation_labels)    
    test_dataset = get_test_EEG_data(subject_id, data_path)
    
    return train_dataset, split_train_dataset, split_validation_dataset, test_dataset

def train_validation_split(features, labels, validation_size, seed=None):
    '''
    Split the training set into a new training set and a validation set
    @author: WenChao Liu
    '''
    if seed:
        np.random.seed(seed)
    unique_labels = np.unique(labels)
    validation_features = []
    validation_labels = []
    train_features = []
    train_labels = []
    for label in unique_labels:
        index = (labels == label)
        label_count = np.sum(index)
        print(f"class-{label}: {label_count}")
        class_features = features[index]
        class_labels = labels[index]
        random_order = np.random.permutation(label_count)
        class_features, class_labels = class_features[random_order], class_labels[random_order]
        print(class_features.shape)
        validation_features.extend(class_features[:int(label_count*validation_size)].tolist())
        validation_labels.extend(class_labels[:int(label_count*validation_size)].tolist())
        train_features.extend(class_features[int(label_count*validation_size):].tolist())
        train_labels.extend(class_labels[int(label_count*validation_size):].tolist())
    
    validation_features = np.array(validation_features)
    validation_labels = np.array(validation_labels).reshape(-1)
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels).reshape(-1)
    
    print(train_features.shape, train_labels.shape)
    print(validation_features.shape, validation_labels.shape)
    return train_features, train_labels, validation_features, validation_labels