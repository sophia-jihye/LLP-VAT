import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FILEPATH = '/media/dmlab/My Passport/DATA/research-notes/Samsung_domain_adaptation/Gas_Sensor_Array_Drift_Dataset.csv'
NUM_CLASSES = 6

# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]

class GSADDataset(Dataset):
    def __init__(self, domain_index, scaler):
        self.num_classes = NUM_CLASSES
        df = pd.read_csv(FILEPATH)
        
        # Select 
        df = df[df.batch==domain_index]

        # Preprocess 
        df['label'] = df['label'].apply(lambda x: x-1)
        feature_cols = [col for col in df.columns if col not in ('label', 'batch')]
        self.input_dim = len(feature_cols)

        features = df[feature_cols].values
        # Rescale
        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)
        
        self.scaler = scaler
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(df.label.values, dtype=torch.long)
        self.n_data = len(df.label.values)

    def __getitem__(self, idx):
        feature, label = self.features[idx], torch.nn.functional.one_hot(self.labels[idx], num_classes = self.num_classes).float()
        return feature, label

    def __len__(self):
        return self.n_data
