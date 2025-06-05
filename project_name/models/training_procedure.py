from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.audio_feature_svm import AudioFeatureSVM
#Change above imports to star once this file is moved out of this dir
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter


class TrainAndEval():
    def __init__(self, aug_features: np.ndarray, aug_labels: np.ndarray, n_augmentations: int) -> None:
        #Do data augmentation on all data
        
        self.aug_features = aug_features #quick hack for label problem
        self.aug_labels = aug_labels
        self.n_augmentations = n_augmentations 
        self.n_samples = int(aug_features.shape[0] / n_augmentations)
        self.compatiblity = {
            "SVM": ["holdout, loocv, kfold"],
            "CNN": ["holdout, loocv, kfold"], 
        }
        self.writer = SummaryWriter('logs/test')

        return None

            

    def train_model(self, model, cross_val: str):
        """
        Method that allows for training calls to the model. 

        Args:
        model pytorch model 
        cross_val: string, specifies type of cross validation to use
        features: np.ndarray, contains features to be trained and tested on
        labels: np.ndarray, contains labels for supervised training problems

        """

        #Check whether provided model is compatible with cross validation. 
        if(not cross_val in self.compatiblity[model.type]):
            raise ValueError(
                "Cross validation type not copatible with model type."
            )
    

        #Model train test
        CNN = MultiheadEmotionCNN()


    def k_fold(model, features, k:int = 10):
        fol_obj = KFold(n_splits=k, shuffle=True)

        pass





    
    def holdout(self, model: MultiheadEmotionCNN | AudioFeatureSVM):
        val_proportion = 0.2
        val_begin = int(self.n_samples * (1 - val_proportion))
        val_end = self.n_samples

        val_features, train_features = self.data_split(self.aug_features, val_begin, val_end)



        emote_val_labels, emote_train_labels = self.data_split(self.aug_labels[0], val_begin, val_end)
        intens_val_labels, intens_train_labels = self.data_split(self.aug_labels[1], val_begin, val_end)

        emote_val_labels -= 1
        emote_train_labels -= 1
        intens_train_labels -= 1
        intens_val_labels -= 1

        emote_val_labels = torch.from_numpy(emote_val_labels)
        intens_val_labels = torch.from_numpy(intens_val_labels)
        emote_train_labels = torch.from_numpy(emote_train_labels)
        intens_train_labels = torch.from_numpy(intens_train_labels)
        train_features = torch.tensor(train_features, dtype=torch.float32)
        val_features =  torch.tensor(val_features,dtype=torch.float32)       

        train_dataset = TensorDataset(train_features, emote_train_labels, intens_train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_dataset = TensorDataset(val_features, emote_val_labels, intens_val_labels)
        print(len(test_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
        
        n_epoch = 50

        train_loss, val_loss = model.cross_val_fit(test_dataloader, train_dataloader, n_epoch)

        for epoch in range(n_epoch):
            self.writer.add_scalars('Loss', {
                'train': train_loss[epoch],
                'val': val_loss[epoch]
            }, epoch)



        
    def loocv(model, features):
        pass

    def data_split(self, data: np.ndarray, val_begin:int, val_end:int) -> tuple:
        val_begin *= self.n_augmentations
        val_end *= self.n_augmentations
        val_data = data[val_begin:val_end]
        val_data = val_data[::(self.n_augmentations)]
        train_data = np.concatenate((data[:val_begin], data[val_end:]))
        return val_data, train_data  
