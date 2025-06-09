from project_name.models.spectrogram_cnn import MultiheadEmotionCNN
from project_name.models.audio_feature_svm import AudioFeatureSVM
#Change above imports to star once this file is moved out of this dir
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class TrainAndEval():
    def __init__(self, aug_features: np.ndarray, aug_labels: tuple, n_augmentations: int) -> None:
        #Do data augmentation on all data
        
        self.aug_features = aug_features #quick hack for label problem
        self.aug_labels = aug_labels[0] - 1, aug_labels[1] - 1 #The labels go from 1-9 and from 1-2, shift down by 1
        self.n_augmentations = n_augmentations 
        self.n_samples = int(aug_features.shape[0] / n_augmentations)
        self.seed = None
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
        """Evaluate using holdout data split"""
        val_proportion = 0.2
        val_begin = int(self.n_samples * (1 - val_proportion))
        val_end = self.n_samples

        val_features, train_features = self.data_split(self.aug_features, val_begin, val_end)

        emote_val_labels, emote_train_labels = self.data_split(self.aug_labels[0], val_begin, val_end)
        intens_val_labels, intens_train_labels = self.data_split(self.aug_labels[1], val_begin, val_end)


        #End of shared code

        if(isinstance(model, MultiheadEmotionCNN)):
            val_feature = torch.tensor(val_features[0:1],dtype=torch.float32)  
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
            
            n_epoch = 4 #computer slow so few for test purposes

            train_loss, val_loss = model.cross_val_fit(test_dataloader, train_dataloader, n_epoch)

            for epoch in range(n_epoch):
                self.writer.add_scalars('Loss', {
                    'train': train_loss[epoch],
                    'val': val_loss[epoch]
                }, epoch)
            self.writer.close()

            #MC TEST

            self.mc_uncertainty(model, val_feature)

        elif(isinstance(model, AudioFeatureSVM)):
            spec_train_flat = train_features.reshape(train_features.shape[0], -1)
            spec_test_flat = val_features.reshape(val_features.shape[0], -1)

            # ____________________________________________
            #       Dimensionality Reduction (PCA)
            # ____________________________________________

            # Reduce feature dimensionality to improve SVM efficiency
            pca = PCA(n_components=200, random_state=self.seed)
            spec_train_reduced = pca.fit_transform(spec_train_flat)
            spec_test_reduced = pca.transform(spec_test_flat)

            # Train spectrogram-based emotion SVM
            print("Training Spectrogram-based Emotion SVM.")
            spec_emotion_svm = OneVsRestClassifier(AudioFeatureSVM(
                regularization_parameter=10, seed=self.seed))
            spec_emotion_svm.fit(spec_train_reduced, emote_train_labels)
            print("Spectrogram-based Emotion SVM trained.")

            # Train spectrogram-based intensity SVM
            print("Training Spectrogram-based Intensity SVM.")
            spec_intensity_svm = AudioFeatureSVM(
                regularization_parameter=10, seed=self.seed)
            spec_intensity_svm.fit(spec_train_reduced, intens_train_labels)
            print("Spectrogram-based Intensity SVM trained.")


        
    def loocv(model, features):
        pass

    def data_split(self, data: np.ndarray, val_begin:int, val_end:int) -> tuple:
        val_begin *= self.n_augmentations
        val_end *= self.n_augmentations
        val_data = data[val_begin:val_end]
        val_data = val_data[::(self.n_augmentations)]
        train_data = np.concatenate((data[:val_begin], data[val_end:]))
        return val_data, train_data  

    def mc_uncertainty(self, model, eval_data, n=100):
        #currently for a single sample
        model.train() #to enable dropout, because no grad or step this wont update weights
        outcomes = []
        with torch.no_grad():
            for _ in range(n):
                output = model(eval_data)
                prob = torch.nn.functional.softmax(output[0], dim=1)
                outcomes.append(prob)

        outcomes = torch.stack(outcomes)
        mean_class_probs = outcomes.mean(dim=0)
        class_std = outcomes.std(dim=0)
        print(mean_class_probs)
        print(class_std)