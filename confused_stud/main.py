import numpy as np
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

GENDER2CAT = {'M':1,'F':0}
CAT2GENDER = {1:'M','F':0}
ETHNICITY2CAT = {'Han Chinese':0,'Bengali':1,'English':2}
CAT2ETHNICITY = {0:'Han Chinese',1:'Bengali',2:'English'}

# NOTE we may want to use something like Pytorch Lightning

# Simple train function to run a single epoch of training
def train1epoch(model, device, train_loader, optimizer):
    model.train()

    avg_loss = 0.0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        avg_loss += torch.sum(loss).item()
        num_batches += 1
    
    num_batches = max(float(num_batches), 1.0)
    avg_loss /= num_batches
    return avg_loss

# Simple testing function (can be used every epoch, at the end, or whenever)
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    percent_correct = 100. * correct / len(test_loader.dataset)
    
    return test_loss, percent_correct

class MemorylessWindowLogisticClassifier(nn.Module):
    # Do zero-padding on the window if necessary (populates from right
    # to left). Use the previous N elements to predict the next element.
    def __init__(self, N=5, num_features=10):
        self.N = N
        self.num_features = num_features

    # TODO we may want to prune for optimization here
    def forward(self, x):
        return F.sigmoid(self.lin(x))

def main():
    ### Code from https://www.kaggle.com/deepak915/are-they-confused-99-5-accuracy

    # Read the files
    df=pd.read_csv('EEG_data.csv')
    data = pd.read_csv('demographic_info.csv')

    # Merge subjects on subject id
    # I think it's trying to do something like this:
    # https://stackoverflow.com/questions/46563833/how-to-merge-combine-columns-in-pandas
    # Basically, I think it:
    # 1. Renames the demographic dataset so that we can match the subjects from the demographics to the EEG dataset
    # 2. Take the intersection of the two columns (by removing people without known Subject IDs). We give the
    #     data followed by the elements we wish to keep. By the documentation note that (left = df, right = data, how = inner,
    #     on = SubjectID). 'on' tells it what column to look at, 'how' tells it whether to take a union, intersection, cartesian
    #     product, etcetera, and left/right are the 
    data = data.rename(columns = {'subject ID': 'SubjectID',' gender':'gender',' age':'age',' ethnicity':'ethnicity'})
    df = df.merge(data, how = 'inner', on = 'SubjectID')
    print(df.head()) # Print the first n=5 rows

    print("shape: {}\n".format(df.shape))
    print(df.info())

    df['gender']=df['gender'].replace(GENDER2CAT)
    df['ethnicity']=df['ethnicity'].replace(ETHNICITY2CAT)
    print(df.head()) # Print the first n=5 rows

    print("### VIDEO ID ###")
    print(df['VideoID'])

    # Check if the dataset is skewed (i.e. there are a ton of data-points for a single individual
    # or there are only a single type of label. We'd like the labels to be 50%/50% roughly (ideally for each
    # person... or at least proportional to what that person experiences normally and sufficient in quantity).
    print("Value counts for video IDs: {}".format(df['VideoID'].value_counts()))
    print("Defined label counts: {}".format(df['predefinedlabel'].value_counts()))

    # What's this?
    for col in df.columns:
        if(df[col].isnull().sum()>0):
            print(col)
    print(df.describe())

    # Visualize the data so we can get a notion for what patterns exist...
    sns.set_style('darkgrid')
    sns.displot(data=df,x='Mediation',kde=True,aspect=16/7)
    fig,ax=plt.subplots(figsize=(7,7))

    # Is "mediation" meditation?
    for y in ['Attention', 'Raw', 'Theta', 'Alpha1', 'Gamma1']:
        sns.scatterplot(data=df,x='Mediation',y='Attention',hue='user-definedlabeln')
        fig,ax=plt.subplots(figsize=(7,7))
    time.sleep(10)

    # Some nonsense to help you select features that will best predict the label
    # y=pd.get_dummies(df['user-definedlabeln'])
    # mi_score=mutual_info_classif(df.drop('user-definedlabeln',axis=1),df['user-definedlabeln'])
    # mi_score=pd.Series(mi_score,index=df.drop('user-definedlabeln',axis=1).columns)
    # mi_score=(mi_score*100).sort_values(ascending=False)
    # print(mi_score)

    # Selects the top 14 features
    # print(mi_score.head(14).index)
    # top_fea=['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta', 'Beta1',
    #    'Alpha1', 'Mediation', 'Gamma2', 'SubjectID', 'Beta2', 'Raw', 'age']
    
    
    # Set to zero mean and unit variance (i.e. divide by variance). This assumes thin tails.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # df_sc=StandardScaler().fit_transform(df[top_fea])

    # TODO pytorch this shit
    # import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras import callbacks,layers

    # TODO train/test split
    # from sklearn.model_selection import train_test_split
    # Xtr,xte,Ytr,yte=train_test_split(df_sc,y,random_state=108,test_size=0.27)
    # xtr,xval,ytr,yval=train_test_split(Xtr,Ytr,random_state=108,test_size=0.27)

    # TODO this is their model, probably too big for what we want to run, but I could be wrong!
    # I'm willing to bet their network is overfitted
    # Model-Building step, stacking the hidden layers
    # model=keras.Sequential([
    #     layers.Dense(64,input_shape=(14,),activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.27),
    #     layers.Dense(124,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.3),
    #     layers.Dense(248,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.32),   
    #     layers.Dense(512,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.27),   
    #     layers.Dense(664,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.3),
    #     layers.Dense(512,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.32),
    #     layers.Dense(264,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.27),
    #     layers.Dense(124,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.3),
    #     layers.Dense(2,activation='sigmoid')
    # ])
    # Compiling the model with Adamax Optimizer
    # model.compile(optimizer='adamax',loss='binary_crossentropy',metrics='accuracy')

    # Creating the callback feature to stop the training in-Between, in case of no improvement
    # call=callbacks.EarlyStopping(patience=20,min_delta=0.0001,restore_best_weights=True)
    # Fitting the model to the training data
    # history=model.fit(xtr,ytr,validation_data=(xval,yval),batch_size=28,epochs=150,callbacks=[call])
    
    # Testing on the testing data
    # model.evaluate(xte,yte)
    # training=pd.DataFrame(history.history)
    # training.loc[:,['loss','val_loss']].plot()
    # training.loc[:,['accuracy','val_accuracy']].plot()


if __name__ == "__main__":
    main()