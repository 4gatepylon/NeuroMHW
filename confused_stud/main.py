import numpy as np
import os
import time
import random

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

VERBOSE = False
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
    if VERBOSE:
        print(df.head()) # Print the first n=5 rows
        print("shape: {}\n".format(df.shape))
        print(df.info())

    df['gender']=df['gender'].replace(GENDER2CAT)
    df['ethnicity']=df['ethnicity'].replace(ETHNICITY2CAT)
    if VERBOSE:
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
        # for y in ['Attention', 'Raw', 'Theta', 'Alpha1', 'Gamma1']:
        #     sns.scatterplot(data=df,x='Mediation',y='Attention',hue='user-definedlabeln')
        #     fig,ax=plt.subplots(figsize=(7,7))
        # time.sleep(10)
    
    # Remove data that will not be similar to that we get from the EEG
    for label in ['Attention','Mediation','Raw','user-definedlabeln','age','ethnicity','gender']:
        del df[label]
    
    # Sort into bins by (subject id, video id tuples so that we infer in a single video and single subject)
    _total_height = df.to_numpy().shape[0]
    user_ids = sorted(map(lambda x: int(x), set(df['SubjectID'].tolist())))
    video_ids = sorted(map(lambda x: int(x), set(df['VideoID'].tolist())))
    _user_rows = {i: df.loc[df['SubjectID'] == i] for i in user_ids}
    experiments = [[_user_rows[i].loc[_user_rows[i]['VideoID'] == j] for j in video_ids] for i in user_ids]

    # Sanity test that we don't leave an IDs in between that are empty
    assert len(user_ids) == max(user_ids) + 1
    assert len(video_ids) == max(video_ids) + 1

    # Sanity check that we got ALL the entries and that they are disjoint sets
    # (they are disjoint sets since i and j are sets, we just need to check the split
    # was corrent and that the heights sum to the total)
    _total_height_est = 0
    for i in user_ids:
        for j in video_ids:
            subjects = list(set(experiments[i][j]['SubjectID'].tolist()))
            videos = list(set(experiments[i][j]['VideoID'].tolist()))
            assert len(subjects) == 1 and subjects[0] == i, "({},{}) There were {} subjects: {}".format(i, j, len(subjects), subjects)
            assert len(videos) == 1 and videos[0] == j, "({},{}) There were {} videos: {}".format(i, j, len(videos), videos)

            height = experiments[i][j].to_numpy().shape[0]
            _total_height_est += height
    assert _total_height_est == _total_height


    # Seperate data and labels into X and Y
    # Pick one random video and one random person
    # 1. the video is used for generalization for all the people
    # 2. the person is used to generalize fully
    # We'll measure accuracy based on the these two ^
    # We can do this various times to select the random person
    # X, Y will both be double arrays of elements with shape (num_samples, num_feats) and (num_samples, num_preds)
    Y = [[experiments[i][j]['predefinedlabel'].to_numpy().reshape((-1, 1)) for j in video_ids] for i in user_ids]
    for i in user_ids:
        for j in video_ids:
            del experiments[i][j]['predefinedlabel']
    X = [[experiments[i][j].to_numpy() for j in video_ids] for i in user_ids]

    # Sanity check that we have enough datapoints (we need one person and one video to generalize)
    assert len(X) > 1 and len(X[0]) > 1 and len(X[0][0].shape) == 2
    assert len(Y) > 1 and len(Y[0]) > 1 and len(Y[0][0].shape) == 2
    num_feats = X[0][0].shape[1]
    num_preds = Y[0][0].shape[1]
    MAX_HEIGHT = X[0][0].shape[0]

    # NOTE we standardize height (right now only width is standardized). Then we can overlay them into a 
    # so that we have the option to batch later. We do it like this: take
    # (num_videos, num_users, height=num_samples, width=num_feats|preds), then batch is
    # (v1:v1+V_BATCH_SIZE, u1:u1+U_BATCH_SIZE,:,:) to do the prediction.
    for i in user_ids:
        for j in video_ids:
            # Sanity check the shape
            assert len(X[i][j].shape) == 2
            assert len(Y[i][j].shape) == 2

            # Sanity check that the network will have the same number of features (types of waves)
            _nf = X[i][j].shape[1]
            _np = Y[i][j].shape[1]
            assert num_feats == _nf, "Should have {} feats but got {}".format(num_feats, _nf)
            assert num_preds == _np, "Should have {} preds but got {}".format(num_preds, _np)

            # Assert the proper shape
            assert Y[i][j].shape[1] == 1
            assert X[i][j].shape[1] == df.to_numpy().shape[1] - 1

            # Get the longest height so we can zero-pad
            assert X[i][j].shape[0] == Y[i][j].shape[0]
            MAX_HEIGHT = max(MAX_HEIGHT, X[i][j].shape[0])
    
    # Standardize the shape.
    for i in user_ids:
        for j in video_ids:
            remaining_height = MAX_HEIGHT - X[i][j].shape[0]
            X[i][j] = np.pad(X[i][j], ((0, remaining_height), (0, 0)))
            Y[i][j] = np.pad(Y[i][j], ((0, remaining_height), (0, 0)))
            assert X[i][j].shape[0] == MAX_HEIGHT
            assert Y[i][j].shape[0] == MAX_HEIGHT
    
    # Pick a random video and person and put it at the end so we can easily pop them out later.
    test_user_idx = random.randint(0, max(user_ids))
    test_video_idx = random.randint(0, max(video_ids))
    X[test_user_idx], X[-1] = X[-1], X[test_user_idx]
    Y[test_user_idx], Y[-1] = Y[-1], Y[test_user_idx]
    for i in user_ids:
        X[i][test_video_idx], X[i][-1] = X[i][-1], X[i][test_video_idx]
        Y[i][test_video_idx], Y[i][-1] = Y[i][-1], Y[i][test_video_idx]
    X, Y = np.stack([np.stack(X[i]) for i in user_ids]), np.stack([np.stack(Y[i]) for i in user_ids])
    print("X ~ {}".format(X.shape))
    print("Y ~ {}".format(Y.shape))

    # Sieve out the randomly chosen video and person
    X_train, Y_train = X[:-1, :-1, :, :], Y[:-1, :-1, :, :]

    # Out of distribution PERSON test (and video)
    X_ptest, Y_ptest = X[-1:, :, :, :], Y[-1: ,: , :, :]

    # Out of distribution VIDEO test (with IN-distribution people)
    X_vtest, Y_vtest = X[:-1, -1:, :, :], Y[:-1, -1:, :, :]
    print("X_train, Y_train ~ {}, {}".format(X_train.shape, Y_train.shape))
    print("X_ptest, Y_ptest ~ {}, {}".format(X_ptest.shape, Y_ptest.shape))
    print("X_vtest, Y_vtest ~ {}, {}".format(X_vtest.shape, Y_vtest.shape))

    # NOTE that at this point we have TIME space data and we MIGHT WANT FREQUENCY SPACE data, though we can decide later
    


if __name__ == "__main__":
    main()