import os
import json
import numpy as np
import pickle as pkl
import time
import scipy
import scipy.io
import pandas as pd
import random

from scipy.stats import skew, pearsonr, spearmanr
from glob import glob
from scipy.interpolate import interp1d
from IPython.display import clear_output

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import hashlib

import zipfile, os
import boto3

from scipy.stats import skew, pearsonr, spearmanr

from sklearn.model_selection import LeaveOneOut, train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error, r2_score, median_absolute_error
from sklearn.inspection import permutation_importance 

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn import linear_model

import shap
import warnings



def get_min_max_percentile(name):
    try:
        my_min= int(name[-5:-3])
    except: 
        my_min= int(name[-4:-3])
    my_max=int(name[-2:]) 
    return([my_min, my_max])
    
def get_feature_type(name):
    my_type = name[:name.find('_')]
    return my_type

def get_readable_policy (features_df,idx):
    CC1=features_df.loc[idx]['CC1']
    CC2=features_df.loc[idx]['CC2']
    pct=features_df.loc[idx]['SOC shift in CC']
    y =str(CC1)+'('+str(round(pct))+'%)-'+str(CC2)+'C'
    return y

def get_feature_class(name):
    my_type = name[:name.find('_')]
    if  my_type == 'discharge':
        return ('2')
    elif my_type =='charge' or my_type =='full':
        return('1')
    
def get_feature_stream(name):
    find_underscores  = [i for i in range(len(name)) if name.startswith('_', i)]
    find_dash  = [i for i in range(len(name)) if name.startswith('-', i)]
    return (name[find_dash[0]+5:find_underscores[-2]])

def get_NS_name(name):
    find_underscores  = [i for i in range(len(name)) if name.startswith('_', i)]
    return (name[find_underscores[2]+1:])


#ML work...
def get_RF_MAPE (x_train,x_test,y_train,nonlog_y_test):
    #function that takes 1 array of selected features to train a RF on, and test on its test sets
    #it outputs the test MAPE error
    RF = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=2)
    RF.fit(x_train, y_train)
    train_predictions = RF.predict(x_train)
    predictions = RF.predict(x_test)

    #log transformation
    nonlog_train_predictions = 10**train_predictions
    nonlog_predictions = 10**predictions
    
    MAPE = round(mean_absolute_percentage_error(nonlog_predictions, nonlog_y_test),3)
    return (MAPE)


def get_model_MAPE (X_train,X_test,y_train,y_test, model_choice = 'RF'):
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=2)
    if model_choice == 'LinReg':
        print('yo')
        model = linear_model.LinearRegression()
    elif model_choice =='Lasso':
        model = linear_model.Lasso()
    elif model_choice =='ElasticNet':
        model = ElasticNet(random_state=25)
    elif model_choice =='KRR':
        model = KernelRidge()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)

    
    train_MAPE = round(mean_absolute_percentage_error(y_train,train_predictions),3)
    MAPE = round(mean_absolute_percentage_error(y_test,predictions),3)
    return (train_MAPE,MAPE)

    
def make_cv_model_object(df,
                                 X_col,Y_col=['y'],
                                 cv_splits = 5,
#                                  split_lists=split_lists,
                                 model=RandomForestRegressor(random_state=0),
                                 model_hyperparams={'n_estimators': [20,40,80,160],
                                                    'min_samples_leaf':[1,2],#[1,2,4,8]
                                                    'min_samples_split':[2,4,8]}
                                ):
    Y = df[Y_col]
    X = df[X_col]

    # Normalize data
    Y_scaler = StandardScaler()
    Y_scaled = pd.DataFrame(Y_scaler.fit_transform(Y),columns=Y_col)

    X_scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(X_scaler.fit_transform(X),columns=X_col)

    # 5-fold cross-validation
    fit_model = GridSearchCV(model, model_hyperparams, cv = cv_splits, n_jobs=-1)
    fit_model.fit(X_scaled,Y_scaled.values.ravel())
#     model.best_params_
    return fit_model, X_scaled, Y_scaled, X_scaler, Y_scaler

def make_cv_model_object2(df,
                                 X_col,Y_col=['y'],
                                 cv_splits = 5,
#                                  split_lists=split_lists,
                                 model=RandomForestRegressor(random_state=0),
                                 model_hyperparams={'n_estimators': [20,40,80,160],
                                                    'min_samples_leaf':[1,2],#[1,2,4,8]
                                                    'min_samples_split':[2,4,8]}
                                ):
    Y = df[Y_col]
    X = df[X_col]

    X_scaler = StandardScaler()
    X_scaled = pd.DataFrame(X_scaler.fit_transform(X),columns=X_col)

    # 5-fold cross-validation
    fit_model = GridSearchCV(model, model_hyperparams, cv = cv_splits)
    fit_model.fit(X_scaled,Y.values.ravel())
#     model.best_params_
    return fit_model, X_scaled, X_scaler

def get_train_test_MAPE (fit_model,
                                      X_train, X_test, X_scaler, Y_train, Y_test):
    X_train_scaled = X_scaler.transform(X_train)
    Y_train_pred = fit_model.predict(X_train_scaled)
    train_MAPE = mean_absolute_percentage_error(Y_train.values.ravel(),Y_train_pred) # units [%]
    
    ## Scaling test X
    X_test_scaled = X_scaler.transform(X_test)
    ## Feedforward X test into cv-trained model
    Y_test_pred = fit_model.predict(X_test_scaled)
    test_MAPE = mean_absolute_percentage_error(Y_test.values.ravel(),Y_test_pred) # units [%]
    
    return(train_MAPE, test_MAPE)


def plot_train_test_model_predictions(fit_model,
                                      X_train_scaled, X_test, X_scaler, Y_train, Y_test, Y_scaler,
                                        X_col,Y_col=['y'],
                                     plot_bounds = [0,2000],plot=True):
    
    plot_bounds = [0,2000]
    metric_string = 'cycles'

    ### Plotting and error calculating for TRAIN

    ## Feedforward X train into cv-trained model
    cv_train_Y_pred_scaled = fit_model.predict(X_train_scaled)
    Y_train_pred = Y_scaler.inverse_transform(np.reshape(cv_train_Y_pred_scaled,(-1,1)))[:,0]
    # Reshape Y for plotting
    Y_train_reshaped = np.reshape(Y_train.values,(-1,1))[:,0]

    RMSE = mean_squared_error(Y_train_reshaped,Y_train_pred,squared=False) # units EFC
    MAE  = mean_absolute_error(Y_train_reshaped,Y_train_pred) # units EFC
    MAPE = mean_absolute_percentage_error(Y_train_reshaped,Y_train_pred) # units [%]

    ## Scaling test X
    X_test_scaled = pd.DataFrame(X_scaler.transform(X_test),columns=X_col)
    # Y_test_scaled = pd.DataFrame(Y_scaler.transform(Y_test),columns=Y_col)

    ## Feedforward X test into cv-trained model
    Y_test_pred_scaled = fit_model.predict(X_test_scaled)
    Y_test_pred = Y_scaler.inverse_transform(np.reshape(Y_test_pred_scaled,(-1,1)))[:,0]

    RMSE_test = mean_squared_error(Y_test,Y_test_pred,squared=False) # units EFC
    MAE_test  = mean_absolute_error(Y_test,Y_test_pred) # units EFC
    MAPE_test = mean_absolute_percentage_error(Y_test,Y_test_pred) # units [%]
    
    ## Plotting train parity plot
    if plot ==True:
        fig, axs = plt.subplots(1,1,figsize=(5,4))
        axs.plot([plot_bounds[0],plot_bounds[1]],
                     [plot_bounds[0],plot_bounds[1]],
                     linestyle='--',color='grey') # parity line
        axs.scatter(Y_train_reshaped,Y_train_pred,s=36) # train points
        axs.set_xlim([plot_bounds[0],plot_bounds[1]])
        axs.set_ylim([plot_bounds[0],plot_bounds[1]])

        axs.set_xlabel(f'Observed {metric_string}',fontsize=14,fontweight='bold')
        #     axs.xticks(fontsize=14)
        axs.set_ylabel(f'Predicted {metric_string}',fontsize=14,fontweight='bold',labelpad=0)
        #     axs.yticks(fontsize=14)

        axs.annotate('       Train',(0.05,0.9),fontsize=14,color='tab:blue',
                     xycoords='axes fraction')
        axs.annotate('MAPE = '+format(MAPE*100,'.3g')+'%',(0.05,0.84),fontsize=14,color='tab:blue',
                     xycoords='axes fraction')
        axs.annotate('RMSE = '+format(RMSE,'.3g'),(0.05,0.78),fontsize=14,color='tab:blue',
                     xycoords='axes fraction')
        axs.annotate(' MAE  = '+format(MAE,'.3g'),(0.05,0.72),fontsize=14,color='tab:blue',
                     xycoords='axes fraction')

        ## Plotting test parity plot
        axs.scatter(Y_test,Y_test_pred,s=36)

        axs.annotate('       Test',(0.50,(0.9-0.6)),fontsize=14,color='tab:orange',
                     xycoords='axes fraction')
        axs.annotate('MAPE = '+format(MAPE_test*100,'.3g')+'%',(0.50,(0.84-0.6)),fontsize=14,color='tab:orange',
                     xycoords='axes fraction')
        axs.annotate('RMSE = '+format(RMSE_test,'.3g'),(0.50,(0.78-0.6)),fontsize=14,color='tab:orange',
                     xycoords='axes fraction')
        axs.annotate(' MAE  = '+format(MAE_test,'.3g'),(0.50,(0.72-0.6)),fontsize=14,color='tab:orange',
                     xycoords='axes fraction')

    return MAPE,MAPE_test,RMSE,RMSE_test,MAE, MAE_test

    
    
def plot_n_MAPE (y_train, train_predictions,y_test, predictions,plot=True):
    train_MAPE = np.around(mean_absolute_percentage_error(y_train, train_predictions),3)
    test_MAPE = np.around(mean_absolute_percentage_error(y_test,predictions),3)
    if plot==True:
        fig = plt.subplots(dpi = 80)
        plt.scatter(y_train, train_predictions, c='blue',label = 'TRAIN || MAPE:' + 
                        "{:.1f}%".format(train_MAPE*100) )
        plt.scatter(y_test, predictions, c='pink',label = 'TEST || MAPE:' + 
                        "{:.1f}%".format(test_MAPE*100))
        mymin = min(min(y_test), min(y_train))
        mymax = max(max(y_train),max(y_test))
        plt.plot([mymin,mymax],[mymin,mymax],color="gold")
        plt.ylabel('Predicted Values')
        plt.xlabel('Real Values')
        plt.legend(loc = 'best')
    return(train_MAPE, test_MAPE)

def plot_n_MAPE_TTS (y_train, train_predictions,y_test, predictions,plot=True):
    # the only difference with the function above is that it's returning all three errors

    train_MAPE = np.around(mean_absolute_percentage_error(y_train, train_predictions),3)
    test_MAPE = np.around(mean_absolute_percentage_error(y_test,predictions),3)
    train_RMSE =  np.around(mean_squared_error(y_train,train_predictions,squared=False),3) # units EFC
    test_RMSE =  np.around(mean_squared_error(y_test,predictions,squared=False),3)
    train_MAE = np.round(mean_absolute_error(y_train,train_predictions),3)
    test_MAE = np.round(mean_absolute_error(y_test,predictions),3)
    
    if plot==True:
        fig = plt.subplots(dpi = 80)
        plt.scatter(y_train, train_predictions, c='blue',label = 'TRAIN || MAPE:' + 
                        "{:.1f}%".format(train_MAPE*100) )
        plt.scatter(y_test, predictions, c='pink',label = 'TEST || MAPE:' + 
                        "{:.1f}%".format(test_MAPE*100))
        mymin = min(min(y_test), min(y_train))
        mymax = max(max(y_train),max(y_test))
        plt.plot([mymin,mymax],[mymin,mymax],color="gold")
        plt.ylabel('Predicted Values')
        plt.xlabel('Real Values')
        plt.legend(loc = 'best')
    return(train_MAPE, test_MAPE,train_RMSE,test_RMSE,train_MAE,test_MAE)


def ruleout_split (X,Y,ruleout_list):
    train_indices = np.array([df_index for df_index in X.index if df_index not in ruleout_list])
    X_train = X.loc[train_indices]
    X_test = X.loc[ruleout_list]
    Y_train = Y.loc[train_indices]
    Y_test = Y.loc[ruleout_list]
    return(X_train,X_test, Y_train,Y_test)