#Final Project - Soccer Boots
'''Created by Gael Mota'''

import sys
import csv
import math
import pandas as pd
import numpy as np
import joblib
import time
import urllib
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import seaborn as sns
from operator import itemgetter
from collections import Counter
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import make_scorer, roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipelinex
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import f1_score
import time

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#############################################################################
#
# Global parameters
#
#####################

cross_val=1                                         #Control Switch for CV (0=Test/Train, 1=Cross Validation)                                                                                                                                                     
fold=5                                              #Cross validation fold settings    
feat_select=1                                       #Control Switch for Feature Selection (0=off,1=on)                                                                                   
fs_type=1                                           #Feature Selection type (1=Wrapper Select, 2=Univariate Selection)                        

feat_start=1                                        #Start column of features
k_cnt=5                                             #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
rand_st=1                                           #Set Random State variable for randomizing splits on runs

#############################################################################
#
# Load Data
#
#####################

## Read the CSV file into a pandas DataFrame
df = pd.read_csv('boots_data.csv', encoding='utf-8')

#############################################################################
#
# Preprocess data
#
##########################################

## Encode the data
#Define columns to be encoded and treated as categories
float_columns = ['Log_PlayerMarketValue']

encode_columns = [col for col in df.columns if col not in float_columns]

#Initialize label encoders
label_encoders = {col: LabelEncoder() for col in encode_columns}

#Fit label encoders and transform the data
for col in encode_columns:
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

#Handle float columns, ensuring missing values are properly handled
for col in float_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

## Data and target split
target_column = 'BootsBrand'
target = df[target_column]
data = df.drop(columns=[target_column])

data_np=np.asarray(data)
target_np=np.asarray(target)

## Mappings for encoded labels
#To original brand names
boots_brand_encoder = label_encoders[target_column]
encoded_to_original = {index: brand for index, brand in enumerate(boots_brand_encoder.classes_)}

#To original boots names
top_boots = 'BootsName'
boots_name_encoder = label_encoders[top_boots]
encoded_to_original_name = {index: name for index, name in enumerate(boots_name_encoder.classes_)}

## Color Palette for plots
unique_values, counts = np.unique(target_np, return_counts=True)
custom_colors = {
    0: (0.41511616124999995, 0.63238414875, 0.55518600375, 1),
    2: (0.24500366000000007, 0.4849521874999999, 0.5093681, 1),
    1: (0.16781848249999992, 0.3548046799999998, 0.4759132175000001, 1)
}
colors = [custom_colors[brand] for brand in unique_values]

corrected_colors = {
    0: custom_colors[1],
    2: custom_colors[0],
    1: custom_colors[2]}
colors2 = [corrected_colors[brand] for brand in unique_values]

########## Classifiers ##########

#List of models for feature selection
classifiers_feat_select = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=rand_st),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=3, criterion='entropy', random_state=rand_st),
    'Gradient Boosting': GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=rand_st),
    'Ada Boosting': AdaBoostClassifier(estimator=None, n_estimators=100, learning_rate=0.1, random_state=rand_st, algorithm='SAMME')}


# List of models for training 
classifiers = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=15, min_samples_split=10, min_samples_leaf=5, max_features=None, random_state=rand_st),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, criterion='entropy', random_state=rand_st),
    'Gradient Boosting': GradientBoostingClassifier(loss='log_loss', learning_rate=0.01, n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=rand_st),
    'Ada Boosting': AdaBoostClassifier(estimator=None, n_estimators=50, learning_rate=0.01, random_state=rand_st, algorithm='SAMME'),
    'Neural Network': MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,), random_state=rand_st)}

results = []
feature_importances = {}

#############################################################################
#
# Test/Train Split
#
##########################################
if cross_val == 0:
    print('Test/Train Split\n')

    ## Split the data
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=rand_st)

    ## Impute missing values
    imputer = IterativeImputer(random_state=rand_st)
    data_train_imputed = imputer.fit_transform(data_train)
    data_test_imputed = imputer.transform(data_test)

    ## Handle Class Imbalance
    smote = SMOTE(sampling_strategy = {1 : 1066}, random_state=rand_st)
    data_train_resampled, target_train_resampled = smote.fit_resample(data_train_imputed, target_train)

    ########## Feature Selection ##########   
    for name, clf in classifiers_feat_select.items():
        clf.fit(data_train_resampled, target_train_resampled)
        if hasattr(clf, 'feature_importances_'):
            feature_importances[name] = clf.feature_importances_

    if feat_select==1:
        print('--Feature Selection On--')

        wrapper_printed = False
        univariate_printed = False

        for clf_name, clf in classifiers_feat_select.items():
            
            #Wrapper Selection
            if fs_type == 1 and not wrapper_printed:
                print (f'Wrapper-Based Feature Selection')
                wrapper_printed = True
                sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)
                fit_mod=sel.fit(data_train_resampled, target_train_resampled)    
                sel_idx=fit_mod.get_support()

            #Univariate Feature Selection - Chi-squared
            elif fs_type == 2 and not univariate_printed:
                print ('Univariate Feature Selection (Chi2)')
                univariate_printed = True
                sel=SelectKBest(chi2, k=k_cnt)
                fit_mod=sel.fit(data_train_resampled, target_train_resampled)                                         
                sel_idx=fit_mod.get_support()

            ##2) Get lists of selected and non-selected features (names and indexes) #######
            header = df.drop(columns=['BootsBrand']).columns.to_list()
            temp=[]
            temp_idx=[]
            temp_del=[]
            for i in range(len(data_train_resampled[0])):
                if sel_idx[i]==1:                                                           #Selected Features get added to temp header
                    temp.append(header[i])
                    temp_idx.append(i)
                else:                                                                       #Indexes of non-selected features get added to delete array
                    temp_del.append(i)

            print(f'{clf_name} selected features: {temp}')
            print(f'Selected {len(temp)} out of {len(data_train_resampled[0])}\n')

        ##3) Filter selected columns from original dataset
        data_train_resampled = data_train_resampled[:, temp_idx]
        data_test_imputed = data_test_imputed[:, temp_idx]

        feature_importances_selected = {}
        for name, clf in classifiers_feat_select.items():
            clf.fit(data_train_resampled, target_train_resampled)
            if hasattr(clf, 'feature_importances_'):
                feature_importances_selected[name] = clf.feature_importances_

        ## Plotting feature importance
        feature_names = df.drop(columns=['BootsBrand']).columns
        avg_importances = np.zeros(len(feature_names))

        for name, importances in feature_importances_selected.items():
            for idx, feature in enumerate(temp_idx):
                avg_importances[feature] += importances[idx]

        avg_importances /= len(feature_importances_selected)

        importance_list = [(feature, importance) for feature, importance in zip(feature_names, avg_importances)]
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': avg_importances})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance Across Models', fontname='Consolas', fontsize=20, fontweight='bold')
        plt.xlabel('Importance', fontname='Consolas', fontsize=15)
        plt.ylabel('Feature', fontname='Consolas', fontsize=15)
        plt.tight_layout()
##        plt.show()
        plt.close()

    ########## Training the models ##########
    print('--ML Model Output--')
    for name,clf in classifiers.items():
        start_ts = time.time()
        clf.fit(data_train_resampled, target_train_resampled)

        # Training performance
        train_acc = clf.score(data_train_resampled, target_train_resampled)
        train_f1 = metrics.f1_score(target_train_resampled, clf.predict(data_train_resampled), average='weighted')

        # Test performance
        test_acc = clf.score(data_test_imputed, target_test)
        test_f1 = metrics.f1_score(target_test, clf.predict(data_test_imputed), average='weighted')
        run_time = time.time() - start_ts

        results.append({
            'Classifier': name,
            'Train Accuracy': str(train_acc)[:6],
            'Train F1': str(train_f1)[:6],
            'Test Accuracy': str(test_acc)[:6],
            'Test F1': str(test_f1)[:6],
            'Run Time': str(run_time)[:6]
        })
    
    results_np = pd.DataFrame(results)
    print(results_np)

#############################################################################
#
# Cross Validation
#
##########################################

####Cross-Val Classifiers####
if cross_val == 1: 
    f1_scorer = make_scorer(f1_score, average='weighted')
    scorers = {'Accuracy': 'accuracy', 'f1': f1_scorer}
    selected_features_dict = {name: set() for name in classifiers.keys()}
    feature_importances_dict = {name: np.zeros(data.shape[1]) for name in classifiers.keys()}
    fold_selected_features = []
    fold_importances = []

    print(f'{fold}-fold Cross  Validation\n')
    skf = StratifiedKFold(n_splits=fold, random_state=rand_st, shuffle=True)

    legend_printed = False
    wrapper_printed = False
    univariate_printed = False
    
    for name, clf in classifiers.items():
        start_ts = time.time()
        for fold_idx, (train_index, test_index) in enumerate(skf.split(data, target)):
            data_train, data_test = data.iloc[train_index], data.iloc[test_index]
            target_train, target_test = target[train_index], target[test_index]
            
            pipeline_steps = [('imputer', IterativeImputer(random_state=rand_st)),
                              ('smoteenn', SMOTEENN(smote=SMOTE(random_state=rand_st), enn=EditedNearestNeighbours(), random_state=rand_st))]

            if feat_select == 1 and name in classifiers_feat_select:
                if not legend_printed:
                    print('--Feature Selection On--')
                    legend_printed = True
                if fs_type == 1 and not wrapper_printed:
                    print ('Wrapper-Based Feature Selection')
                    wrapper_printed = True
                    selector = SelectFromModel(classifiers_feat_select[name], threshold='mean', max_features=None)
                elif fs_type == 2 and not univariate_printed:
                    print ('Univariate Feature Selection (Chi2)')
                    univariate_printed = True
                    selector = SelectKBest(chi2, k=k_cnt)
                selector.fit(data_train, target_train)
                selected_features = selector.get_support()

                fold_selected_features.append(selected_features)
                selected_features_dict[name].update(data.columns[selected_features])
            
                data_train = data_train.loc[:, selected_features]
                data_test = data_test.loc[:, selected_features]

            pipeline_steps.append(('classifier', clf))
            pipeline = ImbPipeline(steps=pipeline_steps)

            pipeline.fit(data_train, target_train)
            y_pred = pipeline.predict(data_test)
     
            if feat_select ==1 and name in classifiers_feat_select and (hasattr(clf, 'feature_importances_')) or (hasattr(clf, 'coef_')):
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                elif hasattr(clf, 'coef_'):
                    importances = (clf.coef_[0])
                    
                full_importances = np.zeros(data.shape[1])
                full_importances[selected_features] = importances
                fold_importances.append(full_importances)

        if feat_select == 1 and fold_importances and name in classifiers_feat_select:
            avg_importances = np.mean(fold_importances, axis=0)
            feature_importances_dict[name] = avg_importances

        scores = cross_validate(pipeline, data_train, target_train, scoring=scorers, cv=skf, return_train_score=True)
        
        scores_Acc = scores['test_Accuracy']            
        scores_f1 = scores['test_f1']
        scores_Acc_train = scores['train_Accuracy']            
        scores_f1_train = scores['train_f1']   
        run_time = time.time() - start_ts

        results.append({'Classifier': name,
                        'Accuracy  Train': f"{scores_Acc_train.mean():.2f} (+/- {scores_Acc_train.std() * 2:.2f})",
                        'F1 Train': f"{scores_f1_train.mean():.2f} (+/- {scores_f1_train.std() * 2:.2f})",
                        'Accuracy  Test': f"{scores_Acc.mean():.2f} (+/- {scores_Acc.std() * 2:.2f})",
                        'F1 Test': f"{scores_f1.mean():.2f} (+/- {scores_f1.std() * 2:.2f})",
                        'Run Time': str(run_time)[:6]})

        if feat_select == 1 and name in classifiers_feat_select:
            print(f"{name} selected features: {sorted(selected_features_dict[name])}")

    if feat_select == 1:
        ## Plotting feature importance
        feature_names = df.drop(columns=['BootsBrand']).columns
        avg_importances = np.zeros(len(feature_names))

        for importances in feature_importances_dict.values():
            avg_importances += importances

        avg_importances /= len(feature_importances_dict)

        importance_df = pd.DataFrame({'feature': feature_names, 'importance': avg_importances})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance Across Variables', fontname='Consolas', fontsize=20, fontweight='bold')
        plt.xlabel('Importance', fontname='Consolas', fontsize=15)
        plt.ylabel('Feature', fontname='Consolas', fontsize=15)
        plt.tight_layout()
##        plt.show()
        plt.close()

    results_np = pd.DataFrame(results)
    print(results_np)


