#Final Project - Soccer Boots
'''Created by Gael Mota'''

import sys
import csv
import math
import pandas as pd
import numpy as np
from operator import itemgetter
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

rand_st=1                                           #Set Random State variable for randomizing splits on runs

#############################################################################
#
# Load Data
#
#####################

##Import the dataset
boots = pd.read_csv('soccerboots.csv', delimiter=',', encoding='utf-8')

# Test Print
print(f"Number of initial samples: {len(boots)}")

#############################################################################
#
# Preprocess data
#
##########################################

## Remove unnecessary rows
boots.drop(boots.columns[[0,1,15,16,17,18,]], axis=1, inplace=True)

## Display columns with missing values and their counts
missing_values = boots.isnull().sum()
##print("\nColumns with missing values and their counts:")
##print(missing_values[missing_values > 0])

rows_with_missing_values = boots.isnull().any(axis=1).sum()
##print((rows_with_missing_values/len(boots))*100)

plt.figure(figsize=(15,10), dpi=300)
plt.hlines(y=missing_values.index, xmin=0, xmax=missing_values.values, color='black', linewidth=1)
plt.title('Missing Values in Each Feature', fontname='Consolas', fontsize=10, fontweight='bold')
plt.xlabel('Missing Values', fontname='Consolas', fontsize=8)
plt.ylabel('Features', fontname='Consolas', fontsize=8)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.scatter(missing_values.values, missing_values.index, color='black', s=5, zorder=5)
for i in range(len(missing_values)):
    plt.annotate(text=missing_values.iloc[i], xy=(missing_values.values[i], missing_values.index[i]), 
                 xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
##plt.show()
plt.close()

# Removing rows with missing boot brand
boots = boots.dropna(subset=['BootsBrand'])
missing_values = boots.isnull().sum()

rows_with_missing_values = boots.isnull().any(axis=1).sum()
print(rows_with_missing_values)

plt.figure(figsize=(15,10), dpi=300)
plt.hlines(y=missing_values.index, xmin=0, xmax=missing_values.values, color='black', linewidth=1)
plt.title('Missing Values in Each Feature', fontname='Consolas', fontsize=10, fontweight='bold')
plt.xlabel('Missing Values', fontname='Consolas', fontsize=8)
plt.ylabel('Features', fontname='Consolas', fontsize=8)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.scatter(missing_values.values, missing_values.index, color='black', s=5, zorder=5)
for i in range(len(missing_values)):
    plt.annotate(text=missing_values.iloc[i], xy=(missing_values.values[i], missing_values.index[i]), 
                 xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
##plt.show()
plt.close()

## Checking again for missing values
missing_values = boots.isnull().sum()
##print("\nColumns with missing values and their counts:")
##print(missing_values[missing_values > 0], "\n")

# Remove brands with less than 50 samples
boots_brand_counts = boots['BootsBrand'].value_counts()
brands_to_remove = []
for brand, count in boots_brand_counts.items():
    if count < 50:
        brands_to_remove.append(brand)

for brand in brands_to_remove:
    boots = boots[boots['BootsBrand'] != brand]

brand_counts = boots['BootsBrand'].value_counts()
print(brand_counts)

# Final check for missing values
missing_values = boots.isnull().sum()
##print("\nColumns with missing values and their counts:")
##print(missing_values[missing_values > 0], "\n")

                                          
#Color Palette for plots
boots_brand_counts = boots['BootsBrand'].value_counts()
unique = boots_brand_counts.index
counts = boots_brand_counts.values

colors = sns.color_palette("crest", len(boots_brand_counts))

### Target Original Plot Distribution
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=unique,y=counts, hue=unique, legend=False, palette=colors)
plt.title("Brands' Class Distribution", fontname='Consolas', fontsize=20, fontweight='bold')
plt.xlabel("Brand", fontname='Consolas', fontsize=15)
plt.ylabel('Count', fontname='Consolas', fontsize=15)
plt.xticks(rotation=45)
plt.yticks(np.arange(0, counts.max() + 250, 250))
for i in range(len(unique)):
    plt.text(i, counts[i] + 70, counts[i], ha='center', va='top')
##plt.show()
plt.close()

used_colors = {unique[i]: bar_plot.patches[i].get_facecolor() for i in range(len(unique))}
##for brand, color in used_colors.items():
##    print(f"Brand: {brand}, Color: {color}")


# Normalize the 'PlayerMarketValue' feature
##Histogram plot
boots['PlayerMarketValue'] = boots['PlayerMarketValue'].str.replace('€', '').str.replace(' million', '').astype(float)    
plt.figure(figsize=(10, 6))
sns.histplot(boots['PlayerMarketValue'], bins=30, kde=True)
plt.title('Distribution of Player Market Value In Millions')
plt.xlabel('Player Market Value (in millions)')
plt.ylabel('Frequency')
##plt.show()
plt.close()

##Histogram plot after Normalization
boots['Log_PlayerMarketValue'] = np.log1p(boots['PlayerMarketValue'])
plt.figure(figsize=(10, 6))
sns.histplot(boots['Log_PlayerMarketValue'], bins=30, kde=True)
plt.title('Distribution of Player Market Value')
plt.xlabel('Player Market Value (in millions)')
plt.ylabel('Frequency')
for i in range(len(unique)):
    plt.text(i, counts[i] + 70, counts[i], ha='center', va='top')
##plt.show()
plt.close()

boots.drop(boots.columns[[12]], axis=1, inplace=True)

print(f"\nNumber of samples after data cleaning: {len(boots)}")

print(f"\nFinal Features: {boots.columns.tolist()}")

#Write an output file with the cleaned data
##boots.to_csv('boots_data.csv', index=False, encoding='utf-8')


