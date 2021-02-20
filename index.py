# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# d a new cell, type '# %%'<br>
# To add a new markdown cell, type '# %% [markdown]'<br>
# %%<br>
# python imports

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import argparse
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
# custom imports

from modules.preprocessor import Preprocessor
from modules.cleanup import Cleanup
from modules.htmlutils import HTMLUtils
from modules.plotutils import PlotUtils
from modules.polynomial_features import PolynomialFeatures
from modules.domain_features import DomainFeatures

warnings.filterwarnings("ignore")

data_folder = './input'
output_folder = '.output'
htmlutils = HTMLUtils(os.getcwd())
tmp_text=""

train_ds_path = os.path.join(data_folder,'application_train.csv')
test_ds_path = os.path.join(data_folder,'application_test.csv')

app_train = pd.read_csv(train_ds_path)
htmlutils.write_text(f"Training data shape: {app_train.shape}")
app_test = pd.read_csv(test_ds_path)
htmlutils.write_text(f"Test data shape: {app_test.shape}")

app_train['TARGET'].astype(int).plot.hist(title='Home Default Risk Target Distribution')
htmlutils.write_image(plt)
htmlutils.close_section()

preprocessor = Preprocessor() 
missing_values = preprocessor.missing_values_table(app_train)
preprocessor.column_analysis(app_train)
app_train,app_test = preprocessor.encode_categorical_vars(app_train,app_test)
app_train,app_test = preprocessor.align_train_test(app_train,app_test)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days employment histogram')
plt.xlabel('Days Employment')
htmlutils.write_image(plt)

cleanup = Cleanup()
app_train,app_test = cleanup.remove_anomalies(app_train,app_test)
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days employment histogram')
plt.xlabel('Days Employment')
htmlutils.write_image(plt)
htmlutils.close_section()

correlations = app_train.corr()['TARGET'].sort_values()
htmlutils.write_dataframe('Most positive correlations',correlations.tail(15))
htmlutils.write_dataframe('Most negative correlations',correlations.head(15))
htmlutils.close_section()

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
htmlutils.write_text(f"Correlation between DAYS_BIRTH and TARGET: {app_train['DAYS_BIRTH'].corr(app_train['TARGET'])}")
plt.style.use('fivethirtyeight')
plt.hist(app_train['DAYS_BIRTH']/365,edgecolor='k',bins=25)
plt.title('Age Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.savefig('age.png')
htmlutils.write_image(plt)

plt.figure(figsize=(5,4))
sns.kdeplot(app_train.loc[app_train['TARGET']==0,'DAYS_BIRTH']/365,label='target = 0')
curve_0 = mpatches.Patch(color='blue',label='target 0')
sns.kdeplot(app_train.loc[app_train['TARGET']==1,'DAYS_BIRTH']/365,label='target = 1')
curve_1 = mpatches.Patch(color='red',label='target 1')
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')
plt.legend(handles=[curve_0,curve_1])
htmlutils.write_image(plt)
htmlutils.close_section()

age_data = app_train[['TARGET','DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
age_data['YEARS_BIRTH_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],bins=np.linspace(20,70,num=11))
htmlutils.write_dataframe('Age data converted to years',age_data.head(10))

age_group = age_data.groupby('YEARS_BIRTH_BINNED').mean()
htmlutils.write_dataframe('Average for age grouped by bins',age_group)

plt.bar(age_group.index.astype(str),age_group['TARGET']*100)
plt.xticks(rotation=75)
plt.xlabel('Age group (years)')
plt.ylabel('Failure to Replay (%)')
plt.title('Failure to repay by age group')
htmlutils.write_image(plt)
htmlutils.close_section()

ex_data = app_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]
ex_data_corrs = ex_data.corr()
htmlutils.write_dataframe('External sources correlation',ex_data_corrs)

plt.figure(figsize=(8,6))
sns.heatmap(ex_data_corrs,cmap = plt.cm.RdYlBu_r,vmin=-0.25,annot=True,vmax=0.6)
plt.title('External Sources Correlation Heatmap')
htmlutils.write_image(plt)

plt.figure(figsize=(10,12))
for i,src in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
    plt.subplot(3,1,i+1)
    sns.kdeplot(app_train.loc[app_train['TARGET']==0,src],label='target==0')
    sns.kdeplot(app_train.loc[app_train['TARGET']==1,src],label='target==1')
    plt.title('Distribution of %s by Target value'%src)
    plt.xlabel('%s'%src)
    plt.ylabel('Density')

htmlutils.write_image(plt)

plotutils = PlotUtils()
plot_data = ex_data.drop(columns=['DAYS_BIRTH']).copy()
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']
plot_data = plot_data.dropna().loc[:100000,:]

grid = sns.PairGrid(data = plot_data,size=3,diag_sharey=False,hue='TARGET', vars = [x for x in list(plot_data.columns) if x!='TARGET'])
grid.map_upper(plt.scatter,alpha=0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot,cmap = plt.cm.OrRd_r)
plt.suptitle('Ext Source And Age Features Pair Plot',size=32,y=1.05)
htmlutils.write_image(plt)
htmlutils.close_section()

poly_features_obj = PolynomialFeatures(app_train,app_test)
poly_features,poly_features_test,poly_target,poly_names = poly_features_obj.get_polynomial_features(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])
htmlutils.write_text(f"Polynomial features shape: {poly_features.shape}")
htmlutils.write_text(f"Polynomial features name: {poly_names[:15]}")
htmlutils.close_section()
poly_features = pd.DataFrame(poly_features,columns=poly_names)
poly_features['TARGET'] = poly_target
poly_corrs = poly_features.corr()['TARGET'].sort_values()
htmlutils.write_dataframe('Most negative correlations',poly_corrs.head(10))
htmlutils.write_dataframe('Most positive correlations',poly_corrs.tail(5))
htmlutils.close_section()
app_train_poly, app_test_poly = poly_features_obj.append_data_polynom_features()
htmlutils.write_text(f"Training data with polynomial feature shape: {app_train_poly.shape}")
htmlutils.write_text(f"Test data with polynomial features shape: {app_test_poly.shape}")
htmlutils.close_section()

domain_features_obj = DomainFeatures(app_train,app_test)
app_train_domain, app_test_domain = domain_features_obj.get_domain_features()
plt.figure(figsize=(12,20))
for i, feature in enumerate(['CREDIT_INCOME_PERCENT','ANNUITY_INCOME_PERCENT','CREDIT_TERM','DAYS_EMPLOYED_PERCENT']):
    plt.subplot(4,1,i+1)
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET']==0,feature],label='target == 0')
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET']==1,feature],label='target == 1')
    plt.title('Distribution of %s by target value'%feature)
    plt.xlabel('%s'%feature)
    plt.ylabel('Density')
plt.tight_layout(h_pad=2.5)
htmlutils.write_image(plt)
htmlutils.close_section()
