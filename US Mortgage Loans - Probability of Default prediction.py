# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:44:00 2023

@author: Lenovo
"""

import warnings; warnings.simplefilter('ignore')
import os
os.chdir('C:\\Users\\Lenovo\\Desktop\\ML1and2')
import pickle
import joblib
import numpy as np
import random
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import functools
import miceforest as mf
from scipy.stats import chi2_contingency, pointbiserialr, spearmanr, binom_test, beta
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


def woe(data_in, target, variable, bins, binning):
    
    df = data_in
    df2 = data_in[[target, variable]].rename(columns={target: 'Target', variable: 'Variable'}).dropna()
    
    if binning == 'True':
       df2['key'] = pd.qcut(df2.Variable, bins, labels=False, duplicates='drop')
    if binning == 'False':
       df2['key'] = df2.Variable
    table = pd.crosstab(df2.key, df2.Target, margins= True)
    table = table.drop(['All'], axis=0)
    table = table.rename(columns={1: 'deft', 0: 'nondeft'}).reset_index(drop=False)

    table.loc[:, 'fracdeft'] = table.deft/np.sum(table.deft)
    table.loc[:, 'fracnondeft'] = table.nondeft/np.sum(table.nondeft)

    table.loc[:, 'WOE'] = np.log(table.fracdeft/table.fracnondeft)
    table.loc[:, 'IV'] = (table.fracdeft-table.fracnondeft)*table.WOE
    
    table.rename(columns={'WOE': variable}, inplace=True)
    table=table.add_suffix('_WOE')
    table.rename(columns={table.columns[0]: 'key' }, inplace = True)
    WOE = table.iloc[:, [0,-2]]
    
    df = pd.merge(df, df2.key, right_index=True, left_index=True)
      
    outputWOE = pd.merge(df, WOE, on='key').drop(['key'], axis=1)
    outputIV = pd.DataFrame(data={'name': [variable], 'IV': table.IV_WOE.sum()})
    
    return outputWOE, outputIV

def test_result(fit, outcome, time):
    
    fitP=pd.DataFrame(data=fit)
    outcomeP=pd.DataFrame(data=outcome)
    timeP=pd.DataFrame(data=time)
    
    if isinstance(fit, pd.Series):
        fit=fit.values
    if isinstance(outcome, pd.Series):
        outcome=outcome.values
    if isinstance(time, pd.Series):
        time=time.values
    
    df = pd.concat([fitP, outcomeP, timeP], axis=1)
    df.columns = ['fit', 'outcome', 'time']
    return df

def calculate_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    precision = precision_score(y_test, y_pred, average= 'weighted')
    
    return acc, auc, f1, recall, precision

def validation(fit, outcome , time, model_name, continuous=False):

    plt.rcParams['figure.dpi']= 300
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams.update({'font.size': 16})
    
    fitP=pd.DataFrame(data=fit)
    outcomeP=pd.DataFrame(data=outcome)
    timeP=pd.DataFrame(data=time)
    
    if isinstance(fit, pd.Series):
        fit=fit.values
    if isinstance(outcome, pd.Series):
        outcome=outcome.values
    if isinstance(time, pd.Series):
        time=time.values
    
    data_in = pd.concat([fitP, outcomeP, timeP], axis=1)
    data_in.columns = ['fit', 'outcome', 'time']
    #means = data_in.groupby('time')[['fit', 'outcome']].mean().reset_index(drop=False)
    means = data_in.groupby('time').aggregate({'fit':'mean','outcome':'mean'}).reset_index()
  
    data_in['outcomeD']=data_in.loc[:,'outcome']    
    if continuous==True:
        data_in.loc[data_in['outcome'] >= data_in.outcome.mean(), 'outcomeD'] = 1
        data_in.loc[data_in['outcome'] <  data_in.outcome.mean(), 'outcomeD'] = 0
    
    outcomeD=data_in.loc[:,'outcomeD'].values

    roc_auc = np.nan
    binom_p = np.nan
    Jeffreys_p =  np.nan
    
    max_outcome_fit=np.maximum(max(outcome), max(fit))
    min_outcome_fit=np.minimum(min(outcome), min(fit)) 
    if min_outcome_fit>=0 and max_outcome_fit<=1:
        #roc_auc = roc_auc_score(outcomeD, fit).round(4)
        outcomeP[0] = outcomeP[0].astype(np.int64)
        roc_auc = roc_auc_score(outcomeP, fit).round(4)    
        binom_p = binom_test(sum(outcomeD), n=len(outcomeD), p= np.mean(fit), alternative='greater').round(decimals=4)
        Jeffreys_p =  beta.cdf(np.mean(fit), sum(outcomeD)+0.5, len(outcomeD)-sum(outcomeD)+0.5).round(decimals=4)

    
    the_table = [['Counts', len(outcome)],
                      ['Mean outcome', round(sum(outcome)/len(outcome),4)],
                      ['Mean fit', np.mean(fit).round(4)],
                      ['AUC', roc_auc],
                      ['MAE', mean_absolute_error(means['outcome'], means['fit']).round(decimals=4)],
                      ['MSE', mean_squared_error(means['outcome'], means['fit']).round(decimals=4)],
                      ['RMSE/ SQR(Brier score)', round(np.sqrt(((outcome-fit).dot(outcome-fit))/len(outcome)),4)],
                      ['Binomial p-value', binom_p],
                      ['Jeffreys p-value', Jeffreys_p]]
    the_table=pd.DataFrame(data=the_table)
    the_table.columns = ['Metric', 'Value']
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
 
    plt.subplot(221)
    plt.title('Summary_' + model_name)
    plt.axis('off')
    plt.axis('tight')
    test=plt.table(cellText=the_table.values, colLabels=the_table.columns, loc='center', cellLoc='center', colWidths=[0.34, 0.2])
    test.auto_set_font_size(False)
    test.set_fontsize(16) 
    test.scale(2, 1.5)
    
    plt.subplot(222)
    plt.title('Time-Series Real-Fit')
    plt.plot(means['time'],means['outcome'])
    plt.plot(means['time'],means['fit'], color='red', ls='dashed')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Mean', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(('Outcome','Fit'), loc='best', fontsize=15)
    
    plt.subplot(223)
    plt.title('Fit Histogram')
    plt.hist(fit, bins=20, histtype='bar', density=True)
    plt.xlabel('Fit', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    
    data_in['cat'] = pd.qcut(data_in.fit, 10, labels=False, duplicates='drop')
    real_fit = data_in.groupby('cat').aggregate({'fit':'mean','outcome':'mean'})
    #real_fit = data_in.groupby('cat')[['fit', 'outcome']].mean()
    mpv=real_fit.fit.values
    fop=real_fit.outcome.values
    
    maximum=np.maximum(max(fop), max(mpv))       
    maximum=np.ceil(maximum*100)/100
    minimum=np.minimum(min(fop), min(mpv))
    minimum=np.floor(minimum*100)/100
    
    plt.subplot(224)
    plt.title('Calibration Curve')
    plt.plot(mpv, fop, marker='.', linestyle='', markersize=18)
    plt.plot([minimum,maximum],[minimum,maximum], linestyle='--', color='gray')
    plt.xlim((minimum,maximum))
    plt.ylim((minimum,maximum))
    plt.xlabel('Mean fit', fontsize=15)
    plt.ylabel('Mean outcome', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.show()
    
    
# LOAD DATASET:
    # full dataset:
data = pd.read_csv('dcr.csv')
data_loan_id =  data['id'].unique() # 50,000 loans
data['default_time'].value_counts(normalize = True)*100 # 97.56/2.44 ratio

    # sample dataset:
#sample_data = pd.read_csv('dcr_sample.csv')
#sample_loan_id = sample_data['id'].unique() # 5,000 loans
#sample_data['default_time'].value_counts(normalize = True)*100 # 97.54/2.46 ratio
#del sample_data
# the downsampled dataset still maintains the same negative/positive classes ratio

# FEATURE ENGINEERING:
id_col = data['id']
   # The following collums are dropped:
   # id, payoff_time, status_time are dropped due to being irrelevant to PD modelling
   # lgd_time and recovery_res are dropped as they are related to LGD component 
data.drop(columns=['id', 'res_time', 'payoff_time', 'status_time', 'lgd_time', 'recovery_res'], inplace=True)

    # For state_orig_time, we use one-hot encoding to encode this categorical data
data = pd.concat([data, pd.get_dummies(data["state_orig_time"],prefix="state")], axis=1)
    # remove the original column after one-hot encoding
data.drop(columns=['state_orig_time'], inplace=True)

    # FEATURE CREATION:
data['balance_orig_time'] = data['balance_orig_time'].replace(0, 1) # some of the derived economic features need balance at origination to be different than 0 to yield finite values
# Mortgage borrowers make annuity payments (fixed). Annuities are calculated as follows:
data.loc[:, 'annuity'] = ((data.loc[:,'interest_rate_time']/(100*4))*data.loc[:,'balance_orig_time'])/(1-(1+data.loc[:,'interest_rate_time']/(100*4))**(-(data.loc[:,'mat_time']-data.loc[:,'orig_time'])))
# Scheduled balance is computed as the difference between future value of the loan balance at origination and less future value 
# of all annuity payments made
data.loc[:,'balance_scheduled_time']  = data.loc[:,'balance_orig_time']*(1+data.loc[:,'interest_rate_time']/(100*4))**(data.loc[:,'time']-data.loc[:,'orig_time'])-data.loc[:,'annuity']*((1+data.loc[:,'interest_rate_time']/(100*4))**(data.loc[:,'time']-data.loc[:,'orig_time'])-1)/(data.loc[:,'interest_rate_time']/(100*4))
# property  price at origination. It is derived by dividing balance at origination with LTV at origination
data.loc[:,'property_orig_time'] = data.loc[:,'balance_orig_time']/(data.loc[:,'LTV_orig_time']/100)
# cep is an acronym for cummulative excess payment. It is a function of scheduled balance minus actual balance over property value
# cep is an approximation of the liquidity of borrowers
data.loc[:,'cep_time']= (data.loc[:,'balance_scheduled_time'] - data.loc[:,'balance_time'])/data.loc[:,'property_orig_time']

data.loc[:,'age'] = (data.loc[:,'time']-data.loc[:,'first_time']+1)
data.loc[:,'time_till_maturity'] = (data.loc[:,'mat_time']-data.loc[:,'time']+1)

# Equity time variable is created as follows: 
data.loc[:, 'equity_time'] = 1 - (data.loc[:, 'LTV_time']/100)

# Creating lagged macro values:
econ_df = data.groupby('time')['uer_time', 'gdp_time'].max().reset_index (drop=False)
    
econ_df['gdp_time_lag_1'] = econ_df['gdp_time'].shift(1)
econ_df['gdp_time_lag_1'].iloc[0] = econ_df['gdp_time'].iloc[0]
econ_df['gdp_time_lag_2'] = econ_df['gdp_time'].shift(2)
econ_df['gdp_time_lag_2'].iloc[0:2] = econ_df['gdp_time'].iloc[0]

econ_df['uer_time_lag_1'] = econ_df['uer_time'].shift(1)
econ_df['uer_time_lag_1'].iloc[0] = econ_df['uer_time'].iloc[0]
econ_df['uer_time_lag_2'] = econ_df['uer_time'].shift(2)
econ_df['uer_time_lag_2'].iloc[0:2] = econ_df['uer_time'].iloc[0]

econ_df.drop(columns=['uer_time','gdp_time'], inplace=True)

data = pd.merge(data, econ_df, on='time', how='left')

    # MISSING VALUE IMPUTATION:
    
    # Examine for NaN values
nan_columns = data.columns[data.isna().any()]
print("Columns with NaN values:")
print(nan_columns)
nan_values = data.isna().sum()

    # Examine for inf values:
inf_columns = data.columns[np.isinf(data).any()]
inf_rows = data.index[np.isinf(data).any(axis=1)]
inf_df = data.loc[inf_rows]

data['annuity'] = data['annuity'].replace([np.inf, -np.inf], np.nan)

kds = mf.ImputationKernel(
  data,
  save_all_iterations=True,
  random_state=100
)

# Run the MICE algorithm for 5 iterations - the more iteration the more accurate imputation. Recommended number of iteration is 5.
kds.mice(5)

# Return the completed dataset.
data_imputed = kds.complete_data()

# Checking for existence of NaN values
nan_columns = data_imputed.columns[data_imputed.isna().any()]

    # OUTLIERS DETECTION:
    
# These continuous variables are chosen to check for outliers:

cols = ['balance_time', 'LTV_time', 'balance_orig_time', 'FICO_orig_time', 'LTV_orig_time', 'age'
        , 'property_orig_time', 'cep_time', 'annuity', 'balance_scheduled_time']

for col in cols:
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False, bins=100)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col])
    plt.title(col)
    plt.show()
    
# Perform feature clipping on age variable as there are only a few outliers beyond 40
data_imputed.loc[data['age'] >= 40, 'age'] = 40

    # CREATING 5 different dataset (without replacement) from the original dataset:
data_imputed = pd.concat([data_imputed, id_col], axis=1)
remaining_loans = data_loan_id.copy()
remaining_loans = remaining_loans.tolist()
combination_size = 10000
dataset = []

random.seed(42)

while len(remaining_loans) != 0:
    current_combination = random.sample(remaining_loans, combination_size)       
    sub_dataset = data_imputed[data_imputed['id'].isin(current_combination)]
    dataset.append(sub_dataset)
    remaining_loans = [x for x in remaining_loans if x not in current_combination]
    print(len(remaining_loans))

# The ratio of 97.5/2.5 is preserved in each randomly created dataset

for i in range(len(dataset)):
    dataset[i].reset_index(inplace = True)
    dataset[i].drop(['id', 'index'], axis = 1, inplace = True)
    print(dataset[i]['default_time'].value_counts(normalize = True)*100)

    # TRAINING/TESTING SPLIT ON EACH DATASET:
train_test_dict = {}

for i in range(len(dataset)):    
    X_train, X_test, y_train, y_test = train_test_split(dataset[i].drop(['default_time'], axis = 1), dataset[i][['default_time']], 
                                                        test_size=0.2, random_state=4)
    train_test_dict[f'X_train_{i}'] = X_train
    train_test_dict[f'X_test_{i}'] = X_test
    train_test_dict[f'y_train_{i}'] = y_train
    train_test_dict[f'y_test_{i}'] = y_test
    

X_train_df = pd.concat([train_test_dict['X_train_0'], train_test_dict['X_train_1'],
                      train_test_dict['X_train_2'], train_test_dict['X_train_3'], train_test_dict['X_train_4']], 
                     ignore_index=True)

y_train_df = pd.concat([train_test_dict['y_train_0'], train_test_dict['y_train_1'],
                      train_test_dict['y_train_2'], train_test_dict['y_train_3'], train_test_dict['y_train_4']], 
                     ignore_index=True)

    # FEATURE SELECTION:
        
# Since we don't want to repeat feature selection on each dataset, we merge all the training datasets into 1 dataframe.
# The reason why we could perform such merging data is because each dataset is similar to each other in terms of negative/positive ratio.
# Moreover, we would like to select features that has better generalization so performing feature selection on a combined training
# dataset is the way to achieve it.

# Selection of dichomotous explanatory variables:
    
df_dichotomous_variables = X_train_df[['REtype_CO_orig_time','REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time',
                                        'state_AK', 'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO', 'state_CT', 'state_DC',
                                        'state_DE', 'state_FL', 'state_GA', 'state_HI', 'state_IA', 'state_ID', 'state_IL', 'state_IN',
                                        'state_KS', 'state_KY', 'state_LA', 'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN',
                                        'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE', 'state_NH', 'state_NJ',
                                        'state_NM', 'state_NY', 'state_OH', 'state_OK', 'state_OR', 'state_PA', 'state_PR', 'state_RI',
                                        'state_SC', 'state_SD', 'state_TN', 'state_TX', 'state_UT', 'state_VA', 'state_VT',
                                        'state_WA', 'state_WI', 'state_WV', 'state_WY']]

df_dichotomous_variables.reset_index(inplace = True)
df_dichotomous_variables.drop(['index'], axis = 1, inplace = True)
combined_dichotomous = pd.concat([y_train_df, df_dichotomous_variables], axis=1)
chi_squared_df = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for column in combined_dichotomous.columns:
    if column != 'default_time':
            # Create a contingency table
        contingency_table = pd.crosstab(combined_dichotomous['default_time'], combined_dichotomous[column])
            
            # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)
            
            # Store results in the DataFrame
        chi_squared_df = chi_squared_df.append({'Variable': column, 'Chi2': chi2, 'P-value': format(p, '.3f')}, ignore_index=True)

print(chi_squared_df)

# we exclude REtype_CO_orig_time and REtype_PU_orig_time due to statistical insignificance
# for state variables, since majority of them are statistically significant, we will keep them to allow our model to reflect
# regional effect of PD.

df_continuous_variables = X_train_df[['balance_time', 'LTV_time', 'interest_rate_time', 'rate_time', 'hpi_time', 'gdp_time', 'uer_time',
                                   'FICO_orig_time', 'LTV_orig_time', 'Interest_Rate_orig_time', 'hpi_orig_time', 'annuity', 'balance_orig_time',
                                   'balance_scheduled_time', 'property_orig_time', 'cep_time', 'age', 'time_till_maturity', 'mat_time',
                                   'equity_time', 'gdp_time_lag_1', 'gdp_time_lag_2', 'uer_time_lag_1', 'uer_time_lag_2']]

combined_continuous = pd.concat([y_train_df, df_continuous_variables], axis=1)

point_biserial_df = pd.DataFrame(columns=['Variable', 'PointBiserialCorr', 'P-value'])

for column in combined_continuous.columns:
    if column != 'default_time':
        # Calculate point-biserial correlation
        corr, p_value = pointbiserialr(combined_continuous['default_time'], combined_continuous[column])
        
        # Store results in the DataFrame
        point_biserial_df = point_biserial_df.append({'Variable': column, 'PointBiserialCorr': corr, 'P-value': format(p_value, '.3f')}, ignore_index=True)

# Display the results
print(point_biserial_df)

# we exclude balance_orig_time, balance_scheduled_time, cep_time and uer_time_lag_2 due to statistical insignificance

woe_df = pd.DataFrame(columns=['Variable', 'Information_Value'])

for column in combined_continuous.columns:
    if column != 'default_time':
        # Calculate point-biserial correlation
        _, IV = woe(data_in = combined_continuous, target = 'default_time', variable = column, bins = 10, binning = 'True')
        
        # Store results in the DataFrame
        woe_df = woe_df.append({'Variable': IV['name'][0], 'Information_Value': IV['IV'][0]}, ignore_index=True)

woe_df = woe_df.sort_values(by='Information_Value', ascending=False)

woe_df

    # Selection rule:
        
# Less than 0.02: IV is too low to support inclusion of feature in the model;
# Between 0.02 and 0.1: IV suggests a weak relation between a feature and default → consider exclusion of feature in the model;
# Between 0.1 and 0.3: IV suggests a moderate relation between a feature and default → consider inclusion of feature in the model;
# Over 0.3: IV suggests a strong relation between a feature and default → inclusion of feature.

# We further remove the following variables due to weak relationship with default (IV < 0.1):
    # uer_time
    # uer_time_lag_1
    # property_orig_time
    # LTV_orig_time
    # annuity
    # balance_time

# Checking correlation among continuous variables to further exclude highly correlated variables:
    # Selection rules:
        # Favor current variable over at-origination variable and
        # Favor variable with higher Information Value (based on weight of evidence ranking)
continuous_variables = ['LTV_time', 'interest_rate_time', 'rate_time', 'hpi_time', 'gdp_time', 'FICO_orig_time', 'Interest_Rate_orig_time', 
                        'hpi_orig_time', 'age', 'time_till_maturity', 'mat_time', 'equity_time', 'gdp_time_lag_1', 'gdp_time_lag_2', 
                        'uer_time_lag_1']
df_continuous_variables_after_selection = X_train[continuous_variables]
correlation_matrix, p_values = spearmanr(df_continuous_variables_after_selection, nan_policy='omit')     
correlation_df = pd.DataFrame(correlation_matrix, index=continuous_variables, columns=continuous_variables)

# Almost perfect correlation between LTV_time and equity_time => discard equity_time due to lower IV compared to LTV_time
# High correlation between interest_rate_time and Interest_Rate_orig_time => discard Interest_Rate_orig_time due to having lower IV and being at-origination variable
# Discard gdp_time_lag_1 and gdp_time_lag_2 due to high correlation with gdp_time and lower IV

    # DATA EXPLORATION ON SELECTED VARIABLES:

# This step is done to check if any meaningful relationship can be drawn from the selected variables in order to form new variables through:
    # Interaction among features
    # Transforming features
    
drop_variables = ['REtype_CO_orig_time', 'REtype_PU_orig_time','balance_time', 'balance_scheduled_time', 'balance_orig_time',
                  'LTV_orig_time', 'annuity', 'uer_time_lag_2', 'uer_time', 'property_orig_time', 'time',  'uer_time_lag_1',
                  'orig_time', 'first_time', 'gdp_time_lag_1', 'gdp_time_lag_2', 'Interest_Rate_orig_time', 
                  'annuity', 'cep_time', 'equity_time', 'orig_time', 'first_time', 'time']

selected_cont_variables = ['interest_rate_time', 'rate_time', 'hpi_time', 'gdp_time', 'FICO_orig_time',
                        'hpi_orig_time', 'age', 'time_till_maturity', 'mat_time', 'LTV_time']

selected_dichotomous_variables = ['state_AK', 'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO', 'state_WA', 'state_WI',
                                  'state_DE', 'state_FL', 'state_GA', 'state_HI', 'state_IA', 'state_ID', 'state_IL', 'state_IN',
                                  'state_KS', 'state_KY', 'state_LA', 'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN',
                                  'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE', 'state_NH', 'state_NJ',
                                  'state_NM', 'state_NY', 'state_OH', 'state_OK', 'state_OR', 'state_PA', 'state_PR', 'state_RI',
                                  'state_SC', 'state_SD', 'state_TN', 'state_TX', 'state_UT', 'state_VA', 'state_VT', 'state_WV',
                                  'state_WY', 'state_CT', 'state_DC', 'state_NV', 'state_VI', 'REtype_SF_orig_time', 'investor_orig_time']

train_df = pd.concat([X_train_df, y_train_df], axis = 1)
train_cont_df = train_df[['interest_rate_time', 'rate_time', 'hpi_time', 'gdp_time', 'FICO_orig_time',
                        'hpi_orig_time', 'age', 'time_till_maturity', 'mat_time', 'LTV_time', 'default_time']]
X_train_selected = X_train_df.drop(drop_variables, axis = 1)


# Plotting distribution of default on selected continuous variables:
    
for i in X_train_selected[selected_cont_variables].columns:
    
    selected_feature = i
    
    class0 = train_df[train_df['default_time'] == 0][selected_feature]
    class1 = train_df[train_df['default_time'] == 1][selected_feature]
    
    # Clip the data at the 1st and 99th percentiles
    clipped_class0 = np.clip(class0, np.percentile(class0, 1), np.percentile(class0, 99))
    clipped_class1 = np.clip(class1, np.percentile(class1, 1), np.percentile(class1, 99))
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Plot histogram for Class 0
    sns.histplot(clipped_class0, kde=True, color='blue', ax=axes[0], bins = 25)
    axes[0].set_title(f'Histogram for Class 0 - {selected_feature}')
    axes[0].set_xlabel(selected_feature)
    axes[0].set_ylabel('Frequency')
    
    # Plot histogram for Class 1
    sns.histplot(clipped_class1, kde=True, color='orange', ax=axes[1], bins = 25)
    axes[1].set_title(f'Histogram for Class 1 - {selected_feature}')
    axes[1].set_xlabel(selected_feature)
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
# We could see that the distribution of negative class and positive class is very resembling.
# Another observation is that, in all bins, observations of majority class outnumber those of majority class by a large margin.
# This suggests that we need to check for the distribution of default in a higher dimension (default distribution plot on 2 features)

# To plot the distribution of default on 2 features, we consider the combination of LTV_time with other selected continuous variables
# The reason for such choice of combination is that LTV_time has the highest Information Value so it makes a better sense to consider
# only the combination between LTV and other variables. We could check for the distribution based on all possible combination but it is
# simply very time consuming.

x_var = ['LTV_time']
for j in x_var:
    for i in ['interest_rate_time', 'mat_time', 'time_till_maturity', 'hpi_orig_time', 'gdp_time',
              'FICO_orig_time', 'rate_time', 'hpi_time']:
        sns.scatterplot(x = j, y = i, data = train_df, hue = 'default_time',
                        palette= {0: 'blue', 1: 'orange'})
        plt.title(f'Scatter Plot of {i} and {j} vs default = 1')
        plt.xlabel(f'{j}')
        plt.ylabel(f'{i}')
        plt.show()
# As expected, on a 2 dimension level, we still see the majority class overpresents in most of the bins. There are some areas where
# the minority class overepresents but these areas/bins are scattered (or occur at random) without forming a cluster or any meaningful trend
# We plot the distribution of case where default = 1 to see the general distribution of default  = 1 on 2 features    
x_var = ['LTV_time']
for j in x_var:
    for i in ['interest_rate_time', 'mat_time', 'time_till_maturity', 'hpi_orig_time', 'gdp_time',
              'FICO_orig_time', 'rate_time', 'hpi_time']:
        sns.scatterplot(x = j, y = i, data = train_df[train_df['default_time']==1], hue = 'default_time',
                        palette= {0: 'blue', 1: 'orange'})
        plt.title(f'Scatter Plot of {i} and {j} vs default = 1')
        plt.xlabel(f'{j}')
        plt.ylabel(f'{i}')
        plt.show()
# Nothing useful can be drawn from these graphs

# Checking for the default distribution on LTV_time vs. REtype_SF_orig_time and investor_orig_time: 
for i in ['REtype_SF_orig_time', 'investor_orig_time']:
    target_column = 'default_time'
    sns.scatterplot(x = i, y = 'LTV_time' , hue=target_column, data=train_df[train_df[i]==1])
    plt.title(f'Bar Plot of {target_column} for each class of {i} and {j}')
    plt.xlabel(i)
    plt.ylabel(j)
    plt.legend(title=target_column)
    plt.show()
# Again, majority class overepresents at almost any LTV value.

# Based on the data exploration results, we could see that it is almost impossible to create any interaction or transforming the data
# in the way that tells which class apart from the other.

# TRAIN VARIOUS MODELS ON RESAMPLED DATA:
rc = RobustScaler()

# Create optuna study object with relevant hyperparameters and corresponding range of values being considered

    # Logistic Regression:
def objective_lr(trial, X, y):
    params = {'solver': trial.suggest_categorical('solver', ['saga']), # only saga solver is considered due to its ability to work faster on large dataset and it can accommodate all type of regularization
              'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']), 
              'C': trial.suggest_float("C", 0.01, 100, log=True),
             }
    clf = make_pipeline(LogisticRegression(**params, class_weight='balanced', random_state = 16))
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    return np.mean(scores)

    # KNN Classifier:
def objective_knn(trial, X, y):
    params = {'n_neighbors': trial.suggest_int('n_neighbors', 1, 10),
              'weights': trial.suggest_categorical('weights', ['uniform', 'distance']), 
              'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
              'p': trial.suggest_int("p",1,2)
             }
    clf = make_pipeline(KNeighborsClassifier(**params, random_state = 16))
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)

    # Linear SVC:
def objective_linear_svc(trial, X, y):
    params = {'C': trial.suggest_loguniform('C', 1e-4, 1e4), 
             }
    clf = make_pipeline(LinearSVC(**params, class_weight="balanced", random_state = 16))
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)

    # Random Forest:
def objective_rf(trial, X, y):
    params = {'n_estimators': trial.suggest_int('n_estimators', 10, 1000, log=True),
              'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']), 
              'max_depth': trial.suggest_int('max_depth', 2, 100, log=True),
              'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
              'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, log=True),
              'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10, log=True),              
             }
    clf = make_pipeline(RandomForestClassifier(**params, class_weight = 'balanced', random_state = 16))
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    return np.mean(scores)

    # XG Boost Classifier:
def objective_xgbc(trial, X, y):
    params = {"objective": "binary:logistic",
                  'random_state': trial.suggest_int('random_state', 0, 0), 
                  'n_estimators': trial.suggest_int('n_estimators', 10, 1000), # num_boosting_round
                  'max_depth': trial.suggest_int('max_depth', 1, 10), 
                  'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0), 
                  'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 1e+0), # eta
                  'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0), 
                  'min_split_loss': trial.suggest_float('min_split_loss', 1e-8, 1.0, log=True), # gamma
                  'booster': trial.suggest_categorical('booster', ['gbtree']), # , excluding 'gblinear' (less accurate than gbtree), 'dart' (time consuming)
                  'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True), # lambda
                  'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0)# alpha
                 }
    clf = make_pipeline(XGBClassifier(**params, use_label_encoder=False, n_jobs = 8, random_state = 16))
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    return np.mean(scores)

    # Neural Network Classifier:
def objective_nnc(trial, X, y):
    params = {
        "hidden_layer_sizes": trial.suggest_int('hidden_layer_sizes', 2, 100, log=True),
        "activation": trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
        "alpha": trial.suggest_float("alpha", 1e-4, 1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
        "max_iter": trial.suggest_int('max_iter', 100, 500)       
    }
    clf = make_pipeline(MLPClassifier(**params, random_state = 16))
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    return np.mean(scores)

    # Light GBM Classifier:
def objective_lgbc(trial, X, y):
    params = {
        "objective": "binary",
        'verbose':-1,
        'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 3, 30.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 3, 30.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    clf = make_pipeline(LGBMClassifier(**params, class_weight = 'balanced', n_jobs = 8, random_state = 12))
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    return np.mean(scores)

# Perform hyperparameter tunning and save best parameters for every dataset:

    # WARNING !!!!! - Running this loop will take more than a day to finish. Outputs are the best hyperparameters for each model on every dataset
    # Unless you really want to double check, it is advisable to skip this part

### BEGINNING OF SKIPPABLE PART ###  
best_param_dict = {}

for i in range(len(dataset)):    
    
    X_train = train_test_dict[f'X_train_{i}'].copy()
    X_train.reset_index(inplace = True)
    X_train.drop(['index'], axis = 1, inplace = True)
    
    y_train = train_test_dict[f'y_train_{i}'].copy()
    y_train.reset_index(inplace = True)
    y_train.drop(['index'], axis = 1, inplace = True)

    X_train_selected = X_train.drop(drop_variables, axis = 1)

    # Create SMOTEENN instance for training data sampling

    sampler = SMOTEENN(sampling_strategy= 0.1, n_jobs = 8, random_state= 10)
    X_resampled, y_resampled = sampler.fit_resample(X_train_selected, y_train)
    
    # Scaled data is used for svc and knn since these models' fitting mechanism based on distance among observations
    X_train_scaled = pd.DataFrame(
        rc.fit_transform(X_train_selected[selected_cont_variables]),
        columns = X_train_selected[selected_cont_variables].columns
    )
    
    X_train_scaled = pd.concat([X_train_selected[selected_dichotomous_variables], X_train_scaled], axis=1)
    
    X_scaled_resampled, y_scaled_resampled = sampler.fit_resample(X_train_scaled, y_train)

    study_lr = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_lr.optimize(functools.partial(objective_lr, X=X_resampled, y=y_resampled), n_trials=30, n_jobs=8)
    
    study_knn = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_knn.optimize(functools.partial(objective_knn, X=X_scaled_resampled, y=y_scaled_resampled), n_trials=30, n_jobs=8)
    
    study_svc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_svc.optimize(functools.partial(objective_linear_svc, X=X_scaled_resampled, y=y_scaled_resampled), n_trials=30, n_jobs=8)
    
    study_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_rf.optimize(functools.partial(objective_rf, X=X_resampled, y=y_resampled), n_trials=30, n_jobs=8)
    
    study_xgbc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_xgbc.optimize(functools.partial(objective_xgbc, X=X_resampled, y=y_resampled), n_trials=30)
    
    study_nnc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_nnc.optimize(functools.partial(objective_nnc, X=X_resampled, y=y_resampled), n_trials=30, n_jobs=8)
    
    study_lgbc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study_lgbc.optimize(functools.partial(objective_lgbc, X=X_resampled, y=y_resampled), n_trials=30, n_jobs= 8)
    
    
    best_param_dict[f'study_lr_{i}'] = study_lr
    best_param_dict[f'study_knn_{i}'] = study_knn
    best_param_dict[f'study_svc_{i}'] = study_svc
    best_param_dict[f'study_rf_{i}'] = study_rf
    best_param_dict[f'study_xgbc_{i}'] = study_xgbc
    best_param_dict[f'study_nnc_{i}'] = study_nnc
    best_param_dict[f'study_lgbc_{i}'] = study_lgbc
    
    print (f'dataset {i} is done')
### ENDING OF SKIPPABLE PART ###  

model_params = ['lr', 'knn', 'svc', 'rf', 'xgbc', 'nnc', 'lgbc']

# save best params obtained from hyperparameter tunning as a pickle object
#for i in range(len(dataset)):
#    for j in model_params:
#        study = best_param_dict[f'study_{j}_{i}']
#        with open(f'study_{j}_{i}.pkl', 'wb') as study_file:
#            pickle.dump(study, study_file)

# NOTICE !!!!! If you have run the previous loop (the one that takes more than a day to finish) to get the best parameters, please skip the loop below as it simply import the best
# parameters obtained from the previous step

### BEGINNING OF SKIPPABLE PART ###  
best_param_dict = {}
for i in range(len(dataset)):
    for j in model_params:
        study = joblib.load(f'study_{j}_{i}.pkl')
        best_param_dict[f'study_{j}_{i}'] = study
### ENDING OF SKIPPABLE PART ###        

# CALCULATING IN-SAMPLE, OUT-OF-SAMPLE AUC, MAE FOR EACH MODEL ON EACH :
# WARNING !!!!! Running the loop below will take several hours. Unless necessary, this part can be skipped as the output from 
# it can be imported by running the code on line 888.

### BEGINNING OF SKIPPABLE PART ###  
results_df = pd.DataFrame(columns=['Model', 'Test_Set', 'In_Sample_ROC_AUC', 'Out_of_Sample_ROC_AUC', 
                                   'In_Sample_MAE', 'Out_of_Sample_MAE'])

out_sample_df = {}

drop_variables_interim = ['REtype_CO_orig_time', 'REtype_PU_orig_time','balance_time', 'balance_scheduled_time', 'balance_orig_time',
                          'LTV_orig_time', 'annuity', 'uer_time_lag_2', 'uer_time', 'property_orig_time', 'uer_time_lag_1',
                          'orig_time', 'first_time', 'gdp_time_lag_1', 'gdp_time_lag_2', 'Interest_Rate_orig_time', 
                          'annuity', 'cep_time', 'equity_time', 'orig_time', 'first_time']

for i in range(len(dataset)):
    for j in model_params:
        
        if j == 'lr':
            model = LogisticRegression(**best_param_dict[f'study_{j}_{i}'].best_params)
        elif j == 'knn':
            model = KNeighborsClassifier(**best_param_dict[f'study_{j}_{i}'].best_params)
        elif j == 'svc':
            model = CalibratedClassifierCV(LinearSVC(**best_param_dict[f'study_{j}_{i}'].best_params))
        elif j == 'rf':
            model = RandomForestClassifier(**best_param_dict[f'study_{j}_{i}'].best_params)
        elif j == 'xgbc':
            model = XGBClassifier(**best_param_dict[f'study_{j}_{i}'].best_params)
        elif j == 'nnc':
            model = MLPClassifier(**best_param_dict[f'study_{j}_{i}'].best_params)
        else:
            model = LGBMClassifier(**best_param_dict[f'study_{j}_{i}'].best_params)
        
        # Train data:
        
        X_train = train_test_dict[f'X_train_{i}'].copy()
        X_train.reset_index(inplace = True)
        X_train.drop(['index'], axis = 1, inplace = True)
        
        y_train = train_test_dict[f'y_train_{i}'].copy()
        y_train.reset_index(inplace = True)
        y_train.drop(['index'], axis = 1, inplace = True)

        X_train_selected = X_train.drop(drop_variables_interim, axis = 1)

        # Test data:
        
        X_test = train_test_dict[f'X_test_{i}'].copy()
        X_test.reset_index(inplace = True)
        X_test.drop(['index'], axis = 1, inplace = True)
        
        y_test = train_test_dict[f'y_test_{i}'].copy()
        y_test.reset_index(inplace = True)
        y_test.drop(['index'], axis = 1, inplace = True)

        X_test_selected = X_test.drop(drop_variables, axis = 1)
        out_sample_time = X_test['time'].values
        
        # Create SMOTEENN instance
        # I could have saved the resampled training data in previous step. However, I just realized it after running the hyperparameter tunning.
        # Although this step is repeated, the resampled dataset is the same as the one used for hyperparameter tunning.
        sampler = SMOTEENN(sampling_strategy= 0.1, n_jobs = 8, random_state= 10)
        X_resampled, y_resampled = sampler.fit_resample(X_train_selected, y_train)
        in_sample_time = X_resampled['time'].values
        X_resampled.drop(['time'], axis = 1, inplace = True)
        
        X_train_scaled = pd.DataFrame(
            rc.fit_transform(X_train_selected[selected_cont_variables]),
            columns = X_train_selected[selected_cont_variables].columns
        )
        
        X_test_scaled = pd.DataFrame(
            rc.transform(X_test_selected[selected_cont_variables]),
            columns = X_test_selected[selected_cont_variables].columns
        )
        
        X_train_scaled = pd.concat([X_train_selected[selected_dichotomous_variables], X_train_scaled], axis=1)
        X_test_scaled = pd.concat([X_test_selected[selected_dichotomous_variables], X_test_scaled], axis=1)
        
        X_scaled_resampled, y_scaled_resampled = sampler.fit_resample(X_train_scaled, y_train)
        
        # predict probability of each model on testing data
        if j not in ['knn', 'svc']:    
            model.fit(X_resampled, y_resampled)
            in_sample_prob_pred = model.predict_proba(X_resampled)[:,1].T
            out_sample_prob_pred = model.predict_proba(X_test_selected)[:,1].T
            out_sample_pred = model.predict(X_test_selected)
            in_sample_auc = roc_auc_score(y_resampled, in_sample_prob_pred)
            out_sample_auc = roc_auc_score(y_test, out_sample_prob_pred)
            in_sample_test_result = test_result(in_sample_prob_pred, y_resampled['default_time'].values, in_sample_time)
            out_sample_test_result = test_result(out_sample_prob_pred, y_test['default_time'].values, out_sample_time)
        else:
            model.fit(X_scaled_resampled, y_scaled_resampled)
            in_sample_prob_pred = model.predict_proba(X_scaled_resampled)[:,1].T
            out_sample_prob_pred = model.predict_proba(X_test_scaled)[:,1].T
            out_sample_pred = model.predict(X_test_scaled)
            in_sample_auc = roc_auc_score(y_scaled_resampled, in_sample_prob_pred)
            out_sample_auc = roc_auc_score(y_test, out_sample_prob_pred)
            in_sample_test_result = test_result(in_sample_prob_pred, y_resampled['default_time'].values, in_sample_time)
            out_sample_test_result = test_result(out_sample_prob_pred, y_test['default_time'].values, out_sample_time)
        
        # calculating average fitted probability and actual probability on period level
        in_sample_means = in_sample_test_result.groupby('time')[['fit', 'outcome']].mean().reset_index(drop=False)
        out_sample_means = out_sample_test_result.groupby('time')[['fit', 'outcome']].mean().reset_index(drop=False)
        
        # calculating in-sample and out-of-sample mean absolute error
        mae_in_sample = mean_absolute_error(in_sample_means['outcome'], in_sample_means['fit'])
        mae_out_sample = mean_absolute_error(out_sample_means['outcome'], out_sample_means['fit'])
        
        # saving in-sample and out-of-sample auc and mae for each test set
        results_df = results_df.append({'Model':j, 'Test_Set': i, 'In_Sample_ROC_AUC': in_sample_auc,
                                       'Out_of_Sample_ROC_AUC': out_sample_auc, 'In_Sample_MAE': mae_in_sample,
                                       'Out_of_Sample_MAE': mae_out_sample}, ignore_index = True)
        
        # saving out-of-sample probability prediction and actual values
        out_sample_df[f'model_{j}_prob_prediction_{i}'] = out_sample_prob_pred
        out_sample_df[f'model_{j}_prediction_{i}'] = out_sample_pred
        out_sample_df[f'model_{j}_y_test_{i}'] = y_test['default_time'].values
        out_sample_df[f'model_{j}_time_{i}'] = out_sample_time
        
        print (f'model {j} dataset {i} is done')
### ENDING OF SKIPPABLE PART ###  

# Save the calculation to "pickle" file and "csv" file:
    
# with open('out_sample_df.pkl', 'wb') as out_sample_data:
#    pickle.dump(out_sample_df, out_sample_data)

#results_df.to_csv('in_out_sample_results.csv', index=False)

# WARNING !!!! If you skip the code above, you need to run the code below to load the out-of-sample test data and model prediction

### BEGINNING OF SKIPPABLE PART ###
out_sample_df = joblib.load('out_sample_df.pkl')
### ENDING OF SKIPPABLE PART ###  

# Aggregating out-of-sample result from 5 testing dataset into 1 dataframe for each model        
lr_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
knn_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
svc_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
rf_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
xgbc_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
nnc_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
lgbc_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])
ada_results = pd.DataFrame(columns = ['fit', 'outcome', 'time'])

for i in range(len(dataset)):
    
    lr_df = pd.DataFrame({'fit': out_sample_df[f'model_lr_prob_prediction_{i}'].tolist(),
               'outcome': out_sample_df[f'model_lr_y_test_{i}'].tolist(),
               'time': out_sample_df[f'model_lr_time_{i}'].tolist()})
    lr_results = pd.concat([lr_results, lr_df])

    knn_df = pd.DataFrame({'fit': out_sample_df[f'model_knn_prob_prediction_{i}'].tolist(),
                'outcome': out_sample_df[f'model_knn_y_test_{i}'].tolist(),
                'time': out_sample_df[f'model_knn_time_{i}'].tolist()})
    knn_results = pd.concat([knn_results, knn_df])
    
    svc_df = pd.DataFrame({'fit': out_sample_df[f'model_svc_prob_prediction_{i}'].tolist(),
                'outcome': out_sample_df[f'model_svc_y_test_{i}'].tolist(),
                'time': out_sample_df[f'model_svc_time_{i}'].tolist()})
    svc_results = pd.concat([svc_results, svc_df])
    
    rf_df = pd.DataFrame({'fit': out_sample_df[f'model_rf_prob_prediction_{i}'].tolist(),
               'outcome': out_sample_df[f'model_rf_y_test_{i}'].tolist(),
                'time': out_sample_df[f'model_rf_time_{i}'].tolist()})
    rf_results = pd.concat([rf_results, rf_df])
    
    xgbc_df = pd.DataFrame({'fit': out_sample_df[f'model_xgbc_prob_prediction_{i}'].tolist(),
                 'outcome': out_sample_df[f'model_xgbc_y_test_{i}'].tolist(),
                 'time': out_sample_df[f'model_xgbc_time_{i}'].tolist()})
    xgbc_results = pd.concat([xgbc_results, xgbc_df])
    
    nnc_df = pd.DataFrame({'fit': out_sample_df[f'model_nnc_prob_prediction_{i}'].tolist(),
                'outcome': out_sample_df[f'model_nnc_y_test_{i}'].tolist(),
                'time': out_sample_df[f'model_nnc_time_{i}'].tolist()})
    nnc_results = pd.concat([nnc_results, nnc_df])
    
    lgbc_df = pd.DataFrame({'fit': out_sample_df[f'model_lgbc_prob_prediction_{i}'].tolist(),
                 'outcome': out_sample_df[f'model_lgbc_y_test_{i}'].tolist(),
                 'time': out_sample_df[f'model_lgbc_time_{i}'].tolist()})
    lgbc_results = pd.concat([lgbc_results, lgbc_df])

# CALCULATING AGGREGATED FORECAST RESULTS:
validation(lr_results['fit'].values, lr_results['outcome'].values, lr_results['time'].values, 'Logistic Regression')
validation(knn_results['fit'].values, knn_results['outcome'].values, knn_results['time'].values, 'KNN Classifier')
validation(svc_results['fit'].values, svc_results['outcome'].values, svc_results['time'].values, 'Support Vector Classifier')
validation(rf_results['fit'].values, rf_results['outcome'].values, rf_results['time'].values, 'Random Forest Classifier')
validation(xgbc_results['fit'].values, xgbc_results['outcome'].values, xgbc_results['time'].values, 'XGBoost Classifier')
validation(nnc_results['fit'].values, nnc_results['outcome'].values, nnc_results['time'].values, 'Deep Neural Network Classifier')
validation(lgbc_results['fit'].values, lgbc_results['outcome'].values, lgbc_results['time'].values, 'LightGBM Classifier')
validation(ada_results['fit'].values, ada_results['outcome'].values, ada_results['time'].values, 'AdaBoost Classifier')
# Best performer is Random Forest based on overall performance on various metrics. However, it tends to overact during stressed periods.
# Logistic Regression and Support Vector Machine overshoots PD projection in their entirety of prediction.
# Deep Neural Network, XGBoost and LightGBM do a decent job at forecasting but tend to underpredict PD.
# KNN AUC suffers from its overfitting on training data (as a result of n_neighbors = 1 – obtained from tunning) resulting in 
# too many actual negative cases labelled as positive and vice versa.



# CONFUSION MATRIX FOR AGGREGATED RESULTS:
lr_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])
knn_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])
svc_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])
rf_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])
xgbc_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])
lgbc_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])
nnc_confusion_matrix = pd.DataFrame(columns = ['fit', 'outcome'])

for i in range(len(dataset)):
    for j in model_params:
        
        lr_df = pd.DataFrame({'fit': out_sample_df[f'model_lr_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_lr_y_test_{i}'].tolist()})
        lr_confusion_matrix = pd.concat([lr_confusion_matrix, lr_df])
        
        knn_df = pd.DataFrame({'fit': out_sample_df[f'model_knn_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_knn_y_test_{i}'].tolist()})
        knn_confusion_matrix = pd.concat([knn_confusion_matrix, knn_df])
        
        svc_df = pd.DataFrame({'fit': out_sample_df[f'model_svc_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_svc_y_test_{i}'].tolist()})
        svc_confusion_matrix = pd.concat([svc_confusion_matrix, svc_df])
        
        rf_df = pd.DataFrame({'fit': out_sample_df[f'model_rf_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_rf_y_test_{i}'].tolist()})
        rf_confusion_matrix = pd.concat([rf_confusion_matrix, rf_df])
        
        xgbc_df = pd.DataFrame({'fit': out_sample_df[f'model_xgbc_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_xgbc_y_test_{i}'].tolist()})
        xgbc_confusion_matrix = pd.concat([xgbc_confusion_matrix, xgbc_df])
        
        lgbc_df = pd.DataFrame({'fit': out_sample_df[f'model_lgbc_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_lgbc_y_test_{i}'].tolist()})
        lgbc_confusion_matrix = pd.concat([lgbc_confusion_matrix, lgbc_df])
        
        nnc_df = pd.DataFrame({'fit': out_sample_df[f'model_nnc_prediction_{i}'].tolist(),
                   'outcome': out_sample_df[f'model_nnc_y_test_{i}'].tolist()})
        nnc_confusion_matrix = pd.concat([nnc_confusion_matrix, nnc_df])

# Visualizing confusion matrix in a tabular form:
lr_matrix = confusion_matrix(lr_confusion_matrix['outcome'].tolist(), lr_confusion_matrix['fit'].tolist())
knn_matrix = confusion_matrix(knn_confusion_matrix['outcome'].tolist(), knn_confusion_matrix['fit'].tolist())
svc_matrix = confusion_matrix(svc_confusion_matrix['outcome'].tolist(), svc_confusion_matrix['fit'].tolist())
rf_matrix = confusion_matrix(rf_confusion_matrix['outcome'].tolist(), rf_confusion_matrix['fit'].tolist())
xgbc_matrix = confusion_matrix(xgbc_confusion_matrix['outcome'].tolist(), xgbc_confusion_matrix['fit'].tolist())
lgbc_matrix = confusion_matrix(lgbc_confusion_matrix['outcome'].tolist(), lgbc_confusion_matrix['fit'].tolist())
nnc_matrix = confusion_matrix(nnc_confusion_matrix['outcome'].tolist(), nnc_confusion_matrix['fit'].tolist())

matrix_dict = {'Logistic Regression':lr_matrix, 'KNN':knn_matrix, 'SVC':svc_matrix, 'Random Forest':rf_matrix, 
               'XGBoost':xgbc_matrix, 'LightGBM':lgbc_matrix, 'Neural Network': nnc_matrix}

models = ['Logistic Regression', 'KNN', 'SVC', 'Random Forest', 'XGBoost', 'LightGBM', 'Neural Network']
for model in models:
    sns.heatmap(matrix_dict[model], annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    
    plt.title(f'{model} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()        
# On loan level, it looks all the models do a poor job as predicting correctly default cases (default = 1) as they tend to label most of observations
# in the testing dataset as negative class. The number of correct default label (bottom right corner of confusion matrix) is miniscule 
# compared to the correct non-default label (top left corner). This is expected as the data is very imbalanced with majority class observations
# overrepresent in any bin/strata.


# Plotting distribution of out of sample observation:
sns.histplot(lgbc_results['time'].values, kde=True, color='blue')
plt.title('Distribution of out-of-sample observation by period')
plt.xlabel('time')
plt.tight_layout()
plt.show()
# We can see that there are lack of observations during the first 18 periods which results in inaccurate prediction of probability
# of default. As the number of observations increase, so does the accuracy (Look at the Time-Series Real-fit figures for more info)

