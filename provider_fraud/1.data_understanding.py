# Objetives:
# To gain better understanding of the dataset from different levels.
# To explore the dataset in order to uncover the potential hidden patterns.
# And, to find the answers of various WHYs.

import os
import sys
import math
import scipy as scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_graphs import cal_display_percentiles

#%matplotlib inline

pd.set_option('display.max_columns',30)
label_font_dict = {'family':'sans-serif','size':13.5,'color':'brown','style':'italic'}
title_font_dict = {'family':'sans-serif','size':16.5,'color':'Blue','style':'italic'}

# importing dataset

train_bene_df = pd.read_csv("data/Train_Beneficiarydata-1542865627584.csv")
train_ip_df = pd.read_csv("data/Train_Inpatientdata-1542865627584.csv")
train_op_df = pd.read_csv("data/Train_Outpatientdata-1542865627584.csv")

# Q1. How many unique beneficiaries we have in our dataset?
print(train_bene_df['BeneID'].nunique())

# Q2. How many records we have at the GENDER level?
print(train_bene_df['Gender'].unique())
train_bene_df['Gender'] = train_bene_df['Gender'].apply(lambda val: 0 if val == 2 else 1)

# Here, I'm displaying the distribution of BENEFICIARIES on the basis of GENDER
plt.figure(figsize=(10, 8))
fig = train_bene_df['Gender'].value_counts().plot(kind='bar', color=['yellow', 'purple'])

# Using the "patches" function we will get the location of the rectangle bars from the graph.
## Then by using those location(width & height) values we will add the annotations
for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height * 100) / train_bene_df.shape[0], 2))+"%"}', (x + width/2, y + height*1.015), ha='center', fontsize=13.5)

# Providing the labels and title to the graph
plt.xlabel("Gender Code", fontdict=label_font_dict)
plt.ylabel("Number or % share of patients\n", fontdict=label_font_dict)
plt.grid(which='major', linestyle="--", color='lightgrey')
plt.minorticks_on()
plt.title("Distribution of BENEFICIARIES based on GENDER\n", fontdict=title_font_dict)
plt.savefig('fig/gender_beneficiaries.png')

# Q3. Lets calculate the AGE of every BENEFICIARY?

train_bene_df['DOB'] = pd.to_datetime(train_bene_df['DOB'], format="%Y-%m-%d")
train_bene_df['Patient_Age_Year'] = train_bene_df['DOB'].dt.year
train_bene_df['Patient_Age_Month'] = train_bene_df['DOB'].dt.month

bene_age_year_df = pd.DataFrame(train_bene_df['Patient_Age_Year'].value_counts()).reset_index(drop=False)
bene_age_year_df.columns= ['year','num_of_beneficiaries']
bene_age_year_df = bene_age_year_df.sort_values(by='year')

sns.set_style('whitegrid')  # Setting Seaborn style

plt.figure(figsize=(21, 9))
fig = sns.barplot(data=bene_age_year_df, x='year', y='num_of_beneficiaries', palette='inferno')

for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/train_bene_df.shape[0],1))+"%"}',
                 (x + width/2, y + height*1.025),
                 ha='center', fontsize=13.5, rotation=90)

plt.xlabel("\nBeneficiary YEAR of Birth", fontdict=label_font_dict)
plt.xticks(rotation=90)
plt.ylabel("Number or % share of patients\n", fontdict=label_font_dict)
plt.minorticks_on()
plt.grid(which='major', linestyle="--", color='lightgrey')
plt.title("Distribution of BENEFICIARIES based on their YEAR of birth\n", fontdict=title_font_dict)
plt.savefig('fig/year_beneficiaries.png')

# Q6. Lets see the number of beneficiaries on the basis of State Codes.

sns.set_style('whitegrid')  # Setting Seaborn style

plt.figure(figsize=(20, 9))
fig = train_bene_df['State'].value_counts().plot(kind='bar')

for p in fig.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    fig.annotate(f'{str(round((height*100)/train_bene_df.shape[0],2))+"%"}',
                 (x + width/2, y + height*1.03),
                 ha='center', fontsize=13.5, rotation=90)

plt.xlabel("\nState Codes", fontdict=label_font_dict)
plt.ylabel("Number or % share of patients\n", fontdict=label_font_dict)
plt.grid(which='major', linestyle="--", color='lightgrey')
plt.minorticks_on()
plt.title("Distribution of BENEFICIARIES on the basis of States\n", fontdict=title_font_dict)
plt.savefig('fig/state_year_beneficiaries.png')
plt.show()

# ----------------------

RKD_YES_IP_R_percentiles = cal_display_percentiles(train_bene_df, label_font_dict,
                                                    title_font_dict,
                                                   x_col='RenalDiseaseIndicator',
                                                   y_col='IPAnnualReimbursementAmt',
                                                   title_lbl="Renal Kidney Disease = YES",
                                                   x_filter_code='Y')

# graphs for number of missings

# claims duration -----------------------------------------------------------------------------------------
train_op_df['ClaimStartDt'] = pd.to_datetime(train_op_df['ClaimStartDt'], format="%Y-%m-%d")
train_op_df['ClaimEndDt'] = pd.to_datetime(train_op_df['ClaimEndDt'], format="%Y-%m-%d")
train_op_df['Claim_Duration'] = (train_op_df['ClaimEndDt'] - train_op_df['ClaimStartDt']).dt.days
tot_claims_filed_for_specific_days = pd.DataFrame(train_op_df.groupby(['Claim_Duration'])['ClaimID'].count())
tot_insc_amount_for_claim_durations = pd.DataFrame(train_op_df.groupby(['Claim_Duration'])['InscClaimAmtReimbursed'].sum())
claim_clearance_amts = pd.merge(left=tot_claims_filed_for_specific_days, right=tot_insc_amount_for_claim_durations,
                                how='inner',
                                left_on=tot_claims_filed_for_specific_days.index,
                                right_on=tot_insc_amount_for_claim_durations.index)

claim_clearance_amts.columns = ['Claim_durations_in_days', 'Total_claims', 'All_Claims_Total_Amount']
claim_clearance_amts.head()

# Assuming claim_clearance_amts, label_font_dict, and title_font_dict are defined somewhere in your code.

sns.set_style('whitegrid')  # Set Seaborn style

plt.figure(figsize=(16, 16))
sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Total_claims',
              color='k', markers="^", linestyles="")
sns.pointplot(data=claim_clearance_amts, x='Claim_durations_in_days', y='Total_claims',
              color='coral', markers="", linestyles="-")

# Providing the labels and title to the graph
plt.xticks(rotation=90)
plt.xlabel("\nClaims Durations (in days)", fontdict=label_font_dict)
plt.ylabel("Total Claims\n", fontdict=label_font_dict)
plt.yticks(np.arange(0, 7500, 200))
plt.grid(which='major', linestyle="--", color='lightgrey')
plt.minorticks_on()
plt.title('\nTrend of "Total Filed Claims" for every duration (in days)', fontdict=title_font_dict)
plt.plot()
plt.savefig('fig/claim_duration.png')
plt.show()

# References

#https://medium.datadriveninvestor.com/medicare-provider-fraud-detection-f551dd941947