#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning - Awards Players

# In[1]:


import pandas as pd
from utils import *
import os

df = pd.read_csv('../data/raw/awards_players.csv')
df.head()

init_num_rows = df.shape[0]


# ## Dealing with Missing Values / Features

# In[2]:


nulls_values_by_column(df)


# ## Dealing with Duplicate values / Redundant Data

# In[3]:


unique_values_by_column(df, 1)


# In[4]:


num_columns = len(df.columns)
df = filter_column_uniques(df, 1)
print(f"Removed {num_columns - len(df.columns)} columns that had only one unique value")


# In[5]:


num_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {num_rows - df.shape[0]} repeated rows")


# ## Outliers and Noise

# ### Removing Noise

# In[6]:


print(df['award'].unique())

df = df[df.award != 'Kim Perrot Sportsmanship']
df = df[df.award != 'Kim Perrot Sportsmanship Award']


# In[7]:


df1 = df[df.award == 'Coach of the Year']
df = df[df.award != 'Coach of the Year']


# In[8]:


print(f"Removed {init_num_rows - df.shape[0]} ({ round((init_num_rows - df.shape[0]) / init_num_rows  * 100, 1)}%) rows.")
df['year'] = df['year'].apply(lambda x: x + 1)
df = df[(df['year'] <= 10) & (df['year'] >= 2)]


# ## Save Dataset

# In[9]:


if not os.path.exists('../data/clean'):
    os.makedirs('../data/clean')

df.to_csv('../data/clean/awards_players.csv', index=False)

df1.rename(columns={'playerID': 'coachID'}, inplace=True)
df1.to_csv('../data/clean/awards_coaches.csv', index=False)

