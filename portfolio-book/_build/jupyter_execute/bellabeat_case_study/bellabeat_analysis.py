#!/usr/bin/env python
# coding: utf-8

# ```{image} bellabeat_banner.png
# :align: left
# :width: 100%
# ```

# # Bellabeat Case Study
# <!-- # <span style="color:#FA8072">Bellabeat Case Study</span> -->
# <!-- !jupyter nbconvert --to markdown README.ipynb -->
# 
# Fahmi I.  
# 
# June 3rd, 2022
# 
# ---

# <!-- ## <span style="color:#FA8072">Table of Contents</span>
# 
# * [1. Introduction](#Introduction)
# * [2. Ask Phase](#Ask-Phase)
# * [3. Prepare Phase](#Prepare-Phase)
#     * [3.1 Business Task](#Business-Task)
# * [4. Process Phase](#Process-Phase)
# * [5. Analyze and Share Phase](#Analyze-and-Share-Phase)
# * [6. Act Phase](#Act-Phase)
# 
# --- -->

# ## Introduction:
# <!-- ## <a name='Introduction'></a> <span style="color:#FA8072">1. Introduction:</span> -->
# 
# introduction text here

# ---

# ## Ask-Phase:
# <!-- ## <a name='Ask-Phase'></a><span style="color:#FA8072">2. Ask Phase:</span> -->
# 
# ask phase text here

# ---

# ## Prepare Phase:
# <!-- <a name='Prepare-Phase'></a><h3 style="color:#FA8072">3. Prepare Phase:</h3> -->

# ---

# ### Business Task:
# <!-- #### <a name='Business-Task'></a>3.1: Business Task -->

# ## Process Phase:
# <!-- <a name='Process-Phase'></a><h3 style="color:#FA8072">4. Process Phase:</h3> -->

# I will be using Python to process, analyze, and visualize the data.

# ### Import Libraries:
# <!-- #### <a name='Import-Libraries'></a>4.1: Import Libraries -->

# Starting off by importing the libraries we will need, all of which are standard for data analysis. 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# We can then take a look at which data files we have available. 

# In[3]:


os.listdir(r'C:\Users\fahmi\Documents\Portfolio\Large Files\Data_Fitbase')


# In[4]:


original_minute_sleep = pd.read_csv(
    r'C:\Users\fahmi\Documents\Portfolio\Large Files\Data_Fitbase/minuteSleep_merged.csv', 
    parse_dates=['date'], infer_datetime_format=False
)


# In[5]:


df_sleep = original_minute_sleep.copy().drop(columns='logId')\
                                .rename(columns={'Id': 'id', 'date': 'date_time'})


# In[6]:


#todo
df_sleep.isna().sum()


# In[7]:


def df_info(df: pd.DataFrame, name: str):
    '''Prints columns and shape of a dataframe'''
    print(f'{name}:\n\tColumns: {list(df.columns)}\n\tShape: {df.shape}')


# In[8]:


df_info(df_sleep, 'df_sleep')


# In[9]:


df_sleep.head(3)


# ---

# ## Analyze and Share Phase:
# <!-- <a name='Analyze-and-Share-Phase'></a><h3 style="color:#FA8072">5. Analyze and Share Phase:</h3> -->

# ---

# ## Act Phase:
# <!-- <a name='Act-Phase'></a><h3 style="color:#FA8072">6. Act Phase:</h3> -->

# ---

# ### Sleep Analysis

# In[10]:


df_sleep['day_of_week'] = df_sleep['date_time'].dt.day_name()
df_sleep['date'] = df_sleep['date_time'].dt.date


# In[11]:


df_sleep.head(3)


# In[12]:


df_sleep['time_diff'] = df_sleep.groupby('id')['date_time'].diff()


# In[13]:


df_sleep.head(5)


# In[14]:


df_sleep.time_diff.value_counts()


# In[15]:


df_sleep = df_sleep[df_sleep.time_diff == np.timedelta64(1, 'm')]


# In[16]:


df_info(df_sleep, 'df_sleep')


# In[17]:


def plot_sleep(df_sleep: pd.DataFrame, id: int, repeat_ylabel: bool = True):
    '''
    Plots sleep data for a given id
    
        Parameters:
            df_sleep (pd.DataFrame): Dataframe with sleep data to plot
            id (int):                ID of the user to be analyzed
            repeat_ylabel (bool):    Whether to repeat y-axis label
    '''

    WEEK_ORDER= ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    SLEEP_STATE = ['Asleep', 'Restless', 'Awake in Bed']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 8), facecolor='w')
    df_id = df_sleep.loc[df_sleep.id == id]
    
    for idx, ax in enumerate(axes):

        minutes = df_id.loc[df_id.value == idx + 1].groupby('day_of_week')['value'].sum()
        days = df_id.groupby('date').day_of_week.value_counts()

        days_count = {}
        for i in range(len(days)):
            days_count[days.index[i][1]] = days_count.get(days.index[i][1], 0) + 1

        for i in minutes.index:
            minutes[i] /= days_count[i]

        sns.barplot(
            orient='h', x=minutes.values, y=minutes.index, 
            palette='icefire', order=WEEK_ORDER, alpha=0.8, ax=ax
        )

        if not repeat_ylabel and idx > 0:
            ax.set_yticks([])
        ax.yaxis.set_tick_params(length=0)
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(xlabel=f'Minutes {SLEEP_STATE[idx]}', fontsize=14, labelpad=10)

        ax.set_title(f'Minutes {SLEEP_STATE[idx]} by Day of Week',
                    fontsize=16, loc='left', pad=20)

        ax.grid(axis='x', linestyle='--', alpha=0.8)
        sns.despine()
    
    fig.suptitle('Average Minutes Spent in Each Sleep State by Day of the Week', 
                 fontsize=22, x=0.123, y=1.05, ha='left')


# In[18]:


ids = df_sleep.id.unique()
len(ids)


# In[19]:


plot_sleep(df_sleep, id=ids[0])


# In[ ]:




