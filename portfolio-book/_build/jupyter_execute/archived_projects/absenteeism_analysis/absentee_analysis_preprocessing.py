#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Notebook

# ### Absenteeism at Work Analysis
# 
# Notebook 1/3 
# 
# In this project, we will analyze the absenteeism data from the UCI Machine Learning Repository [found here](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work).  
# The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.  
# 
# We will define absenteeism as follows:  
# * absence from work during normal working hours resulting in temporary incapacity to execute regular working activity
# 
# In this analysis, we will be trying to answer one primary question:  
# **Which factors are most indicative of excessive absenteeism from work and can we predict absenteeism by identifying these factors?**
# 
# While answering this question, we will also be making use of logistic regression in order to create a model which can predict absenteeism. 
# 
# By answering this question and creating our model, we will be able to predict the absenteeism rate for any given employee, which will help us maximize the quantity and quality of work done. 
# 
# This notebook will be dedicated to preprocessing the data to be later used to create a logistic regression model.

# ### Preprocessing the data

# Starting off with importing the pandas library

# In[1]:


import pandas as pd


# Then reading in the raw CSV data

# In[2]:


raw_data = pd.read_csv('Absenteeism-data.csv')


# In[3]:


raw_data.info()


# We can see that there are 11 columns and 700 rows.  
# The columns are described in the image below:

# ![](data_columns.png)

# In[4]:


raw_data.head(3)


# We can make a copy of the raw dataframe so that we can manipulate it and not lose the original data.

# In[5]:


df = raw_data.copy()


# The ID column is not very useful for our analysis or for trying to predict absenteeism time, so we can drop it.

# In[6]:


df.drop(columns=['ID'], inplace=True)


# In[7]:


df.head(3)


# ### Manipulating 'Reason for Absence':

# As we saw above, there were 28 different reasons identified for absenteeism in this dataset.  
# Instead of trying to analyze each of these reasons individually, it may make more sense to group them into a few groups so that we can attain a greater understanding of the dataset and remove confusion. 

# In[8]:


df['Reason for Absence'].unique()


# In[9]:


len(df['Reason for Absence'].unique())


# In[10]:


sorted(df['Reason for Absence'].unique())


# We can see that of the 28 reasons, number 20 was never used. 

# We can create a dummies dataset for the reasons for absenteeism in order to categorize them. 

# In[11]:


reason_columns = pd.get_dummies(df['Reason for Absence'])


# In[12]:


reason_columns


# We can check to make sure that each row only has one reason associated with it. 

# In[13]:


reason_columns['check'] = reason_columns.sum(axis=1)


# In[14]:


reason_columns.check.value_counts()


# This verifies that there is only ever 1 reason for absenteeism, and there is never no reason or multiple reasons. 

# We have no more use for the 'check' column, so we can now drop it, and we can also recreate the dummies dataframe with the first column, or reason 0, dropped. 

# In[15]:


reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)


# In[16]:


reason_columns


# In[17]:


df.columns.values


# In[18]:


reason_columns.columns.values


# We have already separated out the 'Reason for Absence' column from df already, so we can drop the column. 

# In[19]:


df.drop(columns=['Reason for Absence'], inplace=True)


# We can split the absentee reasons into a few key groups in order to analyze them more easily.

# **Absentee Reason Categories:**  
# * 1-14: related to various diseases
# * 15-17: related to pregnancy or giving birth
# * 18-21: related to poisonings or symptoms not otherwise classified
# * 22-28: 'light reasons' such as a doctor's appointment

# ![](columns_grouped.png)

# We can split up the reasons dataframe according to the reason category. 

# In[20]:


reason_columns.loc[:, :14]


# Finding the max by row for each of these categories will return a series with a 1 representing that the absentee was due to a reason in that category, and a 0 representing that the absentee was not due to a reason in that category.

# In[21]:


reason_columns.loc[:, :14].max(axis=1)


# We can expand this to each of the four categories described above. 

# In[22]:


reason_type_1 = reason_columns.loc[:, :14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)


# We can now concatenate these series into our df. 

# In[23]:


df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)


# In[24]:


df.head(3)


# We have our absentee reasons grouped into four categories in the df, and now we can add meaningful names to the added columns. 

# In[25]:


column_names = df.columns.values.tolist()[:-4] + ['Reason_' + str(i) for i in range(1, 5)]


# In[26]:


column_names


# In[27]:


df.columns = column_names


# In[28]:


df.head(3)


# We can also move around the columns to place the reasons at the start of the df, as the reason was originially at the start. 

# In[29]:


column_names_reordered = ['Reason_' + str(i) for i in range(1, 5)] + df.columns.values.tolist()[:-4]


# In[30]:


column_names_reordered


# In[31]:


df = df[column_names_reordered]


# In[32]:


df.head(3)


# ### Checkpoint 1:

# This may be a good spot to create a checkpoint so that we can make changes without having to rerun the entire notebook.

# In[33]:


df_reason_mod = df.copy()


# ### Manipulating 'Date':

# Since we have time data, we can also analyze whether there are specific months or days of the week that are more likely to result in absenteeism. 

# In[34]:


type(df_reason_mod.Date[0])


# We can see that the dates are currently in the form of strings.  
# Since these are dates, we can convert them into timestamps in order to extract temporal information more easily

# In[35]:


df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')


# In[36]:


df_reason_mod.head(3)


# In[37]:


type(df_reason_mod.Date[0])


# We can see that this retained the format of the dates, while also updating their type. 

# In[38]:


df_reason_mod.Date[0]


# In order to analyze whether there are certain months during which absenteeism rates are higher, we can create a new column specifically for the month of the data points
# 
# By accessing the .month property of the timestamp, we can see that the month is now a number between 1 and 12 and place it in a new column.

# In[39]:


df_reason_mod.Date[0].month


# In[40]:


df_reason_mod['Month Value'] = df_reason_mod.Date.apply(lambda x: x.month)


# We can also take a look at creating a column with the day of the week to see if absenteeism increases on certain days of the week.  
# Instead of using words, we can assign values to the days of the week:
# * Monday = 0
# * Tuesday = 1  
# ...
# * Sunday = 6

# In[41]:


df_reason_mod.Date[0]


# Similar to what we did for month values above, we can do the same for the day of the week by accessing the .weekday attribute to add a new column to the dataframe.

# In[42]:


df_reason_mod.Date[0].weekday()


# In[43]:


df_reason_mod['Day of the Week'] = df_reason_mod.Date.apply(lambda x: x.weekday())


# In[44]:


df_reason_mod.head(3)


# We can reorder the columns to move these time columns to where 'Date' is while also dropping the now redundant 'Date' column.

# In[45]:


df_reason_mod.drop(columns='Date', inplace=True)


# In[46]:


df_reason_mod.columns.values


# In[47]:


df_reason_mod = df_reason_mod[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
                               'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
                               'Daily Work Load Average', 'Body Mass Index', 'Education',
                               'Children', 'Pets', 'Absenteeism Time in Hours']]


# In[48]:


df_reason_mod.head(3)


# ### Checkpoint 2:

# We can now create another checkpoint with completed reason and date columns. 

# In[49]:


df_reason_date_mod = df_reason_mod.copy()


# We can now move on to some of the other columns in the dataframe.

# In[50]:


df_reason_date_mod.info()


# * Transportation Expense: monthly transportation expenses of an individual measured in dollars ($USD)
# * Distance to Work: the kilometers an individual has to travel from home to work
# * Age: age of an individual rounded down
# * Daily Work Load Average: the number of hours an individual works per day on average
# * Body Mass Index: the weight an individual is in kilograms divided by the square of their height in meters, designates if the person is overweight, underweight, obese, etc.
# * Education: nominal data of the level of education an individual has completed
#     * 1: high school education
#     * 2: graduate level education
#     * 3: post graduate education
#     * 4: a master or a doctor
# * Children: the number of children an individual has
# * Pets: the number of pets an individual has
# * Absenteeism Time in Hours: the number of hours an individual has been absent from work

# In[51]:


df_reason_date_mod.Education.value_counts()


# We can see that the majority of individuals only have a highschool education, while the rest have more education. Because there are few data points with a higher level of education, the disparity between those points has less meaning. As a result, we can combine education higher than highschool into a single category.

# In[52]:


df_reason_date_mod.Education = df_reason_date_mod.Education.map({1:0, 2:1, 3:1, 4:1})


# In[53]:


df_reason_date_mod.Education.value_counts()


# We can now save this preprocssed dataframe to a csv file and move on to analyzing it. 

# In[54]:


df_reason_date_mod.to_csv('absenteeism_preprocessed.csv', index=False)

