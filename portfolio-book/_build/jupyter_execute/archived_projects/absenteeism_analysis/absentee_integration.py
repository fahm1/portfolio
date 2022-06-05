#!/usr/bin/env python
# coding: utf-8

# # Integration Notebook

# ### Absenteeism at Work Analysis
# 
# Notebook 3/3 
# 
# At this point, we have created a model that can predict absenteeism for any new individuals, and also a module that will take care of the preprocessing of our data.  
# 
# Now we can take a look at predicting new data and even the entire dataset. 

# Starting off by importing the absenteeism_module that we created in the previous notebook.  
# We will also import pymysql to connect to a database and config to import the credentials to the database. 
# This import will also import the libraries used in the module (pickle, numpy, pandas, and sklearn). 

# In[1]:


from absenteeism_module import *
import pymysql
import config


# In[2]:


df = pd.read_csv('absenteeism_new_data.csv', delimiter=',')


# In[3]:


df.info()


# We can initialize our model with the mode and scaler that we created in the previous notebook.

# In[4]:


model = AbsenteeismModel('model.pkl', 'scaler.pkl')


# Then we can preprocess our data. 

# In[5]:


model.load_and_clean_data('absenteeism_new_data.csv')


# Once our data is clean, we can use the model to predict absenteeism for new individuals.

# In[6]:


df_new_obs = model.predicted_outputs()


# In[7]:


df_new_obs.head(3)


# And just like that, we are successfully able to predict absenteeism for new individuals.

# We can combine this new dataset with the original dataset that we used to train our model while also predicting absenteeism for the original dataset as well. 

# In[8]:


model_old_data = AbsenteeismModel('model.pkl', 'scaler.pkl')
model_old_data.load_and_clean_data('Absenteeism-data.csv')
df_old_obs = model_old_data.predicted_outputs()


# In[9]:


all_data = pd.concat([df_new_obs, df_old_obs], axis=0).reset_index(drop=True)


# In[10]:


all_data.info()


# In[11]:


all_data.head(3)


# We can now import this data into a MySQL database.  
# Note that setting up the proper database and table in MySQL is done outside of this notebook but can be found in the 'sql_script.sql' file.
# Set up proper database and table in MySQL prior to continuing.

# Start of by creating a connection to the database. Note that the password is obfuscated in this notebook. 

# In[12]:


conn = pymysql.connect(user=config.sql_username, password=config.sql_password, database='predicted_outputs')


# Then we initialize our cursor.

# In[13]:


cursor = conn.cursor()


# In[14]:


cursor.execute('SELECT * FROM predicted_outputs')


# We can see that there is currently no data in the predicted_outputs table. 

# In[15]:


all_data.shape


# In[16]:


all_data.columns.values


# In[17]:


all_data.columns.values[7]


# We can access the individual values with the method shown below. 

# In[18]:


print(
    all_data[all_data.columns.values[7]][0], 
    all_data[all_data.columns.values[7]][1]
)


# With that, we can use a nested for loop to insert all of the values into the database. We will first create a string with our insert query. 

# In[19]:


insert_query = 'INSERT INTO predicted_outputs VALUES'

for i in range(all_data.shape[0]):
    insert_query += '('
    
    for j in range(all_data.shape[1]):
        insert_query += str(all_data[all_data.columns.values[j]][i]) + ', '
    
    insert_query = insert_query[:-2] + '), '

insert_query = insert_query[:-2] + ';'


# In[20]:


insert_query[:300]


# We can see that our insert query was correctly created.
# 
# We can now execute our insert query to our data to the database.

# In[21]:


cursor.execute(insert_query)


# In[22]:


cursor.execute('SELECT * FROM predicted_outputs')


# We can see that there our data was successfully inserted into the database.

# In[23]:


conn.commit()


# In[24]:


conn.close()


# With that, we can save our predicted dataframe to a CSV file and conduct further analysis and create visualiztions using Tableau. 

# In[25]:


all_data.to_csv('absenteeism_predictions.csv', index=False)


# In[26]:


pd.read_csv('absenteeism_predictions.csv').head(3)

