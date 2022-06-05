#!/usr/bin/env python
# coding: utf-8

# ### Tweet Analysis Notebook
# 
# In this project, we will analyze Russian troll tweet data from the dataset provided by [fivethirtyeight](https://github.com/fivethirtyeight/russian-troll-tweets/).  
# [Some analysis](http://pwarren.people.clemson.edu/Linvill_Warren_TrollFactory.pdf) of this data has already been completed and published by Linvill and Warren of Clemson university.
# 
# In this analysis, we will be trying to answer one primary question:  
# **What characteristics of troll accounts and their tweets make them successful?**
# 
# In this case, we are equating success to the number of followers the troll account has, as a higher follower account will allow the troll to spread their message to a greater number of individuals. 
# 
# As we answer this question, we will be answering some secondary questions that will help guide our answer, such as:
# * Which account category is most successful?
# * What are some specific characteristics of the most successful account category?
# * What type of language are the top troll accounts using?

# ### Setup (run once):

# This setup will get us to a clean DataFrame with which we will do further analysis and create visualizations with.  
# This process only needs to be run all the way through one time in order to save a DataFrame with the data we need.

# We will start off with some data cleaning and preparation.

# In[ ]:


import numpy as np
import pandas as pd
from textblob import TextBlob


# Note that this CSV is not in this repository due to its size (1 GB+), but it is available from the [fivethirtyeight repository](https://github.com/fivethirtyeight/russian-troll-tweets/). 

# In[ ]:


df = pd.read_csv('all_tweets.csv', low_memory=False)


# In[ ]:


df.info()


# We can see that there are just under 3 million rows and 15 columns.

# We won't be making use of the 'harvested_date' or 'new_june_2018' columns, so we will drop them.

# In[ ]:


df.drop(columns=['harvested_date', 'new_june_2018'], inplace=True)


# In[ ]:


df.head(3)


# We can see that we get we get the actual tweet text in the 'content' column, as well as some other Twitter information, such as author, following, etc. 
# 
# Interestingly, there is also an 'account_category' column, which identifies what type of activity the troll account is performing, which we will analyze later. 
# 
# It is also worth noting that **the 'retweet' column is not a count of retweets for any one tweet**, but rather a binary value of whether or not the tweet is a retweet made by the troll account (1 = retweet, 0 = not a retweet).

# In[ ]:


df.account_category.value_counts()


# We can see that 'NonEnglish' troll accounts weren't placed into a specific category, so we will drop those rows later. 
# 
# We can also see that there is quite the diversity with the account categories, though there seems to be some erroneous data where the listed account_category is 'account_category' in 17 rows.  
# We can filter those out to avoid working with bad data. 

# In[ ]:


df = df[df.account_category != 'account_category']


# In[ ]:


df.dtypes


# We should also change the data types of some of the columns we will do further analysis on so that they will be compatible with the functions that we use. 

# In[ ]:


df = df.astype({'external_author_id': np.float64,
                'following':          np.int64,
                'followers':          np.int64,
                'retweet':            np.int64,
                'content':            'str',
                'account_category':   'str'})


# Since we have tweet content, it may be interesting to take a look at a sentiment analysis of the data to gather more dimensions of the data.  
# Sentiment analysis is best done with English tweets, so we will remove any non-English tweets.

# In[ ]:


df = df[(df.region == 'United States') & (df.language == "English")]


# In[ ]:


df.info()


# This cuts the data down to about 1.86 million rows, which is still a lot of data.

# We can now run a sentiment analysis for each row using the [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) library.  
# This process will take a while to run, so we will save the results to a CSV file so that we don't have to do this again in the future.

# In[ ]:


df['polarity'], df['subjectivity'] = (df.content.apply(lambda x: TextBlob(x).sentiment.polarity)), \
                                     (df.content.apply(lambda x: TextBlob(x).sentiment.subjectivity))

# save to csv if coming back to the file later and to avoid running sentiment analysis again
df.to_csv('df_with_sentiment.csv', index=False)


# With that, the setup of our DataFrame is complete.

# ### Data Analysis

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# We will read in the previously saved CSV file to start to analyze the data.

# In[2]:


# df = pd.read_csv('df_with_sentiment.csv')
df = pd.read_csv('C:/Users/fahmi/Documents/Portfolio/Large Files/df_with_sentiment.csv')


# We may want to consider looking only at tweets that have a non-zero polarity in the future, but this is not done in this analysis.

# In[3]:


# df = df[(df_eng.polarity != 0)   # filter out tweets with no polarity, not used in this analysis


# In[4]:


df.account_category.value_counts()


# We can see that there are still some 'NonEnglish' accounts, but their content is in English.

# In[5]:


df[df.account_category == 'NonEnglish']


# More importantly, we might want to dial down the number of account categories to analyze in order to get a clear understanding of the data.  
# We will focus on the account categories that were analyzed in the [Linvill and Warren paper](http://pwarren.people.clemson.edu/Linvill_Warren_TrollFactory.pdf).

# In[6]:


# filter out account categories not analysed in the Linvill & Warren paper
df_plot = df.loc[(df.account_category != 'NonEnglish')  & \
                 (df.account_category != 'Unknown')     & \
                 (df.account_category != 'Commercial')]   \
            .copy().reset_index(drop=True)


# In[7]:


df_plot.info()


# We are still left with just under 1.75 million tweets. 

# Let's take a quick look at the distributions of followers.

# **Plot 0:** Distribution of Tweet Count by Number of Followers

# In[8]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df_plot.followers, bins=100, color='white', edgecolor='black', linewidth=1.25)
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Number of Followers', fontsize=16)
ax.set_ylabel('Tweet Count', labelpad=10, fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.xlim(0, 50000)
# plt.savefig('tweet_follower_distr.png', bbox_inches='tight', facecolor='w', dpi=200)
# plt.savefig('tweet_follower_distr.png', bbox_inches='tight', facecolor='w', dpi=800)
plt.show()


# We can see that the vast majority of tweets were made by trolls with a follower count of less than 10,000.  
# This reveals to us that most of the tweets were sent out by trolls with a small number of followers, and therefore little success. 

# In order to analyze success in this analysis, we will be looking at the number of followers of the troll account.  
# In this case, we are going to set 10,000 followers as the threshold for success.
# 
# We will also group the tweets by 'external_author_id' in order to avoid repeat counts of the same account.

# In[9]:


MIN_FOLLOWERS = 10000           # somewhat arbitrary minimum follower count, 10,000 in this case to only analyse successful troll accounts
MAX_FOLLOWERS = float('inf')    # maximum number of followers to include in the analysis, may be specified later

# group by external_author_id in order to avoid repeat analysis of the same account
df_plot = df_plot.loc[(df_plot.followers > MIN_FOLLOWERS) & (df_plot.followers < MAX_FOLLOWERS)] \
                 .groupby('external_author_id').agg({'account_category': 'first',
                                                     'following':        np.mean,
                                                     'followers':        np.mean,
                                                     'updates':          np.mean,
                                                     'retweet':          np.mean,
                                                     'polarity':         np.mean,
                                                     'subjectivity':     np.mean}).reset_index()


# In[10]:


df_plot.info()


# This leaves us with 58 individual successful troll accounts. 

# In[11]:


df_plot.head(3)


# ### What characteristics of troll accounts and their tweets make them successful?
# 
# We can now analyze what is making these accounts successful.
# 
# We can create a correlation matrix to see how the number of followers and other characteristics of the account affect one another. 

# In[12]:


# columns to not be included in the correlation matrix
col_to_drop = ['external_author_id', 'updates']

df_corr = df_plot.drop(columns=col_to_drop).copy().corr()
mask = np.tril(np.ones_like(df_corr, dtype=bool))


# In[13]:


# Function used with PairGrid object
def hide_current_axis(*args, **kwargs):
    """Hides the current axis"""
    plt.gca().set_visible(False)


# **Plot (1):** Relationships Between Different Aspects of Successful Russian Troll Accounts and Their Tweets

# In[34]:


# create figure and set default font size
fig = plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 14})

# create pairgrid object
pg = sns.PairGrid(df_plot.drop(columns=col_to_drop), hue='account_category', diag_sharey=False)

# create new axis for heatmap and colorbar
(xmin, _), (_, ymax) = pg.axes[0, 0].get_position().get_points()
(_, ymin), (xmax, _) = pg.axes[-1, -1].get_position().get_points()
ax1 = pg.fig.add_axes([xmin - 0.052, ymin + 0.01, xmax - xmin + 0.003, ymax - ymin + 0.003], facecolor='none')
ax2 = pg.fig.add_axes([0.91, 0.275, 0.04, 0.678], facecolor='none')

# create heatmap labels and colormap
ax1_labels = ['', 'Followers', 'Retweet', 'Polarity', 'Subjectivity']
ax2_labels = ['Following', 'Followers', 'Retweet', 'Polarity', '']
cmap = sns.diverging_palette(0, 210, 100, 60, as_cmap=True)

# create and edit heatmap and colorbar
hm = sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=0.75, vmin=-0.25, square=True,
                 linewidths=0, annot=True, annot_kws={'size': 22}, ax=ax1,
                 cbar=True, cbar_ax=ax2,
                 yticklabels=ax2_labels, xticklabels=ax1_labels)

hm.tick_params(left=False, right=False, top=False, bottom=False,
               labelleft=False, labelright=True, labeltop=True, labelbottom=False,
               labelsize=16)
hm.set_yticklabels(hm.get_yticklabels(), va='center')

cbar = hm.collections[0].colorbar
cbar.set_ticks([-0.25, 0, 0.25, 0.5, 0.75])
cbar.ax.tick_params(labelsize=16)
cbar.outline.set_linewidth(0)
cbar.outline.set_edgecolor('white')

# configure pairgrid object
pg.map_upper(hide_current_axis)
pg.map_diag(sns.kdeplot, legend=False, shade=True, alpha=0.5, linewidth=1, thresh=0)
pg.map_lower(sns.scatterplot, alpha=0.5)

# # work in progress to make the diagonal into boxplots:
# pg.map_diag(hide_current_axis)
# cats = ['following', 'followers', 'retweet', 'polarity', 'subjectivity']
# for idx, ax in enumerate(pg.axes.flat[0:25:6]):
#     # sns.boxplot(data=df_plot, x=cats[idx], y='account_category', ax=ax, palette='tab10', width=0.5)
#     # ax.set_xticklabels([])
#     # ax.set_yticklabels([])
#     ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
#     sns.boxplot(data=df_plot, x='followers', y='account_category', orient='h', 
#                 palette='tab10', width=0.5, ax=ax,
#                 medianprops={'linewidth': 2.25,
#                             'color': 'k',
#                             'alpha': 1},
#                 flierprops={'marker': 'o', 
#                             'markeredgecolor': 'black',
#                             'markersize': 3, 
#                             'alpha': 0.5})
#     # ax.set_xticklabels([])
#     # ax.set_yticklabels([])
#     # ax.xaxis.set_tick_params(length=0)
#     # ax.yaxis.set_tick_params(length=0)
# pg.map_diag(hide_current_axis)
# for idx, ax in enumerate(pg.axes.flat[0:25:6]):
#     print(ax.get_position().get_points())

# add a legend and legend title
pg.add_legend(title='', adjust_subtitles=True, bbox_to_anchor=(0.53, -0.119, .5, .5),
              fontsize=14)
plt.text(.56, 9.5, 'Account Category    ', fontsize=18, weight='medium')

# add linewidths to the heatmap manually
x_pos = [0.39, -0.06, -0.512, -0.962]
y_pos = [65.464, 52.6, 39.81, 27.05]
line_count = [23, 17, 12, 7]
for i in range(4):
    plt.text(x_pos[i], 64.9, '_' * line_count[i], fontsize=60, weight='heavy', 
             c='white', va='top', rotation=90)
    plt.text(0.55, y_pos[i], '_' * line_count[i], fontsize=60, weight='heavy', 
             c='white', ha='right')

# configure and edit labels
label_pads = [31.5, 10, 23, 23, 31.5]
axis_labels = ['Following', 'Followers', 'Retweet', 'Polarity', 'Subjectivity']

for idx, ax in enumerate(pg.axes.flat[0:21:5]):
    ax.set_ylabel(axis_labels[idx], labelpad=label_pads[idx], fontsize=16)

for idx, ax in enumerate(pg.axes.flat[20:25]):
    ax.set_xlabel(axis_labels[idx], fontsize=16)

for ax in pg.axes.flat[1:21:5]:
    ax.set_xticks([0, 50000])

plt.suptitle('Relationships Between Different Aspects of Successful Russian Troll Accounts and Their Tweets',
             fontsize=22, y=1.038, ha='center', va='center')

# plt.savefig('tweet_characteristics.png', bbox_inches='tight', facecolor='w', dpi=200)
# plt.savefig('tweet_characteristics.png', bbox_inches='tight', facecolor='w', dpi=800)

plt.show()


# This is a very rich plot and it reveals to us many aspects of the successful troll accounts and tweets. Starting off with the heatmap, we can analyze linear correlations between some of the features, and we can see how they relate to one another. We see that the strongest positive correlation is between 'Followers' and 'Following' which suggests that as the troll accounts followed more people, they would have more followers themselves, and therefore a larger reach. This principle is observable on many different social media, where bot accounts tend to send out follow requests en masse in the hopes that they will get more followers themselves, which establishes some sort of legitimacy and a greater reach.  
# With the scatterplots and KDE plots, we can analyze any non-linear relationships and any grouping that occurs by group category. For example, we can see that the 'NewsFeed' accounts tend to send fairly neutral tweets in terms of polarity and subjectivity, while the political accounts of 'LeftTrolls' and 'RightTrolls' tend to send tweets that are more subjective and polar. 

# ### Which account category is most successful?

# We can now take a look at which account categories specifically have the most successful accounts.  
# The dataframe is already prepared, we just have to plot it. 

# **Plot 2:** Boxplots showing distributions of followers by account category

# In[35]:


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

sns.boxplot(data=df_plot, x='followers', y='account_category', orient='h', 
            palette='tab10', width=0.5, 
            medianprops={'linewidth': 2.25,
                         'color': 'k',
                         'alpha': 1},
            flierprops={'marker': 'o', 
                        'markeredgecolor': 'black',
                        'markersize': 10, 
                        'alpha': 0.5})
                
ax.set_xlabel('Followers', labelpad=10, fontsize=16)
ax.set_ylabel('Account Category', labelpad=10, fontsize=16)
ax.xaxis.set_tick_params(length=0, labelsize=16)
ax.yaxis.set_tick_params(length=0)
ax.grid(axis='x', linestyle='--', linewidth=0.5, color='k', alpha=0.5)
ax.set_title('Distribution of Followers by Account Category', fontsize=20, x=0.4, y=1.05)

# plt.savefig('followers_by_category.png', bbox_inches='tight', facecolor='w', dpi=200)
# plt.savefig('followers_by_category.png', bbox_inches='tight', facecolor='w', dpi=800)

plt.show()


# We can see that troll accounts classified as 'RightTrolls' have the greatest number of successful accounts on average, followed by the 'LeftTrolls'. Of course, while on average this is true, there is a signficant range of followers for the 'RightTroll' accounts, and some outliers for the 'NewsFeed' accounts are quite large as well. In total, this may imply that the 'RightTrolls' were the most successful accounts on average. 

# ### What are some specific characteristics of the most successful account category?

# First we isolate the DataFrame that was grouped by troll account to just the RightTroll accounts. 

# In[16]:


df_right = df_plot[df_plot['account_category'] == 'RightTroll'].copy().reset_index(drop=True)


# We can then create a correlation matrix and mask for this new DataFrame.

# In[17]:


df_right_corr = df_right.drop(columns=col_to_drop).copy().corr()
mask_right = np.triu(np.triu(df_right_corr)[1:, :-1])


# **Plot 3:** Heatmap showing correlations between characteristics of successful RightTroll accounts

# In[37]:


plt.figure(figsize=(16, 12))

cmap = sns.diverging_palette(0, 210, 100, 60, as_cmap=True)

hm = sns.heatmap(df_right_corr.iloc[1:,:-1], mask=mask_right, cmap=cmap, square=True, fmt='.2f',
                 linewidths=5, annot=True, cbar=True, vmin=-1, vmax=1, annot_kws={'size': 22, 'weight': 'bold'})
                
ax = plt.gca()
ax.xaxis.set_tick_params(length=0)
ax.yaxis.set_tick_params(length=0)
ax.set_xticklabels(labels=[x.get_text().title() for x in ax.get_xticklabels()], fontsize=16)
ax.set_yticklabels(labels=[y.get_text().title() for y in ax.get_yticklabels()], fontsize=16)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)

plt.text(x=3.14, y=1.5,
         s='Correlations Between\nDifferent Aspects\nof RightTroll Accounts',
         fontsize='22', ha='center', va='center', weight='bold')

# plt.savefig('right_heatmap.png', bbox_inches='tight', facecolor='w', dpi=200)
# plt.savefig('right_heatmap.png', bbox_inches='tight', facecolor='w', dpi=800)

plt.show()


# We can see that the 'RightTrolls' have an even greater positive correlation between 'Following' and 'Followers', strongly implying that there is a linear relationship between the two variables. We can also observe that 'Polarity' seems to have a neutral correlation with the success of the 'RightTrolls', while 'Subjectivity' actually has a moderate negative correlation with the follower count. 

# ### What type of language are the top troll accounts using?

# Starting off with some more imports

# In[19]:


from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from imageio import imread


# In order to analyze the language of only the top troll accounts, we will need to isolate just the tweets made by those accounts. 

# In[20]:


df_words = df.content[df.followers > 10000].copy().reset_index(drop=True)


# In[21]:


df_words.shape


# We can see that this leaves us with just under 600,000 tweets.

# In[22]:


# df_words.sort_values(by='publish_date', inplace=True)


# Next we can convert the tweets to a long string of text so that we can tokenize the tweets.

# In[23]:


all_words = df_words.str.cat(sep=' ').lower()


# In[24]:


tt = TweetTokenizer()


# In[25]:


tweet_tokens = TweetTokenizer().tokenize(all_words)


# In[26]:


len(tweet_tokens)


# This leaves us with just about 750,000 'tokens'. 

# We can update the default STOPWORDS list with some more stop words to remove error words and words that are not useful to analyze. 

# In[27]:


STOPWORDS.update(['>>', '<<', 'new', 'say', 'says', 'make', 'year', 'will'])


# Then remove any words that are just links or hashtags.

# In[28]:


only_words = [token for token in tweet_tokens if not token.startswith('#') \
              and not token.startswith('@') and not token.startswith('http')]


# And once again turn the list into a long string. 

# In[29]:


only_words_str = " ".join(only_words)


# We can read in a mask for our wordcloud to make the wordcloud conform to a twitter-like shape and then run the WordCloud function. 

# In[30]:


twitter_mask = imread('twitter_mask.png')


# In[31]:


wordcloud = WordCloud(colormap='viridis', mask=twitter_mask, contour_width=2, contour_color='blue',
                      background_color='white', min_word_length=2, stopwords=STOPWORDS).generate(only_words_str)


# **Plot (4):** Wordcloud of the top words used by successful troll accounts. 

# In[38]:


plt.figure(figsize=(16, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# plt.savefig('tweet_wordcloud.png', bbox_inches='tight', facecolor='w', dpi=200)
# plt.savefig('tweet_wordcloud.png', bbox_inches='tight', facecolor='w', dpi=800)

plt.show()


# We can see that a big focus of tweets sent out or retweeted by the successful troll accounts was political. With 'trump' being the single most common word and 'obama' being up there too, there was a big focus on US presidents in their tweets. Beyond that, some divisive language was very common, such as 'death', 'murder', 'attack', 'shooting', and 'killed'. In general, this language could potentially serve to bring about quarreling and incite fear among those who read these tweets.  
