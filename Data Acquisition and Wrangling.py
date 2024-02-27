#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_excel("dataset_1.xlsx")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.head(2)


# In[7]:


df.tail(2)


# In[8]:


df.tail()


# In[9]:


df.info()


# Loc & Iloc

# In[10]:


df.loc[:,:]


# In[11]:


df.iloc[1:4:,2:5]


# In[12]:


df.loc[1:4,'yr':'weekday']


# In[13]:


df.shape


# In[107]:


cols=df.columns
cols


# # sorting & Filter

# In[15]:


df.head()


# In[16]:


df['temp'].sort_values(ascending=False)


# In[17]:


df['temp'].sort_values(ascending=True)


# In[18]:


df.sort_values(by='temp',ascending=True)
df.head()


# In[19]:


df['temp'].unique()


# In[20]:


df[['instant','season','yr']]


# # Filter Dataframe

# In[21]:


df['temp']==.2


# In[22]:


df.head()


# # sampling

# In[23]:


df_rand=df.sample(n=10)


# In[24]:


df_rand


# In[25]:


df_rand=df.sample(frac=0.1)
df_rand


# In[26]:


df.sample(frac=.2)


# # Filter & Array

# In[27]:


df.filter(like='a')


# In[28]:


df.filter(like='an')


# In[29]:


#regex=regular expression
df.filter(regex='an')


# # Drop

# In[30]:


df.drop(['season'],axis=1)


# In[31]:


df.drop([1,3,5],axis=0)


# # Duplicated

# In[32]:


df.duplicated()


# In[33]:


df.duplicated().sum()


# In[34]:


#remove duplicates
df.drop_duplicates(inplace=True)


# # Missing Values

# In[35]:


df.isnull().sum()


# # Groupby

# In[36]:


df.groupby('yr').first()


# In[37]:


df.groupby(['yr','weekday']).first()


# In[38]:


df.groupby(['yr'])[['holiday','temp']].mean()


# # Concat

# In[39]:


series_1=pd.Series([1,2,43,5,6])
series_2=pd.Series([4,6,7,8,6])


# In[40]:


pd.concat([series_1,series_2])


# In[41]:


df_2=pd.read_excel('dataset_2.xlsx')
df_2


# In[42]:


df_2.head(2)


# In[43]:


df_2.tail(3)


# In[44]:


df_2.info()


# In[45]:


df_2.columns


# In[46]:


df_2[(df_2['hum']==0.93 )& (df_2['instant']==608)]


# In[47]:


df_2[(df_2['hum']==0.93 )| (df_2['instant']==608)]


# In[48]:


combine_data=pd.merge(df,df_2,on='instant',how='outer')
combine_data


# In[49]:


df.shape


# In[50]:


df_2.shape


# In[51]:


df.head(1)


# In[52]:


df_2.head(1)


# In[53]:


df_3=pd.read_excel('dataset_3.xlsx')
df_3


# In[54]:


df_3.shape


# In[55]:


df_3.columns


# In[56]:


df_3.info()


# In[57]:


df_3['holiday'].replace({True:1,False:0},inplace=True)


# In[58]:


df_3


# In[59]:


concat=pd.concat([combine_data,df_3])


# In[60]:


concat


# In[61]:


cc=pd.concat([df,df_2])


# In[62]:


cc


# In[63]:


merge=pd.merge(combine_data,df_3,on='instant',how='inner')


# In[64]:


merge


# In[65]:


concat.head(2)


# In[66]:


concat.dtypes


# In[67]:


#convert the columns to numeric columns
cols=['temp','cnt','casual']
concat[cols]=concat[cols].apply(pd.to_numeric,args=('coerce',))


# In[68]:


concat.head(2)


# In[69]:


#sum of missing values in each columns
concat.isnull().sum()


# In[70]:


#convert "?" with np.nan
concat.replace('?',np.nan,inplace=True)


# In[71]:


concat['Unnamed: 0'].value_counts()


# In[72]:


concat["Unnamed: 0"].isnull().sum()


# In[73]:


concat["Unnamed: 0"].isnull()


# In[74]:


concat["Unnamed: 0"]


# In[75]:


concat.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[76]:


concat.head(2)


# # Central Tendency

# In[77]:


concat['casual'].mode()


# In[78]:


concat['casual'].value_counts().head()


# In[79]:


#median of any column
concat['casual'].median()


# In[80]:


#Average value
concat['casual'].mean()


# # Basic Imputation technique

# # Handling Missing Values

# In[81]:


concat['casual'].unique()


# In[82]:


concat['atemp'].value_counts()


# In[83]:


concat['atemp'].isnull().sum()


# In[84]:


concat['atemp'].replace(np.nan,concat['atemp'].mean(),inplace=True)


# In[85]:


concat['atemp'].isnull().sum()


# In[86]:


concat['atemp'].std


# In[87]:


concat.drop(['dteday'],axis=1,inplace=True)


# In[88]:


#Imputation by using statics(mean,mode,median) of each column with missing values


# In[89]:


from sklearn.impute import SimpleImputer
impute=SimpleImputer(strategy='median')
data_array=impute.fit_transform(concat)


# In[90]:


data_array


# In[91]:


concat


# In[92]:


concat.isnull().sum()


# In[93]:


concat.describe()


# # Outliers

# In[94]:


#q1=25%, q2=50%, q3=75%


# In[95]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[96]:


q1=concat['temp'].quantile(.25)
q3=concat['temp'].quantile(.75)


# In[97]:


def find_outliers(df,cols):
    q1=df[cols].quantile(.25)
    q3=df[cols].quantile(.75)
    iqr=q3-q1
    min_r=q1-1.5*iqr
    max_r=q3+1.5*iqr
    outliers_indices=df.index[(df[cols]<min_r)|(df[cols]>max_r)]
    return outliers_indices


# In[98]:


find_outliers(concat,'temp')


# ## Treating Outliers

# In[99]:


#scipy- use for all statistical function
from scipy import stats


# In[100]:


concat_num=concat.select_dtypes(include=['int64','float64'])


# In[101]:


z=np.abs(stats.zscore(concat_num))


# In[102]:


print(np.where((z>3)|(z< -3))[0])


# # Skewness & Correlation
# 

# In[103]:


concat.corr()


# In[105]:


import scipy 
from scipy.stats import skew


# In[112]:


skew(concat,axis=0,bias=True)


# In[113]:


skew(concat,axis=1,bias=True)

