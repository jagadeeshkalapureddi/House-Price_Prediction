#!/usr/bin/env python
# coding: utf-8

# # --------------------------------@ HOUSE_PRICE DATA ANALYSIS @----------------------------------------------------------------------! LIN REGRESSION !----------------------------------------

# # IMPORT THE REQUIRED PACKAGES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### READ THE DATASET

# In[ ]:


HP = pd.read_csv('House_Price.csv')


# ### DATA UNDERSTANDING

# In[ ]:


HP.head()


# In[ ]:


HP.shape
print("The Data Frame having the Rows of '{}' and Columns of '{}'".format (HP.shape[0],HP.shape[1]))


# In[ ]:


HP.info()


# In[ ]:


HP.isnull().sum()


# ## Individual Variable Understanding.

# ## Dependent Variable.  :  price

# In[ ]:


print('Column_name : ' ,HP.iloc[:,0].name)
print('Type : ',HP.iloc[:,0].dtype)

print('Null_value_count: ',HP.iloc[:,0].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,0].skew())
HP.iloc[:,0].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,0], color = 'green')
plt.xlabel(HP.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,0].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,0], color = 'orange')
plt.xlabel(HP.iloc[:,0].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# #### The Outliers seen in above plots are acceptable.

# ## Independent Variables.

# # 1

# In[ ]:


print('Column_name : ' ,HP.iloc[:,1].name)
print('Type : ',HP.iloc[:,1].dtype)
print('Null_value_count: ',HP.iloc[:,1].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,1].skew())
HP.iloc[:,1].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,1], color = 'green')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,1], color = 'orange')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ### Combined plot

# In[ ]:


## Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Add a graph in each part
plt.suptitle('Box_Dist_Plot ' + HP.iloc[:,1].name, fontsize = 15)
sns.boxplot(HP.iloc[:,1], ax=ax_box, color = 'blue')
sns.distplot(HP.iloc[:,1], ax=ax_hist, color = 'orange')
plt.show()


# ## It is having outliers, then check for quantile ranges.

# In[ ]:


print('5% :', HP.iloc[:,1].quantile(0.05), '\n','95% :', HP.iloc[:,1].quantile(0.95))


# In[ ]:


import numpy as np
HP.iloc[:,1] = np.where(HP.iloc[:,1] > HP.iloc[:,1].quantile(0.95), HP.iloc[:,1].quantile(0.95) + 5, HP.iloc[:,1])
HP.iloc[:,1].describe()


# In[ ]:



print('Skewness :' ,HP.iloc[:,1].skew())
sns.boxplot(HP.iloc[:,1], color = 'green')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,1].name, fontsize = 25)

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'crime_rate', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ### Most of the data accumulated around the zero so we are going for log transformation.
# 
# ## Data transformation using Log. (For Removing the outliers)

# In[ ]:


HP.iloc[:,1] = np.log(HP.iloc[:,1] + 1)


# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'crime_rate', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# In[ ]:


print('Skewness :' ,HP.iloc[:,1].skew())
plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,1], color = 'green')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,1], color = 'orange')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in crime_rate variable.

# # 2

# In[ ]:


print('Type : ',HP.iloc[:,2].dtype)
print('Column_name : ' ,HP.iloc[:,2].name)

print('Null_value_count: ',HP.iloc[:,2].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,1].skew())
HP.iloc[:,2].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,2], color = 'green')
plt.xlabel(HP.iloc[:,2].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,2].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,2], color = 'orange')
plt.xlabel(HP.iloc[:,2].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,2].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in resid_area variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'resid_area', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 3

# In[ ]:


print('Type : ',HP.iloc[:,3].dtype)
print('Column_name : ' ,HP.iloc[:,3].name)

print('Null_value_count: ',HP.iloc[:,3].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,3].skew())
HP.iloc[:,3].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,3], color = 'green')
plt.xlabel(HP.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,3].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,3], color = 'orange')
plt.xlabel(HP.iloc[:,3].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,3].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in air_qual variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'air_qual', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 4

# In[ ]:


print('Type : ',HP.iloc[:,4].dtype)
print('Column_name : ' ,HP.iloc[:,4].name)

print('Null_value_count: ',HP.iloc[:,4].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,4].skew())
HP.iloc[:,4].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,4], color = 'green')
plt.xlabel(HP.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,4].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,4], color = 'orange')
plt.xlabel(HP.iloc[:,4].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,4].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in room_num variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'room_num', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 5

# In[ ]:


print('Type : ',HP.iloc[:,5].dtype)
print('Column_name : ' ,HP.iloc[:,5].name)

print('Null_value_count: ',HP.iloc[:,5].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,5].skew())
HP.iloc[:,5].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,5], color = 'green')
plt.xlabel(HP.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,5].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,5], color = 'orange')
plt.xlabel(HP.iloc[:,5].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,5].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in age variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'age', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 15
# 
# #  6,7,8,9  here we have four variables with relevant data so merge them into one variable.

# In[ ]:


print('Type : ',HP.iloc[:,6].dtype)
print('Column_name : ' ,HP.iloc[:,6].name)

print('Null_value_count: ',HP.iloc[:,6:10].isna().sum())


# In[ ]:


HP["avg_dist"]=HP[["dist1","dist2","dist3","dist4"]].mean(axis=1)
HP.drop(["dist1","dist2","dist3","dist4"], axis=1, inplace=True)


# In[ ]:


print('Column_Index_Number :',HP.columns.get_loc('avg_dist'))


# In[ ]:


print('Skewness: ', HP.iloc[:,15].skew())
HP.iloc[:,15].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,15], color = 'green')
plt.xlabel(HP.iloc[:,15].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,15].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,15], color = 'orange')
plt.xlabel(HP.iloc[:,15].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in avg_dist variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'avg_dist', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 6

# In[ ]:


print('Type : ',HP.iloc[:,6].dtype)
print('Column_name : ' ,HP.iloc[:,6].name)

print('Null_value_count: ',HP.iloc[:,6].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,6].skew())
HP.iloc[:,6].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,6], color = 'green')
plt.xlabel(HP.iloc[:,6].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,6].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,6], color = 'orange')
plt.xlabel(HP.iloc[:,6].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,6].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in teachers variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'teachers', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 7

# In[ ]:


print('Type : ',HP.iloc[:,7].dtype)
print('Column_name : ' ,HP.iloc[:,7].name)

print('Null_value_count: ',HP.iloc[:,7].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,7].skew())
HP.iloc[:,7].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,7], color = 'green')
plt.xlabel(HP.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,7].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,7], color = 'orange')
plt.xlabel(HP.iloc[:,7].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in poor_prop variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'poor_prop', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 8

# In[ ]:


print('Type : ',HP.iloc[:,8].dtype)
print('Column_name : ' ,HP.iloc[:,8].name)

print('Null_value_count: ',HP.iloc[:,8].isna().sum())


# In[ ]:


HP.iloc[:,8].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


HP.iloc[:,8].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,8].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)


# ## Convert the string to integer.

# In[ ]:


HP.iloc[:,8] = np.where(HP.iloc[:,8] == 'YES', 1, 0)
HP.iloc[:,8]


# # 9

# In[ ]:


print('Type : ',HP.iloc[:,9].dtype)
print('Column_name : ' ,HP.iloc[:,9].name)

print('Null_value_count: ',HP.iloc[:,9].isna().sum())


# ## Having Null Values.  So replace with its mean value.

# In[ ]:


HP.iloc[:,9] = np.where(HP.iloc[:,9].isnull() == True, np.mean(HP.iloc[:,9]), HP.iloc[:,9])
HP.iloc[:,9].isna().sum()


# In[ ]:


print('Skewness: ', HP.iloc[:,9].skew())
HP.iloc[:,9].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,9], color = 'green')
plt.xlabel(HP.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,9].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,9], color = 'orange')
plt.xlabel(HP.iloc[:,9].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,9].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in n_hos_beds variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'n_hos_beds', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 10

# In[ ]:


print('Type : ',HP.iloc[:,10].dtype)
print('Column_name : ' ,HP.iloc[:,10].name)

print('Null_value_count: ',HP.iloc[:,10].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,10].skew())
HP.iloc[:,10].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,10], color = 'green')
plt.xlabel(HP.iloc[:,10].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,10].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,10], color = 'orange')
plt.xlabel(HP.iloc[:,10].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,10].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ### Combined plot

# In[ ]:


## Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Add a graph in each part
plt.suptitle('Box_Dist_Plot ' + HP.iloc[:,10].name, fontsize = 15)
sns.boxplot(HP.iloc[:,10], ax=ax_box, color = 'blue')
sns.distplot(HP.iloc[:,10], ax=ax_hist, color = 'orange')
plt.show()


# ## It is having outliers, then check for quantile ranges.

# In[ ]:


print('5% :', HP.iloc[:,10].quantile(0.05), '\n','95% :', HP.iloc[:,10].quantile(0.95))


# In[ ]:


import numpy as np
HP.iloc[:,10] = np.where(HP.iloc[:,10] > HP.iloc[:,10].quantile(0.95), np.median(HP.iloc[:,10]), HP.iloc[:,10])
HP.iloc[:,10].describe()


# In[ ]:


print('Skewness :' ,HP.iloc[:,10].skew())
sns.boxplot(HP.iloc[:,10], color = 'green')
plt.xlabel(HP.iloc[:,10].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,10].name, fontsize = 25)

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in n_hot_rooms variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'n_hot_rooms', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 11 ---- 16,17,18

# In[ ]:


print('Type : ',HP.iloc[:,11].dtype)
print('Column_name : ' ,HP.iloc[:,11].name)

print('Null_value_count: ',HP.iloc[:,11].isna().sum())


# In[ ]:


HP.iloc[:,11].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


HP.iloc[:,11].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,11].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)


# ## We have four categorical elements in the waterbody so we are going for dummification to seperate those four elements into four variables.

# In[ ]:


status = pd.get_dummies(HP.iloc[:,11], drop_first = True)
HP = pd.concat([HP,status],axis=1)
HP.head()


# In[ ]:


HP.drop('waterbody', axis=1, inplace = True)
HP.head()


# # 11

# In[ ]:


print('Type : ',HP.iloc[:,11].dtype)
print('Column_name : ' ,HP.iloc[:,11].name)

print('Null_value_count: ',HP.iloc[:,11].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,11].skew())
HP.iloc[:,11].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,11], color = 'green')
plt.xlabel(HP.iloc[:,11].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,11].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,11], color = 'orange')
plt.xlabel(HP.iloc[:,11].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in rainfall variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'rainfall', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 12

# In[ ]:


print('Type : ',HP.iloc[:,12].dtype)
print('Column_name : ' ,HP.iloc[:,12].name)

print('Null_value_count: ',HP.iloc[:,12].isna().sum())


# In[ ]:


HP.iloc[:,12].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


HP.iloc[:,12].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,12].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,12].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)


# ## It is having only one categorical element so, it wont affect the model.

# In[ ]:


HP.drop('bus_ter', axis = 1, inplace = True)


# # 12

# In[ ]:


print('Type : ',HP.iloc[:,12].dtype)
print('Column_name : ' ,HP.iloc[:,12].name)

print('Null_value_count: ',HP.iloc[:,12].isna().sum())


# In[ ]:


print('Skewness: ', HP.iloc[:,12].skew())
HP.iloc[:,12].describe()


# In[ ]:


plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,12], color = 'green')
plt.xlabel(HP.iloc[:,12].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,12].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,12], color = 'orange')
plt.xlabel(HP.iloc[:,12].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,12].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# ## No Outliers presence in parks variable.

# In[ ]:


##pairplot
sns.pairplot(data = HP, x_vars = 'parks', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)


# # 14,15,16  ---- 13 avg_dist done already

# In[ ]:


print('Type : ',HP.iloc[:,14].dtype)
print('Column_name : ' ,HP.iloc[:,14].name)

print('Null_value_count: ',HP.iloc[:,14].isna().sum())


# In[ ]:


HP.iloc[:,14].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


HP.iloc[:,14].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,14].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,14].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)


# In[ ]:


print('Type : ',HP.iloc[:,15].dtype)
print('Column_name : ' ,HP.iloc[:,15].name)

print('Null_value_count: ',HP.iloc[:,15].isna().sum())


# In[ ]:


HP.iloc[:,15].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


HP.iloc[:,15].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,15].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)


# In[ ]:


print('Type : ',HP.iloc[:,16].dtype)
print('Column_name : ' ,HP.iloc[:,16].name)

print('Null_value_count: ',HP.iloc[:,16].isna().sum())


# In[ ]:


HP.iloc[:,16].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[ ]:


HP.iloc[:,16].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,16].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,16].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)


# In[ ]:


HP.info()


# # Spliting the data.

# In[ ]:


x = HP.iloc[:,1:17]


# In[ ]:


y = HP.iloc[:,0]


# # Testing and Training data creation

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 100)


# Check for the shape (Dimensions)

# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #   Fit the models.

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(x_train,y_train)


# In[ ]:


y_pred = lr.predict(x_test)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 


# In[ ]:


corelation_table = HP.corr()
corelation_table.to_csv(r'corelation_table.csv', index = False)


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(HP.corr(),annot = True)
plt.show()


# # Check for R^2 and MSE and VIF

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x.values, i) 
                          for i in range(len(x.columns))] 
  
print(vif_data)


# ## Build the model
# 
# # full model

# In[ ]:


import statsmodels.api as sm
x_train_sm = x_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
x_train_sm = sm.add_constant(x_train_sm)
# create a fitted model in one line
mlm_1 = sm.OLS(y_train,x_train_sm).fit()

# print the coefficients
mlm_1.params
print(mlm_1.summary())


# In[ ]:


#Actual vs Predicted
c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('price', fontsize=16) 


# ## 1st Model

# In[ ]:


x1 = HP[['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'None', 'River']]


# In[ ]:


x1_train=x_train[['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'None', 'River']]
x1_test =x_test[['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'None', 'River']]


# In[ ]:


lr.fit(x1_train,y_train)


# In[ ]:


y1_pred=lr.predict(x1_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y1_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y1_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_1', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse1 = mean_squared_error(y_test, y1_pred)
r_squared1 = r2_score(y_test, y1_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse1)
print('r_square_value :',r_squared1)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x1.columns

vif_data["VIF"] = [variance_inflation_factor(x1.values, i) 
                          for i in range(len(x1.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x_train_sm = x1_train
 
x_train_sm = sm.add_constant(x_train_sm)

lm_1 = sm.OLS(y_train,x_train_sm).fit()

lm_1.params


# In[ ]:


print(lm_1.summary())


# ## 2nd Model

# In[ ]:


x2 = HP[['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


x2_train=x_train[['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]
x2_test =x_test[['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


lr.fit(x2_train,y_train)


# In[ ]:


y2_pred=lr.predict(x2_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y2_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y2_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_2', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse2 = mean_squared_error(y_test, y2_pred)
r_squared2 = r2_score(y_test, y2_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse2)
print('r_square_value :',r_squared2)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x2.columns

vif_data["VIF"] = [variance_inflation_factor(x2.values, i) 
                          for i in range(len(x2.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x2_train_sm = x2_train
 
x2_train_sm = sm.add_constant(x2_train_sm)

lm_2 = sm.OLS(y_train,x2_train_sm).fit()

lm_2.params


# In[ ]:


print(lm_2.summary())


# ## 3rd Model

# In[ ]:


x3 = HP[['crime_rate', 'resid_area', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


x3_train=x_train[['crime_rate', 'resid_area', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]
x3_test =x_test[['crime_rate', 'resid_area', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


lr.fit(x3_train,y_train)


# In[ ]:


y3_pred=lr.predict(x3_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y3_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y3_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_3', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse3 = mean_squared_error(y_test, y3_pred)
r_squared3 = r2_score(y_test, y3_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse3)
print('r_square_value :',r_squared3)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x3.columns

vif_data["VIF"] = [variance_inflation_factor(x3.values, i) 
                          for i in range(len(x3.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x3_train_sm = x3_train
 
x3_train_sm = sm.add_constant(x3_train_sm)

lm_3 = sm.OLS(y_train,x3_train_sm).fit()

lm_3.params


# In[ ]:


print(lm_3.summary())


# ## 4th Model

# In[ ]:


x4 = HP[['crime_rate', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


x4_train=x_train[['crime_rate', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]
x4_test =x_test[['crime_rate', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'parks', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


lr.fit(x4_train,y_train)


# In[ ]:


y4_pred=lr.predict(x4_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y4_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y4_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_4', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse4 = mean_squared_error(y_test, y4_pred)
r_squared4 = r2_score(y_test, y4_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse4)
print('r_square_value :',r_squared4)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x4.columns

vif_data["VIF"] = [variance_inflation_factor(x4.values, i) 
                          for i in range(len(x4.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x4_train_sm = x4_train
 
x4_train_sm = sm.add_constant(x4_train_sm)

lm_4 = sm.OLS(y_train,x4_train_sm).fit()

lm_4.params


# In[ ]:


print(lm_4.summary())


# ## 5th Model

# In[ ]:


x5 = HP[['crime_rate', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


x5_train=x_train[['crime_rate', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'Lake and River', 'River']]
x5_test =x_test[['crime_rate', 'air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


lr.fit(x5_train,y_train)


# In[ ]:


y5_pred=lr.predict(x5_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y5_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y5_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_5', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse5 = mean_squared_error(y_test, y5_pred)
r_squared5 = r2_score(y_test, y5_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse5)
print('r_square_value :',r_squared5)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x5.columns

vif_data["VIF"] = [variance_inflation_factor(x5.values, i) 
                          for i in range(len(x5.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x5_train_sm = x5_train
 
x5_train_sm = sm.add_constant(x5_train_sm)

lm_5 = sm.OLS(y_train,x5_train_sm).fit()

lm_5.params


# In[ ]:


print(lm_5.summary())


# ## 6th Model

# In[ ]:


x6 = HP[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall','avg_dist', 'Lake and River', 'River']]


# In[ ]:


x6_train=x_train[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'Lake and River', 'River']]
x6_test =x_test[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'Lake and River', 'River']]


# In[ ]:


lr.fit(x6_train,y_train)


# In[ ]:


y6_pred=lr.predict(x6_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y6_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y6_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_6', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse6 = mean_squared_error(y_test, y6_pred)
r_squared6 = r2_score(y_test, y6_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse6)
print('r_square_value :',r_squared6)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x6.columns

vif_data["VIF"] = [variance_inflation_factor(x6.values, i) 
                          for i in range(len(x6.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x6_train_sm = x6_train
 
x6_train_sm = sm.add_constant(x6_train_sm)

lm_6 = sm.OLS(y_train,x6_train_sm).fit()

lm_6.params


# In[ ]:


print(lm_6.summary())


# ## 7th Model

# In[ ]:


x7 = HP[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'River']]


# In[ ]:


x7_train=x_train[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'River']]
x7_test =x_test[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist', 'River']]


# In[ ]:


lr.fit(x7_train,y_train)


# In[ ]:


y7_pred=lr.predict(x7_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y7_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y7_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_7', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse7 = mean_squared_error(y_test, y7_pred)
r_squared7 = r2_score(y_test, y7_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse7)
print('r_square_value :',r_squared7)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x7.columns

vif_data["VIF"] = [variance_inflation_factor(x7.values, i) 
                          for i in range(len(x7.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x7_train_sm = x7_train
 
x7_train_sm = sm.add_constant(x7_train_sm)

lm_7 = sm.OLS(y_train,x7_train_sm).fit()

lm_7.params


# In[ ]:


print(lm_7.summary())


# ## 8th Model

# In[ ]:


x8 = HP[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist']]


# In[ ]:


x8_train=x_train[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist']]
x8_test =x_test[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'rainfall', 'avg_dist']]


# In[ ]:


lr.fit(x8_train,y_train)


# In[ ]:


y8_pred=lr.predict(x8_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y8_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y8_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_8', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse8 = mean_squared_error(y_test, y8_pred)
r_squared8 = r2_score(y_test, y8_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse8)
print('r_square_value :',r_squared8)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x8.columns

vif_data["VIF"] = [variance_inflation_factor(x8.values, i) 
                          for i in range(len(x8.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x8_train_sm = x8_train
 
x8_train_sm = sm.add_constant(x8_train_sm)

lm_8 = sm.OLS(y_train,x8_train_sm).fit()

lm_8.params


# In[ ]:


print(lm_8.summary())


# ## 9th Model

# In[ ]:


x9 = HP[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'avg_dist']]


# In[ ]:


x9_train=x_train[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'avg_dist']]
x9_test =x_test[['air_qual', 'room_num',
       'teachers', 'poor_prop', 'airport', 'n_hos_beds',
       'avg_dist']]


# In[ ]:


lr.fit(x9_train,y_train)


# In[ ]:


y9_pred=lr.predict(x9_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y9_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y9_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_9', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse9 = mean_squared_error(y_test, y9_pred)
r_squared9 = r2_score(y_test, y9_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse9)
print('r_square_value :',r_squared9)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x9.columns

vif_data["VIF"] = [variance_inflation_factor(x9.values, i) 
                          for i in range(len(x9.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x9_train_sm = x9_train
 
x9_train_sm = sm.add_constant(x9_train_sm)

lm_9 = sm.OLS(y_train,x9_train_sm).fit()

lm_9.params


# In[ ]:


print(lm_9.summary())


# ## 10th Model

# In[ ]:


x10 = HP[['air_qual', 'room_num',
       'teachers', 'poor_prop',
       'avg_dist']]


# In[ ]:


x10_train=x_train[['air_qual', 'room_num',
       'teachers', 'poor_prop',
       'avg_dist']]
x10_test =x_test[['air_qual', 'room_num',
       'teachers', 'poor_prop',
       'avg_dist']]


# In[ ]:


lr.fit(x10_train,y_train)


# In[ ]:


y10_pred=lr.predict(x10_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y10_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y10_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_10', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse10 = mean_squared_error(y_test, y10_pred)
r_squared10 = r2_score(y_test, y10_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse10)
print('r_square_value :',r_squared10)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x10.columns

vif_data["VIF"] = [variance_inflation_factor(x10.values, i) 
                          for i in range(len(x10.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x10_train_sm = x10_train
 
x10_train_sm = sm.add_constant(x10_train_sm)

lm_10 = sm.OLS(y_train,x10_train_sm).fit()

lm_10.params


# In[ ]:


print(lm_10.summary())


# ## 11th Model

# In[ ]:


x11 = HP[['room_num',
       'teachers', 'poor_prop']]


# In[ ]:


x11_train=x_train[['room_num',
       'teachers', 'poor_prop']]
x11_test =x_test[['room_num',
       'teachers', 'poor_prop']]


# In[ ]:


lr.fit(x11_train,y_train)


# In[ ]:


y11_pred=lr.predict(x11_test)


# In[ ]:


c=[i for i in range(1,103,1)]
fig=plt.figure()
plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")
plt.plot(c,y11_pred,color="red",linewidth=2.5,linestyle="-")
fig.suptitle('Actual and Predicted',fontsize=20)
plt.xlabel("index",fontsize=18)
plt.ylabel("price",fontsize=16)


# In[ ]:


c = [i for i in range(1,103,1)]
fig = plt.figure()
plt.plot(c,y_test-y11_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)            
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Error_Line_11', fontsize=16) 


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse11 = mean_squared_error(y_test, y11_pred)
r_squared11 = r2_score(y_test, y11_pred)


# In[ ]:


print('Mean_Squared_Error :' ,mse11)
print('r_square_value :',r_squared11)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x11.columns

vif_data["VIF"] = [variance_inflation_factor(x11.values, i) 
                          for i in range(len(x11.columns))] 
  
print(vif_data)


# In[ ]:


import statsmodels.api as sm
x11_train_sm = x11_train
 
x11_train_sm = sm.add_constant(x11_train_sm)

lm_11 = sm.OLS(y_train,x11_train_sm).fit()

lm_11.params


# In[ ]:


print(lm_11.summary())


# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(x10.corr(),annot = True)


# ### Finally we got the result :  Model-10
# 
# #### After building  a models we choosen the model 10 because of:
# 
# #### As per the model seen  Adj-R^2 69 % that the model get fitted with following variables -  'air_qual', 'room_num', 'teachers', 'poor_prop', 'avg_dist'

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**@** **Analysis done by,**
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- **@**   **Jagadeesh K**
