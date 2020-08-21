#!/usr/bin/env python
## coding: utf-8

# ### Initial Import for Standard Libraries

# In[1]:


# importing libraries.!!

import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Load the Data

df = pd.read_csv('./heart.csv')


# _____________

# In[3]:


#Evaluating Non Zero Columns

df.isnull().sum()


# There are No Columns that have Non Zero Value

# In[53]:


df.head()


# ____________________

# In[5]:


f,ax=plt.subplots(1,2,figsize=(8,4))

sns.set_context("paper", font_scale = 2, rc = {"font.size": 12,"axes.titlesize": 15,"axes.labelsize": 12}) 

df.loc[df['sex']==1, 'target'].value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[0],shadow=True)
df.loc[df['sex']==0, 'target'].value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Patients (male)')
ax[1].set_title('Patients (female)')

plt.show()


# In[6]:


plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[7]:


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[8]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#2E86C1','#F1C40F' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()


# Use Pandas Profiling to Improve EDA for data.

# In[9]:


pip install pandas-profiling


# In[10]:


import pandas_profiling


# In[11]:


pandas_profiling.ProfileReport(df)


# __________________________________________________

# Plot the Split of Diseases across Male and Female

# In[12]:


print("{} % of Women Suffer from Heart Diseases".format(100*(df.loc[df.sex == 0].target.sum()/df.loc[df.sex == 0].target.count())))
print("{} % of Men Suffer from Heart Diseases".format(round(100*(df.loc[df.sex == 1].target.sum()/df.loc[df.sex == 1].target.count()))))


# ### Train Test Split

# In[13]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[15]:


X_train


# ______________________________

# ## Model Evaluation

# In[16]:


# Importing Models that needs to be evaluated

from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[17]:


# prepare models

model_list = []


# In[18]:


model_list.append(('Multinomial NB', MultinomialNB(alpha=0.1)))
model_list.append(('Decision Tree', DecisionTreeClassifier()))
model_list.append(('SVM',SVC(kernel='linear')))
model_list.append(('ADA Boost With Decision Tree', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
    n_estimators=200)))
model_list.append(('Logistic Regression',LogisticRegression(solver='liblinear')))
model_list.append(('Random Forest',RandomForestClassifier()))


# In[19]:


# Variable to Score Results
results = []
names = []
scoring = 'accuracy'
seed = 5


# #### Cross Validation Score

# In[20]:


# Evaluation of Each Model One by One (Cross Validation Score)

for name, model in model_list:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# #####  Preparing a DataFrame to Plot the Scores

# In[21]:


def insert(df, row):
    insert_loc = df.index.max()
    if np.isnan(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row


# In[22]:


cv_score = df_ = pd.DataFrame(columns = ['Model_Name','CV_Score'])
cv_score.head()


# In[23]:


for key,value in enumerate(results):
        insert(cv_score,[names[key],value.mean()*100])


# In[24]:


cv_score


# In[25]:


sns.set_context("paper", font_scale = 1, rc = {"font.size": 12,"axes.titlesize": 15,"axes.labelsize": 12}) 
plot = sns.catplot(x="Model_Name", y="CV_Score", hue="CV_Score", kind="point", data=cv_score,height=5,aspect=1.5,markers="^")


# ### Libaries to explain the Models

# In[26]:


pip install eli5


# In[27]:


#Libraries for Explaning ML Models

import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance


# In[28]:


df1 = df[df.columns.difference(['target'])]


# In[29]:


perm_list =[]


# In[30]:


# Evaluation of Each Model One by One (Cross Validation Score)

for name, model in model_list:
    model.fit(X_train, y_train)
    perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
    perm_list.append(perm)


# In[31]:


for index,value in enumerate(perm_list):
    eli5.show_weights(perm_list[index], feature_names = df1.columns.tolist())


# In[32]:


# Feature Importance using MultinomialNB
eli5.show_weights(perm_list[0], feature_names = df1.columns.tolist())


# In[33]:


# Feature Importance using DecisionTreeClassifier
eli5.show_weights(perm_list[1], feature_names = df1.columns.tolist())


# In[34]:


# Feature Importance using SVM
eli5.show_weights(perm_list[2], feature_names = df1.columns.tolist())


# In[35]:


# Feature Importance using AdaBoostClassifier
eli5.show_weights(perm_list[3], feature_names = df1.columns.tolist())


# In[36]:


# Feature Importance using LogisticRegression
eli5.show_weights(perm_list[4], feature_names = df1.columns.tolist())


# In[37]:


# Feature Importance using RandomForestClassifier
eli5.show_weights(perm_list[5], feature_names = df1.columns.tolist())


# ### Accuracy & RoC Curve

# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


# In[39]:


def plotting(true,pred):
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    precision,recall,threshold = precision_recall_curve(true,pred[:,1])
    ax[0].plot(recall,precision,'g--')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))
    fpr,tpr,threshold = roc_curve(true,pred[:,1])
    ax[1].plot(fpr,tpr)
    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))
    ax[1].plot([0,1],[0,1],'k--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')


# In[40]:


sns.set_context("paper", font_scale = 1.5, rc = {"font.size": 11,"axes.titlesize": 14,"axes.labelsize": 11}) 


# #### Random Forest

# In[41]:


RandomForestClassifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
RandomForestClassifier.fit(X_train,y_train)


# In[42]:


plotting(y_test,RandomForestClassifier.predict_proba(X_test))
plt.figure()


# #### MultinomialNB

# In[43]:


MultinomialNB = MultinomialNB(alpha=0.1)
MultinomialNB.fit(X_train,y_train)


# In[44]:


plotting(y_test,MultinomialNB.predict_proba(X_test))
plt.figure()


# #### DecisionTreeClassifier

# In[45]:


DecisionTreeClassifier = DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_train,y_train)


# In[46]:


plotting(y_test,DecisionTreeClassifier.predict_proba(X_test))
plt.figure()


# #### Linear SVC

# In[47]:


LinearSVC = SVC(kernel='linear',probability=True)
LinearSVC.fit(X_train,y_train)


# In[48]:


plotting(y_test,LinearSVC.predict_proba(X_test))
plt.figure()


# #### AdaBoostClassifier

# In[49]:


AdaBoostClassifier = AdaBoostClassifier()
AdaBoostClassifier.fit(X_train,y_train)


# In[50]:


plotting(y_test,AdaBoostClassifier.predict_proba(X_test))
plt.figure()


# ### LogisticRegression

# In[51]:


LogisticRegression = LogisticRegression(solver='liblinear')
LogisticRegression.fit(X_train,y_train)


# In[52]:


plotting(y_test,LogisticRegression.predict_proba(X_test))
plt.figure()


# In[ ]:





# In[ ]:




