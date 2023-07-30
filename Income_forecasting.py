#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv('D:\\Engineering\\Machine learning\\Datasets\\Income\\adult.csv')
data.shape


# In[3]:


missing=[' ?']
data=pd.read_csv('D:\\Engineering\\Machine learning\\Datasets\\Income\\adult.csv',na_values=missing)
data.isnull().sum()


# In[4]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[5]:


data=data.fillna(method='ffill')
data


# In[6]:


##Native Country
L=data['native-country'].values
a=[]
for i in L:
    if i not in a:
        a.append(i)
a.remove(' United-States')
data=data.replace(to_replace=a,value='other')


# In[7]:


##race
R=data['race'].values
a=[]
for i in R:
    if i not in a:
        a.append(i)
a.remove(' White')
data=data.replace(to_replace=a,value='other')


# In[8]:


##education
E=data['education'].values
a=[]
for i in E:
    if i not in a:
        a.append(i)
a1=[]
for i in a:
    if 'th' in i:
        a1.append(i)
data=data.replace(to_replace=a1,value='School')


# In[9]:


#income
data['Income']=data['Income'].replace({' <=50K':0,' >50K':1})


# In[10]:


X=data.drop(columns='Income',axis=1)
Y=data['Income']


# In[11]:


std=StandardScaler()
X1=X.select_dtypes(include=['int64','float64'])
columns=X1.columns
X1=std.fit_transform(X1)
X1=pd.DataFrame(X1,columns=columns)
X1


# In[12]:


X2=data.select_dtypes(include=['object'])
X2=pd.get_dummies(X2)
X2


# In[13]:


X=pd.concat([X1,X2],axis=1)
X


# In[14]:


Y=data['Income']
Y=pd.DataFrame(Y)
Y


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#LogisticRegression
# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print('Accuracy =',round(accuracy,2))
cm=confusion_matrix(y_test,y_pred)
precision=(cm[0][0]/(cm[0][0]+cm[1][0]))
recall=(cm[0][0]/(cm[0][0]+cm[0][1]))
f1_score=2*(precision*recall)/(precision+recall)
print('Precision =',round(precision,2))
print('Recall =',round(recall,2))
print('F1 Score =',round(f1_score,2))
print(cm)

#support vector machine
# In[45]:


from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
y_svm_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_svm_pred)
sum1=cm[0][0]+cm[1][1]
sum2=sum1+cm[0][1]+cm[1][0]
accuracy=(sum1/sum2)
print('Accuracy =',round(accuracy,2))
precision=(cm[0][0]/(cm[0][0]+cm[1][0]))
recall=(cm[0][0]/(cm[0][0]+cm[0][1]))
f1_score=2*(precision*recall)/(precision+recall)
print('Precision =',round(precision,2))
print('Recall =',round(recall,2))
print('F1 Score =',round(f1_score,2))
print(cm)

#KNN
# In[44]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2 )
classifier.fit(x_train,y_train)
y_knn_pred=classifier.predict(x_test)
cm=confusion_matrix(y_knn_pred,y_test)
sum1=cm[0][0]+cm[1][1]
sum2=sum1+cm[0][1]+cm[1][0]
accuracy_knn=round((sum1/sum2),2)
print('Accuracy =',accuracy_knn)
precision=(cm[0][0]/(cm[0][0]+cm[1][0]))
recall=(cm[0][0]/(cm[0][0]+cm[0][1]))
f1_score=2*(precision*recall)/(precision+recall)
print('Precision =',round(precision,2))
print('Recall =',round(recall,2))
print('F1 Score =',round(f1_score,2))
print(cm)

#Random forest classifier
# In[43]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
classifier.fit(x_train,y_train)
y_rf_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_rf_pred)
sum1=cm[0][0]+cm[1][1]
sum2=sum1+cm[0][1]+cm[1][0]
accuracy_rf=round((sum1/sum2),2)
print('Accuracy =',accuracy_rf)
precision=(cm[0][0]/(cm[0][0]+cm[1][0]))
recall=(cm[0][0]/(cm[0][0]+cm[0][1]))
f1_score=2*(precision*recall)/(precision+recall)
print('Precision =',round(precision,2))
print('Recall =',round(recall,2))
print('F1 Score =',round(f1_score,2))
print(cm)

#Decision Tree Classifier
# In[42]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
y_dt_pred=classifier.predict(x_test)
cm=confusion_matrix(y_dt_pred,y_test)
sum1=cm[0][0]+cm[1][1]
sum2=sum1+cm[0][1]+cm[1][0]
accuracy_dt=round((sum1/sum2),2)
print('Accuracy =',accuracy_dt)
precision=(cm[0][0]/(cm[0][0]+cm[1][0]))
recall=(cm[0][0]/(cm[0][0]+cm[0][1]))
f1_score=2*(precision*recall)/(precision+recall)
print('Precision =',round(precision,2))
print('Recall =',round(recall,2))
print('F1 Score =',round(f1_score,2))
print(cm)

#Naive Bayes
# In[41]:


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_nb_pred=classifier.predict(x_test)
cm_nb=confusion_matrix(y_nb_pred,y_test)
sum1=cm[0][0]+cm[1][1]
sum2=sum1+cm[1][0]+cm[0][1]
accuracy_nb=round((sum1/sum2),2)
print('Accuracy =',accuracy_nb)
precision=(cm[0][0]/(cm[0][0]+cm[1][0]))
recall=(cm[0][0]/(cm[0][0]+cm[0][1]))
f1_score=2*(precision*recall)/(precision+recall)
print('Precision =',round(precision,2))
print('Recall =',round(recall,2))
print('F1 Score =',round(f1_score,2))
print(cm)


# In[ ]:




