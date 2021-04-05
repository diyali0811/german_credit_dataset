#!/usr/bin/env python
# coding: utf-8

# In[213]:


import numpy as np
import pandas as pd


# In[214]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report


# ### READING THE DATASET

# In[215]:


credit_dat=pd.read_csv("GermanData.csv")


# In[216]:


print(credit_dat.head())


# In[217]:


credit_dat.shape


# In[ ]:


dataset=credit_dat.dropna(axis=0)


# ### DATA PRE-PROCESSING

# In[219]:


from sklearn.preprocessing import LabelEncoder
le  = LabelEncoder()

credit_dat["Checking_acc_status"] = le.fit_transform(credit_dat["Checking_acc_status"])
credit_dat["Credit_his"] = le.fit_transform(credit_dat["Credit_his"])
credit_dat["Purpose"] = le.fit_transform(credit_dat["Purpose"])
credit_dat["Saving_acc"] = le.fit_transform(credit_dat["Saving_acc"])
credit_dat["Present_empl"] = le.fit_transform(credit_dat["Present_empl"])
credit_dat["Personal_status"] = le.fit_transform(credit_dat["Personal_status"])
credit_dat["Guarantors"] = le.fit_transform(credit_dat["Guarantors"])
credit_dat["Property"] = le.fit_transform(credit_dat["Property"])
credit_dat["Other_installment"] = le.fit_transform(credit_dat["Other_installment"])
credit_dat["Housing"] = le.fit_transform(credit_dat["Housing"])
credit_dat["Job"] = le.fit_transform(credit_dat["Job"])
credit_dat["Telephone"] = le.fit_transform(credit_dat["Telephone"])
credit_dat["Foreign_workers"] = le.fit_transform(credit_dat["Foreign_workers"])




# In[220]:


credit_dat.head(20)


# In[222]:


target=credit_dat["Credit_risk"]


# In[224]:


data=credit_dat.drop(columns=["Credit_risk"])


# In[225]:


data.head()


# In[226]:


CrossTabResult=pd.crosstab(index=credit_dat['Checking_acc_status'], columns=credit_dat['Credit_risk'])
CrossTabResult


# In[227]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data, target, test_size=0.3,random_state=0)


# ### SPLITTING IN DATA INTO TRAIN AND TEST 

# In[228]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.fit_transform(x_test)


# In[229]:


pd.DataFrame(x_train_sc).describe()


# ### LOGISTIC REGRESSION

# In[230]:


from sklearn. linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train_sc, y_train)
pred_train_lr=lr.predict(x_train_sc)


# In[231]:


from sklearn.metrics import accuracy_score
l_train=(accuracy_score(pred_train_lr, y_train))

pred_test_lr=lr.predict(x_test_sc)
l_test=(accuracy_score(pred_test_lr, y_test))
print("Train Accuracy:",l_train)
print("Test Accuracy:",l_test)


# ### K-NEAREST NEIGHBOUR

# In[232]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[233]:


knn.fit(x_train, y_train)


# In[234]:


pred_train_knn=knn.predict(x_train)
pred_test_knn=knn.predict(x_test)


# In[235]:


k_train=accuracy_score(pred_train_knn, y_train)
k_test=accuracy_score(pred_test_knn, y_test)
print("Train Accuracy:",k_train)
print("Test Accuracy:",k_test)


# ### KNN FOR DIFFERENT VALUES OF 'K'

# In[236]:


for k in range(1,11):
  print("k: ", k)
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train,y_train)
  print("Train Accuracy:",accuracy_score(knn.predict(x_train),y_train))
  print("Test Accuracy:",accuracy_score(knn.predict(x_test), y_test))


# ### NAIVE BAYES

# In[237]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train_sc, y_train)


# In[238]:


from sklearn.metrics import accuracy_score
nb_train=accuracy_score(nb.predict(x_train_sc), y_train)
nb_test=accuracy_score(nb.predict(x_test_sc), y_test)
print("Train Accuracy:", nb_train)
print("Test Accuracy:",nb_test)


# ### DECISION TREE

# In[239]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train_sc, y_train)


# In[240]:


dt_train=accuracy_score(dt.predict(x_train_sc), y_train)
dt_test=accuracy_score(dt.predict(x_test_sc),y_test)
print("Train Accuracy:",dt_train)
print("Test Accuracy:",dt_test)


# In[241]:


dt1=DecisionTreeClassifier(criterion='entropy')
dt1.fit(x_train, y_train)


# In[242]:


dt_train1=accuracy_score(dt1.predict(x_train_sc), y_train)
dt_test1=accuracy_score(dt1.predict(x_test_sc),y_test)
print("Training Accuracy:",dt_train1)
print("Testing Accuracy:",dt_test1)


# ### SUPPORT VECTOR MACHINE

# In[243]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)


# In[244]:


pred_train=svc.predict(x_train)


# In[245]:


from sklearn.metrics import accuracy_score
sv_train=accuracy_score(pred_train, y_train)
pred_test=svc.predict(x_test)
print("Training Accuracy:",sv_train)
sv_test=accuracy_score(pred_test, y_test)
print("Testing Accuracy:",sv_test)


# ### GRAPH OF TRAINING ACCURACY OF ABOVE ALGORITHMS

# In[246]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
x=np.arange(1)
w=1
plt.bar(x-w,l_train,w,color='green')
plt.bar(x,k_train,w,color='red')
plt.bar(x+w,nb_train,w,color='blue')
plt.bar(x+w+1,dt_train1,w,color='orange')
plt.bar(x+w+2,sv_train,w,color='yellow')

green_patch=mpatches.Patch(color='green',label='Logistic Regression')
red_patch=mpatches.Patch(color='red',label='KNN')
blue_patch=mpatches.Patch(color='blue',label='NAIVE BAYES')
orange_patch=mpatches.Patch(color='orange',label='DECISION TREE')
yellow_patch=mpatches.Patch(color='yellow',label='SVM')
plt.legend(handles=[green_patch,red_patch,blue_patch,orange_patch,yellow_patch])
plt.xlabel("ALGORITHM")
plt.ylabel("ACCURACY")
plt.title("Train Accuracy of different Algorithms")

plt.show()



# ### GRAPH OF TESTING ACCURACY OF ABOVE ALGORITHMS

# In[247]:



x=np.arange(1)
w=1
plt.bar(x-w,l_test,w,color='green')
plt.bar(x,k_test,w,color='red')
plt.bar(x+w,nb_test,w,color='blue')
plt.bar(x+w+1,dt_test,w,color='orange')
plt.bar(x+w+2,sv_test,w,color='yellow')

green_patch=mpatches.Patch(color='green',label='Logistic Regression')
red_patch=mpatches.Patch(color='red',label='KNN')
blue_patch=mpatches.Patch(color='blue',label='NAIVE BAYES')
orange_patch=mpatches.Patch(color='orange',label='DECISION TREE')
yellow_patch=mpatches.Patch(color='yellow',label='SVM')
plt.legend(handles=[green_patch,red_patch,blue_patch,orange_patch,yellow_patch])
plt.xlabel("ALGORITHM")
plt.ylabel("ACCURACY")
plt.title("Test Accuracy of different Algorithms")

plt.show()


# In[ ]:




