# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
df
```

<img width="1558" height="711" alt="Screenshot 2025-10-18 102947" src="https://github.com/user-attachments/assets/406b9a3e-62ea-42fd-a79d-a341c2fb05b0" />

```

df.isnull().sum()
```


<img width="840" height="543" alt="Screenshot 2025-10-18 103043" src="https://github.com/user-attachments/assets/0f66e143-5032-4f8d-83ec-e748443cab1d" />

```
ms=df[df.isnull().any(axis=1)]
ms
```

<img width="1552" height="660" alt="Screenshot 2025-10-18 103139" src="https://github.com/user-attachments/assets/7854b673-b683-4570-8bcd-c286b64632e9" />

```
ms=df[df.isnull().any(axis=1)]
ms
```

<img width="1548" height="656" alt="Screenshot 2025-10-18 103230" src="https://github.com/user-attachments/assets/c5182254-574e-421a-ae69-50434d1d3482" />

```
df2=df.dropna(axis=0)
df2
```

<img width="1546" height="656" alt="Screenshot 2025-10-18 103320" src="https://github.com/user-attachments/assets/775dc8b1-cc64-43a1-85df-acb8bfd5d1fd" />

```
s=df["SalStat"]
df2["SalStat"]=df["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
```

<img width="1418" height="365" alt="Screenshot 2025-10-18 103408" src="https://github.com/user-attachments/assets/2490482a-0353-4107-a0b0-31df92000db6" />

```
s2=df2['SalStat']
dfs=pd.concat([s,s2],axis=1)
dfs
```

<img width="596" height="462" alt="Screenshot 2025-10-18 103454" src="https://github.com/user-attachments/assets/0aab82d9-e177-427f-918d-d4b0f22a43b5" />

```
new_df2=pd.get_dummies(df2, drop_first=True)
new_df2
```

<img width="1695" height="278" alt="Screenshot 2025-10-18 103701" src="https://github.com/user-attachments/assets/11bc252a-877c-4665-b2aa-16fd3c02df34" />

```
col=list(new_df2.columns)
print(col)
```

<img width="1635" height="43" alt="Screenshot 2025-10-18 103959" src="https://github.com/user-attachments/assets/4dae5aa0-af1e-479e-9f20-bad55e0f84be" />
```
fea=list(set(col)-set(['SalStat']))
print(fea)
```

<img width="1609" height="44" alt="Screenshot 2025-10-18 104120" src="https://github.com/user-attachments/assets/d40e7a5c-7a56-4a00-9fce-6626d7b00052" />

```
y=new_df2['SalStat'].values
print(y)
```

<img width="356" height="22" alt="Screenshot 2025-10-18 104254" src="https://github.com/user-attachments/assets/ac7542a8-cd29-44a7-81a8-47eb1b2ed39b" />

```
x=new_df2[fea].values
print(x)
```

<img width="441" height="123" alt="Screenshot 2025-10-18 104422" src="https://github.com/user-attachments/assets/8c1682ca-0cda-40a5-a6e4-1c28c4bc1709" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="286" height="61" alt="Screenshot 2025-10-18 104513" src="https://github.com/user-attachments/assets/b6bda020-9e0e-4db7-b072-4b43dc734229" />

```
pred=KNN_classifier.predict(test_x)
con=confusion_matrix(test_y, pred)
print(con)
```

<img width="194" height="45" alt="Screenshot 2025-10-18 104615" src="https://github.com/user-attachments/assets/3a05ddf9-4978-4aed-927e-d35700a3b993" />

```
acc=accuracy_score(test_y,pred)
print(acc)
```

<img width="253" height="24" alt="Screenshot 2025-10-18 104718" src="https://github.com/user-attachments/assets/4d5bf6a7-5d5b-4a60-acc2-4d8074f71f0d" />

```
print("Misclassified Samples : %d" % (test_y !=pred).sum())
```

<img width="283" height="27" alt="Screenshot 2025-10-18 104809" src="https://github.com/user-attachments/assets/dfa97fe3-65f8-4010-a880-369bffc9bfa3" />

```
df.shape
```

<img width="137" height="27" alt="Screenshot 2025-10-18 104852" src="https://github.com/user-attachments/assets/c14c0d20-0e9c-4adf-984e-e4128110ec25" />
```
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1609" height="90" alt="Screenshot 2025-10-18 105040" src="https://github.com/user-attachments/assets/ff7120a9-e758-4db3-b5ac-dc1b3c6c356b" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="516" height="190" alt="Screenshot 2025-10-18 105122" src="https://github.com/user-attachments/assets/e29022fd-4717-4987-a56d-9e0aa53c31f4" />
```
tips.time.unique()
```

<img width="395" height="57" alt="Screenshot 2025-10-18 105207" src="https://github.com/user-attachments/assets/f8e3f6b6-2596-4093-8175-c2c31c63d872" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="259" height="79" alt="Screenshot 2025-10-18 105307" src="https://github.com/user-attachments/assets/b4493097-600a-4100-8d62-cb6bb907694f" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="372" height="51" alt="Screenshot 2025-10-18 105347" src="https://github.com/user-attachments/assets/cedda8d8-3316-4a7f-948f-c9a8e5f4064d" />




















# RESULT:
Hence the proram is done and the output is verified successfully.
