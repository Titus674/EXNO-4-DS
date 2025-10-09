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
# FEATURE SCALING
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

 ```

<img width="1400" height="493" alt="image" src="https://github.com/user-attachments/assets/509232a8-64d9-4775-8514-bba79feabe15" />


```

df_null_sum=df.isnull().sum()
df_null_sum

```

<img width="1461" height="346" alt="image" src="https://github.com/user-attachments/assets/0f121b32-1059-4b78-bb18-c9a9bcd7574c" />


```

df.dropna()

```

<img width="1402" height="586" alt="image" src="https://github.com/user-attachments/assets/cff1e7c5-fc6a-46cd-b664-c99102ef8f12" />


```

max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).

```

<img width="1387" height="367" alt="image" src="https://github.com/user-attachments/assets/8d3f251f-3905-4570-8fa3-b30253c219c5" />

```

# Standard Scaling
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()

```

<img width="1408" height="451" alt="image" src="https://github.com/user-attachments/assets/741c46c8-8965-454e-9a9c-177eeb3d1eec" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

```

<img width="1399" height="614" alt="image" src="https://github.com/user-attachments/assets/3bf447ce-ccbd-4f1f-821e-e97d280fc5e6" />


```

#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

```

<img width="1403" height="628" alt="image" src="https://github.com/user-attachments/assets/9cd0913a-87b9-48a0-995d-1667e207e604" />

```
#MAXIMUM ABSOLUTE SCALING:
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

```

<img width="1418" height="789" alt="image" src="https://github.com/user-attachments/assets/b178857a-d866-453e-92f7-83828b98ff49" />


```

#ROBUST SCALING
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()

```

<img width="1409" height="490" alt="image" src="https://github.com/user-attachments/assets/b0573322-2e79-4e11-90ca-477376452e05" />


```

#FEATURE SELECTION:
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

```

<img width="1408" height="570" alt="image" src="https://github.com/user-attachments/assets/2f7b6cda-2eb0-4017-af7b-c0f6a6d55a2e" />


```

df_null_sum=df.isnull().sum()
df_null_sum

```

<img width="1429" height="699" alt="image" src="https://github.com/user-attachments/assets/a657a603-d560-428f-a5d8-e140da8872ae" />



````

# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
# process non-numeric features. It ensures the model treats these columns as categories,
# not as continuous numerical values.
df[categorical_columns]


```

<img width="1417" height="789" alt="image" src="https://github.com/user-attachments/assets/0e8fc050-14e8-4647-84a8-abd07aa313e8" />

```

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]

```

<img width="1413" height="682" alt="image" src="https://github.com/user-attachments/assets/a7ba71a4-14d1-4dc7-bf7d-d97470d86d65" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```

<img width="1403" height="382" alt="image" src="https://github.com/user-attachments/assets/dd03deb7-6c70-4dbc-8661-06a3565f0d03" />


```

y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

```

<img width="1402" height="557" alt="image" src="https://github.com/user-attachments/assets/17371fea-5a74-4b4c-a1e1-034dd0915c34" />

```

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```

<img width="1416" height="688" alt="image" src="https://github.com/user-attachments/assets/bf40cc31-7225-4dc0-8d00-9794c7b11e96" />
       
```

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

```

<img width="1411" height="615" alt="image" src="https://github.com/user-attachments/assets/3cc4154a-c44c-4d41-b742-89f0ee10e2a5" />

```

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)

```

<img width="1422" height="335" alt="image" src="https://github.com/user-attachments/assets/4a437ae4-bb1c-4985-9f4a-7bad9f590077" />


```

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```

<img width="1429" height="452" alt="image" src="https://github.com/user-attachments/assets/8fec7c53-74ef-4168-b2af-7d1d30044117" />


```

y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")

```

<img width="1413" height="167" alt="image" src="https://github.com/user-attachments/assets/8f15f44f-759a-4459-a7b3-6f4eeb06c8ed" />




```
!pip install skfeature-chappers

```

<img width="1418" height="447" alt="image" src="https://github.com/user-attachments/assets/9be8b00e-788f-4d74-9af1-0da8cb25b0b7" />

```

import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]

```

<img width="1285" height="529" alt="image" src="https://github.com/user-attachments/assets/6c31d003-bec1-4931-ad29-1fc410029f3e" />


```

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)

```

<img width="1416" height="348" alt="image" src="https://github.com/user-attachments/assets/a2f506a2-89be-4f3b-83e1-c10fba317d93" />


```

#Index(['age', 'maritalstatus', 'relationship', 'capitalgain', 'hoursperweek'], dtype='object')
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

```

<img width="1327" height="532" alt="image" src="https://github.com/user-attachments/assets/6e097d3c-690f-4308-8271-cdaf6ea78be3" />

              


# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
