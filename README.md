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
from scipy import stats
import numpy as np
df = pd.read_csv("C:/Users/acer/Downloads/bmi.csv")
df

```

<img width="410" height="291" alt="image" src="https://github.com/user-attachments/assets/6c4f70d2-e2a8-469f-abce-a41f0e31d6e1" />


```
df.head()

```

<img width="254" height="146" alt="image" src="https://github.com/user-attachments/assets/8ae68f57-8822-4375-957f-6d7b4bf48b4c" />


```
df.dropna()

```

<img width="342" height="299" alt="image" src="https://github.com/user-attachments/assets/cc84103c-0112-4a37-80e8-94009f0568b8" />


```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Hs','Ws']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)

```
<img width="429" height="245" alt="image" src="https://github.com/user-attachments/assets/835ed727-26d3-45b0-bf47-b4508241dcd8" />


```
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df[['Hm','Wm']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)

```
<img width="464" height="244" alt="image" src="https://github.com/user-attachments/assets/19f22238-021c-4ead-9df0-ab007211675c" />


```
import pandas as pd
df = pd.read_csv("C:/Users/acer/Downloads/titanic_dataset.csv")
df.columns

```

<img width="572" height="70" alt="image" src="https://github.com/user-attachments/assets/d68dc1f0-7e23-4d48-839f-e67c300258e5" />


```
df

```

<img width="687" height="411" alt="image" src="https://github.com/user-attachments/assets/5268c847-82f0-4550-814c-f5e42bfc7bd1" />


```
df.isnull().sum()

```
<img width="263" height="200" alt="image" src="https://github.com/user-attachments/assets/d987b597-e3da-4551-88fc-ceab8e33fa7b" />


```

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Input and output
X = df[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = df['Survived']

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=4)

X_new = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)

selected_features = X.columns[selected_feature_indices]

print("Selected Features:")
print(selected_features)


```

<img width="539" height="56" alt="image" src="https://github.com/user-attachments/assets/63dd773a-3313-4271-b94f-9f2a216ce2a0" />


```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```

<img width="358" height="140" alt="image" src="https://github.com/user-attachments/assets/29f89b13-0edf-44f9-85c0-51927526d951" />


```

tips.time.unique()

```
<img width="435" height="48" alt="image" src="https://github.com/user-attachments/assets/598a2ba4-0949-45bd-bd33-9084ec7e9e36" />


```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```

<img width="200" height="71" alt="image" src="https://github.com/user-attachments/assets/9e086467-4738-4cea-8595-02277c3639cc" />


```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
<img width="402" height="49" alt="image" src="https://github.com/user-attachments/assets/f15c7647-f41d-4db4-8fd6-f727bdeecb4a" />





# RESULT:
Thus the given data is read and performed the Feature Scaling and Feature Selection process and saved the file
