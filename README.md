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
  import pandas as pd
 import numpy as np
 import seaborn as sns
  from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("income.csv",na_values=[ " ?"])
 data
 <img width="922" height="635" alt="image" src="https://github.com/user-attachments/assets/f7fb8e22-c998-4ab7-a9b7-85c217db5cfe" />
 data.isnull().sum()
 <img width="407" height="361" alt="image" src="https://github.com/user-attachments/assets/890703d6-efe7-4408-9d27-56f5c9dfeb84" />
 missing=data[data.isnull().any(axis=1)]
 missing
 <img width="776" height="636" alt="image" src="https://github.com/user-attachments/assets/952fc808-7add-40a8-b16c-c6d1436664ff" />

# RESULT:
       # INCLUDE YOUR RESULT HERE
