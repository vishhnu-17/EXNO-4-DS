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
  import pandas as pd <br>
 import numpy as np <br>
 import seaborn as sns <br>
  from sklearn.model_selection import train_test_split <br>
 from sklearn.neighbors import KNeighborsClassifier <br>
 from sklearn.metrics import accuracy_score, confusion_matrix <br>
 data=pd.read_csv("income.csv",na_values=[ " ?"]) <br>
 data <br>
 <img width="922" height="635" alt="image" src="https://github.com/user-attachments/assets/f7fb8e22-c998-4ab7-a9b7-85c217db5cfe" /> <br>
 data.isnull().sum() <br>
 <img width="407" height="361" alt="image" src="https://github.com/user-attachments/assets/890703d6-efe7-4408-9d27-56f5c9dfeb84" /> <br>
 missing=data[data.isnull().any(axis=1)] <br>
 missing <br>
 <img width="776" height="636" alt="image" src="https://github.com/user-attachments/assets/952fc808-7add-40a8-b16c-c6d1436664ff" /> <br>
 data2=data.dropna(axis=0) <br>
 data2 <br>
 <img width="905" height="661" alt="image" src="https://github.com/user-attachments/assets/5f97433f-3bcc-4bac-85d0-f4cfc0eac2ef" /> <br>
  sal=data["SalStat"]<br>
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1}) <br>
 print(data2['SalStat']) <br>
 <img width="906" height="584" alt="image" src="https://github.com/user-attachments/assets/78f09c8c-da81-45e1-bf43-76c7175c3ea1" /> <br>
 sal2=data2['SalStat'] <br>
 dfs=pd.concat([sal,sal2],axis=1) <br>
 dfs <br>
 <img width="662" height="552" alt="image" src="https://github.com/user-attachments/assets/67a2dcef-0c22-4a8d-93d4-62fab12caeb7" /> <br>
data2 <br>
<img width="895" height="591" alt="image" src="https://github.com/user-attachments/assets/42dd6428-2295-45af-b7de-173feb1ba528" /> <br>
 new_data=pd.get_dummies(data2, drop_first=True) <br>
 new_data <br>
 <img width="901" height="585" alt="image" src="https://github.com/user-attachments/assets/2eea0b67-0eb2-46ed-930c-8c3594aefce3" /> <br>
 columns_list=list(new_data.columns) <br>
 print(columns_list) <br>

<img width="894" height="679" alt="image" src="https://github.com/user-attachments/assets/70307a82-4d7a-4def-88d9-dc87eec329f4" /> <br>
features=list(set(columns_list)-set(['SalStat'])) <br>
print(features) <br>

<img width="901" height="668" alt="image" src="https://github.com/user-attachments/assets/bd1f7132-c9ca-41f9-8bb2-decf91e2663c" /> <br>
 y=new_data['SalStat'].values <br>
 print(y) <br>

 <img width="464" height="112" alt="image" src="https://github.com/user-attachments/assets/3fb85704-33eb-4d70-ad6d-1b60f6253f49" /> <br>
 x=new_data[features].values <br>
 print(x) <br>

 <img width="506" height="237" alt="image" src="https://github.com/user-attachments/assets/a1263185-4be2-418c-aafa-ab72f01a9425" /> <br>
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0) <br>
KNN_classifier=KNeighborsClassifier(n_neighbors = 5) <br>
KNN_classifier.fit(train_x,train_y) <br>
prediction=KNN_classifier.predict(test_x) <br>
confusionMatrix=confusion_matrix(test_y, prediction) <br>
print(confusionMatrix) <br>
<img width="874" height="223" alt="image" src="https://github.com/user-attachments/assets/89e93810-3ef9-4b41-a96f-9e2f06c66ce0" /> <br>
 accuracy_score=accuracy_score(test_y,prediction) <br>
 print(accuracy_score) <br>
 
 <img width="878" height="121" alt="image" src="https://github.com/user-attachments/assets/d432ab5f-c4dc-46af-a979-7f22bf599495" /> <br>
 print("Misclassified Samples : %d" % (test_y !=prediction).sum()) <br>
<img width="902" height="111" alt="image" src="https://github.com/user-attachments/assets/5780dfb3-8caa-4336-878a-93c2ddb5a483" /> <br>
data.shape <br>

<img width="873" height="97" alt="image" src="https://github.com/user-attachments/assets/89f39b53-5a66-4903-a44d-fe7210865d3e" /> <br>
import pandas as pd <br>
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif <br>
data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 } <br>
df=pd.DataFrame(data) <br>
x=df[['Feature1','Feature3']] <br>
y=df[['Target']] <br>
selector=SelectKBest(score_func=mutual_info_classif,k=1) <br>
x_new=selector.fit_transform(x,y) <br>
selected_feature_indices=selector.get_support(indices=True) <br>
selected_features=x.columns[selected_feature_indices]<br>
print("Selected Features:")<br>
print(selected_features)<br>

<img width="907" height="596" alt="image" src="https://github.com/user-attachments/assets/218915e9-99b4-4cab-aea8-41704fd9eecd" /><br>
import pandas as pd<br>
import numpy as np<br>
from scipy.stats import chi2_contingency<br>
import seaborn as sns<br>
tips=sns.load_dataset('tips')<br>
tips.head()<br>

<img width="825" height="382" alt="image" src="https://github.com/user-attachments/assets/807623e1-25ff-4484-8852-d304ad0602d2" /><br>
 tips.time.unique()<br>

 <img width="920" height="122" alt="image" src="https://github.com/user-attachments/assets/facd7bc6-e300-4730-a81f-81f65e40ce6c" /><br>
 contingency_table=pd.crosstab(tips['sex'],tips['time'])<br>
 print(contingency_table)<br>

 <img width="934" height="190" alt="image" src="https://github.com/user-attachments/assets/08801f2c-49ec-4b90-8093-8cb008ccf27c" /><br>
 chi2,p,_,_=chi2_contingency(contingency_table)<br>
 print(f"Chi-Square Statistics: {chi2}")<br>
 print(f"P-Value: {p}")<br>

 <img width="940" height="172" alt="image" src="https://github.com/user-attachments/assets/c94183ea-15c4-4970-be3a-de6fcc456dc8" /><br>

# RESULT:
       # INCLUDE YOUR RESULT HERE
