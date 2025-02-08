
#Hello wrold ...
   #LETS code the program -->
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os,time,sys
from pathlib import Path
from icecream import ic

  #LETS load the data -->
data_path = Path('/content/laptop_data.csv')
if data_path.exists():
  data=pd.read_csv(data_path)
  ic('data is loaded sucessfully...')
else :
  raise FileNotFoundError(f'This file path is not founded ...')

  #LETS check the data -->
ic(data.info()) #1303 entries and  float64(2), int64(1), object(9)
ic(data.isnull().sum())  #No NaN values in dataset -->
ic(data.describe())

 #LETS create a function that will return the outliers columns -->

def outliers_detection_columns(data):
    outliers_columns = []
    data_columns_outliers = data.select_dtypes(include=['number']).columns
    for out in data_columns_outliers:
        Q1 = data[out].quantile(0.25)
        Q3 = data[out].quantile(0.75)
        IQR = Q3 - Q1
        min_lower_bound = Q1 - (1.5 * IQR)
        max_higher_bound = Q3 + (1.5 * IQR)

        outliers = data[(data[out] < min_lower_bound) | (data[out] > max_higher_bound)]
        if not outliers.empty:
            outliers_columns.append(out)
    return outliers_columns
outliers_detection_columns(data) # ['Inches', 'Price'] columns have outliers

    #Lets check with graph -->
plt.figure(figsize=(6,4))
sns.boxplot(x=data['Inches'], color='r', flierprops=dict(markerfacecolor='r', marker='o'))
plt.title('This is Inches outliers')
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data['Price'], color='r', flierprops=dict(markerfacecolor='r', marker='o'))
plt.title('This is Price outliers')
plt.grid(True)
plt.show()


   #LETS clear this Outliers
Q1_01 = data['Inches'].quantile(0.25)
Q3_01 = data['Inches'].quantile(0.75)
IQR_01= Q3_01 - Q1_01
min_range_01 = Q1_01 - (1.5 * IQR_01)
max_range_01 = Q3_01 + (1.5 * IQR_01)
data = data[(data['Inches'] < min_range_01) | (data['Inches'] > max_range_01)]
ic(data['Inches'].describe())

Q1_02 = data['Price'].quantile(0.25)
Q3_02 = data['Price'].quantile(0.75)
IQR_02= Q3_02 - Q1_02
min_range_02 = Q1_02 - (1.5 * IQR_02)
max_range_02 = Q3_02 + (1.5 * IQR_02)
data = data[(data['Price'] < min_range_02) | (data['Price'] > max_range_02)]
ic(data['Price'].describe())
data.drop_duplicates(inplace=True)

ic(data.isnull().sum())

   #The outliers is cleared ..
   #LETS GO for encoding ---->
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data_columns_01 = data.select_dtypes(include=['object']).columns
output_01 = pd.DataFrame(index=data.index)
for col in data_columns_01:
  output_01[col+"_n"] = encoder.fit_transform(data[col])
  ic(output_01)

from sklearn.preprocessing import MinMaxScaler
scalor_01= MinMaxScaler()
data_columns_02 =data.select_dtypes(include=['float64']).columns
output_02 =pd.DataFrame(index=data.index)
for col in data_columns_02:
  output_02[col+"_scl"]= scalor_01.fit_transform(data[[col]]).reshape(-1,1)
  ic(output_02)

from sklearn.preprocessing import StandardScaler
scalor_02= StandardScaler()
data_columns_03 =data.select_dtypes(include=['int64']).columns
output_03 =pd.DataFrame(index=data.index)
for col in data_columns_03:
  output_02[col+"_ss"]= scalor_01.fit_transform(data[[col]])
  ic(output_03)

concat_data =pd.concat([output_03,output_01,output_02],axis=1)
concat_data.to_csv('fresh_data.csv',index=False)
ic('fresh data creating sucessfully...')

 #LETS laod the new data -->
df =pd.read_csv('fresh_data.csv')
ic(df.head(5))
ic(df.isnull().sum())
ic(df.describe())

  #LETS select the features --->
x= df.iloc[:,:-2]
y= df['Price_scl']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=45,shuffle=True)
ic(x_train.shape,x_test.shape)
ic(y_train.shape,y_test.shape)

  #LETS check the data corr relation between the featrues-->
corr_matrix = pd.concat([x_train, y_train], axis=1).corr()
corr_with_y = corr_matrix[['Price_scl']]  # Get correlation with y_train columns
print(corr_with_y)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_with_y, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation between x_train features and y_train')
plt.show()

plt.figure(figsize=(10, 8))
for feature in x_train.columns:
    sns.scatterplot(data=x_train, x=feature, y=y_train, label=feature)
plt.title('Scatterplot of Features vs Target')
plt.legend()
plt.show()

  #LETS Build the model -->
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
model_accuracy = lr.score(x_train, y_train)
print(f'Model Accuracy (R² Score): {model_accuracy:.4f}')

test_accuracy = lr.score(x_test, y_test)
print(f'Test Accuracy (R² Score): {test_accuracy:.4f}') # 1.0 accuacy

import joblib
joblib.dump(lr,'model_laptops.h5')

