   #Hello world ...
   #Lets code the program --->
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time,os,functools
from pathlib import Path
from icecream import ic
import warnings
   
   #lets build this --> 
data_path = Path('/content/anxiety_attack_dataset.csv')
if data_path.is_file():
  data = pd.read_csv(data_path)
  ic('data load sucessfully...')
else:
  print(f"{data_path} does not exist")

  #Lets check the data --> 
ic(data.head(5))
ic(data.info())  # float64(2), int64(11), object(7)  and  12000 entries 
ic(data.isnull().sum()) # No NaN Values 
ic(data.describe())  #detact the outliers with functions 

def detect_outliers(data):
    data_columns = data.select_dtypes(include=['number']).columns
    outlier_columns = []
    for col in data_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if data[(data[col] < lower_bound) | (data[col] > upper_bound)].shape[0] > 0:
            outlier_columns.append(col)
    return outlier_columns
outliers = detect_outliers(data)
print("Columns with outliers:", outliers)  #There are No outliers 
  #Lets encode the data --> 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
output_01 = pd.DataFrame(index=data.index)
data_columns_01 = data.select_dtypes(include=['object']).columns
for col in data_columns_01:
    output_01[col + '_en'] = label_encoder.fit_transform(data[col])
print("Encoded DataFrame:\n", output_01)

from sklearn.preprocessing import StandardScaler
scalor = StandardScaler() 
data_columns_02 = data.select_dtypes(include=['int64']).columns 
output_02 = pd.DataFrame(index=data.index)
for col in data_columns_02:
    output_02[col + '_sc'] = scalor.fit_transform(data[col].values.reshape(-1, 1))
print("Scaled DataFrame:\n", output_02) 

from sklearn.preprocessing import MinMaxScaler
scalor_01= MinMaxScaler()
data_columns_03 = data.select_dtypes(include=['float64']).columns
output_03 = pd.DataFrame(index=data.index)
for col in data_columns_03:
    output_03[col + '_sc'] = scalor_01.fit_transform(data[col].values.reshape(-1, 1))
    print("Scaled DataFrame:\n", output_03)

# Concatenating all the processed DataFrames
concat_data = pd.concat([output_01, output_02, output_03], axis=1)
concat_data.to_csv('Fresh_data.csv', index=False)
  #lets load the data ---> 
df = pd.read_csv('/content/Fresh_data.csv')
df.head(2) 
ic(df.isnull().sum())
ic(df.info()) # 12000 entries and 
ic(df.describe()) # float64(13), int64(7)
 #LETS TRAIN The features ---> 
df.columns
x = df[['Gender_en','Smoking_en','Family History of Anxiety_en',
        'Medication_en','Alcohol Consumption (drinks/week)_sc',
        'Heart Rate (bpm during attack)_sc','Sleep Hours_sc']]
y = df[['Severity of Anxiety Attack (1-10)_sc']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
ic(x_train.shape,x_test.shape,y_train.shape,y_test.shape) 
   
   #LETS check the corr relation between the x_train and y_train ---> 
corr_matrix = pd.concat([x_train, y_train], axis=1).corr()  
corr_with_y = corr_matrix[['Severity of Anxiety Attack (1-10)_sc']]  
print(corr_with_y)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_with_y, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation between x_train features and y_train')
plt.show()
   #There Are No relation between the features 
   # So WE Have to go for Non linear algorithoms ---> 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
model_01 = RandomForestRegressor(n_estimators=100, random_state=42,)
model_01.fit(x_train, y_train)
y_pred = model_01.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")  # The score is TO low 
    #LETS Move to Neural network 
       #LETS Build this neurons ---> 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Define the model
model = Sequential([
    Input(shape=(7,)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model (Assuming x_train and y_train are defined)
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) 
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}") 

# Save the model
model.save('anxiety_attack_model.h5')

