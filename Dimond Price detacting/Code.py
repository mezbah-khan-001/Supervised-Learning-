    #Hello wrold ....
   #Lets code the program ---->
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from pathlib import Path
import os,time,functools
import warnings 
from icecream import ic
   
   # lets code --> 
data_path =Path('/content/diamonds.csv')
if data_path.exists(): 
  data = pd.read_csv(data_path)
  print('data load sucessfully.....')
else : 
  raise FileNotFoundError(f'This file path {data_path} doest founeded ......')

  # lETS check the data --> 
ic(data.info()) #  53940 entries and float64(6), int64(1), object(3) ..
ic(data.isnull().sum()) # No NaN value in dataset 
ic(data.describe())  #Every columns have outliers 
  
plt.figure(figsize=(10,5)) 
sns.boxenplot(x='carat',data=data,color='red')
plt.show() # Have outliers 

plt.figure(figsize=(10,5)) 
sns.boxenplot(x='depth',data=data,color='red')
plt.show() # Have outliers

plt.figure(figsize=(10,5)) 
sns.boxenplot(x='table',data=data,color='red')
plt.show() # Have outliers

plt.figure(figsize=(10,5)) 
sns.boxenplot(x='price',data=data,color='red')
plt.show() # No outliers  ---> 
    
    #so We 3 columns of Outliers [carat,depth ,table ]
    #lets create a function that handover us the columns of Outliers 

def outliers_detection(data): 
    data_columns = data.select_dtypes(include='number').columns 
    outliers_columns = []

    for column in data_columns: 
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if ((data[column] < lower_bound) | (data[column] > upper_bound)).any():
            outliers_columns.append(column)
    return outliers_columns
outliers_columns = outliers_detection(data)
ic(outliers_columns) 
   
   #LETS remove the outliers --> 
def remove_outliers(data): 
    data_columns = data.select_dtypes(include='number').columns
    for column in data_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

data = remove_outliers(data)
ic(data.info())
ic(data.describe())  

   #The outliers is removed... 
   #lets encode the data 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data_columns_01 = data.select_dtypes(include=['object']).columns
output_01 = pd.DataFrame(index=data.index) 
for col in data_columns_01: 
  output_01[col +"en"] = label_encoder.fit_transform(data[col])
ic(output_01.head(2))

from sklearn.preprocessing import MinMaxScaler
scaler_01 = MinMaxScaler()
data_columns_02 = data.select_dtypes(include=['float64']).columns
output_02 = pd.DataFrame(index=data.index)
for col in data_columns_02: 
  output_02[col] = scaler_01.fit_transform(data[[col]])
ic(output_02.head(2))

from sklearn.preprocessing import StandardScaler
scaler_02 = StandardScaler()
data_columns_03 = data.select_dtypes(include=['int64']).columns
output_03 = pd.DataFrame(index=data.index)
for col in data_columns_03: 
  output_03[col] = scaler_02.fit_transform(data[[col]])
  ic(output_03.head(2))
   
   #lets concat the outputs and add them into fresh data --> 
concat_data= pd.concat([output_01,output_02,output_03],axis=1)
concat_data.to_csv('Fresh_data.csv',index=False)
df = pd.read_csv('Fresh_data.csv')
ic(df.isnull().sum())
ic(df.info())
ic(df.describe())  


   # Lets load train the features ---> 
from sklearn.model_selection import train_test_split
x = df.iloc[:,: -1]
y = df['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=43,shuffle=True)
ic(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
   
   #lETS check the corr relation between the features --> 
corr_matrix = pd.concat([x_train, y_train], axis=1).corr()  # Combine x_train and y_train, then calculate correlation
corr_with_y = corr_matrix[['price']]  # Get correlation with y_train columns
print(corr_with_y)

plt.figure(figsize=(10,5))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.show()
   
   # There are few connection between the features 
   # lets build the model with neural network --> 
import tensorflow as tf 
from tensorflow.keras.layers import Dense,Input,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True) 
history = model.fit(x_train,y_train,validation_split=0.2,epochs=50,batch_size=40,callbacks=[early_stopping])
history_df = pd.DataFrame(history.history)
ic(history_df.head())
  
  #lets check the model accuracy --> 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
ic(mae,mse,r2)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
ic(mae,mse,r2)  
   
    # mae: 0.262813514163335
    # mse: 0.1272303427705885
    # r2: 0.8690098495040957
    # The model is perfoming with 88% accuracy ---> 
    # LETS save the model into a file 

import joblib 
joblib.dump(model,'diamond_price_model.joblib')

