   # Hello wrold... 
   # Lets code the program ---> 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from icecream import ic 
from pathlib import Path 
import warnings 
import os,time,joblib
    
    # Lets laod the dataset --> 
data_path = Path('/content/road_accident_dataset.csv')
if data_path.exists():
    data = pd.read_csv(data_path)
    ic('data load sucessfully ... ')
else:
    print(f"{data_path} does not exist")

    # lets check the data --> 
ic(data.info())  #  132000 entries and float64(7), int64(9), object(14) 
ic(data.isnull().sum())  # No NaN value in dataset 
ic(data.describe())
   #Lets  build the function for detacting the outliers ---> 
def outliers_detect(data):
    outliers = {}
    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_data = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
        
        if not outliers_data.empty:
            outliers[column] = outliers_data.tolist()  # Store outliers in a dictionary
    return outliers
outliers = outliers_detect(data)
print(outliers)  
   # Lets detact the outliers with Graphs ---> 
   # Check and Deleted .. Confirm there are No outliers 
   # Lets encode the data ---> 

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
encoder = LabelEncoder()
data_columns_01 = data.select_dtypes(include=['object']).columns
output_01 = pd.DataFrame()
for col in data_columns_01:
    output_01[col + '_en'] = encoder.fit_transform(data[col])
    ic(output_01.head(1))

scalor_01 = MinMaxScaler()
data_columns_02 = data.select_dtypes(include=['float64']).columns
output_02 = pd.DataFrame(index=data.index)  
for sal in data_columns_02:
    output_02[sal + '_sc'] = scalor_01.fit_transform(data[[sal]])
    ic(output_02.head(1))

scalor_02 = StandardScaler()
data_columns_03 = data.select_dtypes(include=['int64']).columns
output_03 = pd.DataFrame(index=data.index)  
for sal in data_columns_03:
    output_03[sal + '_sc'] = scalor_02.fit_transform(data[[sal]])
    ic(output_03.head(1))

concat_data = pd.concat([output_01, output_02, output_03], axis=1)
concat_data.to_csv('Final_data.csv', index=False)
concat_data

   # lets load the baianry data --> 
df = pd.read_csv('Final_data.csv')
ic(df.head(5)) 
ic(df.isnull().sum()) 
ic(df.info())
df.columns # Country_en , ,Driver Age Group_en ,Vehicle Condition_en ,
           # Road Condition_en ,Driver Alcohol Level_sc ,Traffic Volume_sc ,Speed Limit_sc
           # Economic Loss_sc  and y= (Insurance Claims_sc,Economic Loss_sc) 

    # Lets train the features ---> 
x = df[['Country_en','Driver Age Group_en', 'Vehicle Condition_en', 
        'Road Condition_en', 'Driver Alcohol Level_sc', 'Traffic Volume_sc', 'Speed Limit_sc']]
y = df[['Insurance Claims_sc', 'Economic Loss_sc']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

   #Lets check the datas corr with x_train to Y_train ---> 
corr_matrix = pd.concat([x_train, y_train], axis=1).corr()  # Combine x_train and y_train, then calculate correlation
corr_with_y = corr_matrix[['Insurance Claims_sc', 'Economic Loss_sc']]  # Get correlation with y_train columns
print(corr_with_y)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_with_y, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation between x_train features and y_train')
plt.show()

for column in x_train.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=x_train[column], y=y_train['Insurance Claims_sc'], label='Insurance Claims_sc')
    sns.scatterplot(x=x_train[column], y=y_train['Economic Loss_sc'], label='Economic Loss_sc', color='red')
    plt.xlabel(column)
    plt.ylabel('y_train')
    plt.title(f'{column} vs Insurance Claims_sc and Economic Loss_sc')
    plt.legend()
    plt.show()
    # There are No connection between the featues,
    # So lets GO for Non linear regression 
       # Lets build with Neural network --->
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Model Architecture
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # For regression, change to 'sigmoid' for classification
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
history = model.fit(x_train, y_train, epochs=100, batch_size=42, validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint], verbose=2)
model.summary()
joblib.dump(model.get_weights(), 'mode.pkl')
model.set_weights(joblib.load('Neural_netwrok_model.pkl'))

  