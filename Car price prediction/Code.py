   # Hello wrold 
   # Lets code the program .... 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os, time
import warnings 
from icecream import ic 
from pathlib import Path 
from tqdm import tqdm
    
   # Lets load the dataset ----> 
data_path = Path('/content/CarPrice_Assignment.csv')
if data_path.is_file():
    df = pd.read_csv(data_path)
    ic('Data load succesfuly.....')
else:
    FileNotFoundError(f'The file path{data_path} doest founded .....')
          # lets check the datas objects and work On it --> 
ic(df.head(5))
ic(df.info())  #  205 entries and float64(8), int64(8), object(10)
ic(df.isnull().sum()) # No NaN values in data 
ic(df.describe())  # wheelbase,carlength,carlength,carlength,carlength columns might have outliers 
   #Lets detact the outliers and remove the outliers 
def detect_outliers(df):
    outlier_columns = []
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if there are any outliers in the column
        if len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]) > 0:
            outlier_columns.append(col)
    
    return outlier_columns

outlier_columns = detect_outliers(df)
print("Columns with outliers:", outlier_columns) 

    # this columns have outliers -> 
    #[['wheelbase', 'carlength', 'carwidth','enginesize', 'stroke', 'compressionratio',
    #  'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']]

    # Lets remove the Outliers except price columns --->

def remove_outliers(df):
    for col in df.select_dtypes(include=['number']).columns:  # Process only numeric columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.drop(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index, inplace=True)
    ic("Outliers removed successfully!")
remove_outliers(df)  # This modifies 'df' directly
ic(df.describe())  # Check the modified dataset
ic(df.head(10))


   # lets encode the data ---> 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_columns_01 = df.select_dtypes(include=['object']).columns
for col in data_columns_01:
    df[col] = le.fit_transform(df[col])
ic(df.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_columns_02 = df.select_dtypes(include=['float64','int64']).columns
df[data_columns_02] = scaler.fit_transform(df[data_columns_02])
ic(df.head())

ic(df.head(10))
   #Lets build this model ---> 
from sklearn.model_selection import train_test_split
from sklearn.cluster import dbscan
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
   
   # Our independent verriable is -1(price) 
x = df.iloc[:, :-1]
y = df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41,shuffle=True)
ic(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
   
   #Lets Build the model ---> 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
   
   # Lets evalute the model ---> 
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ic("Mean Squared Error:", mse)
ic("model_accuracy: ", r2)

   # mse: 0.05786932313630812 'model_accuracy: ', r2: 0.9353853407358054
   # model_accuracy: ', 0.9353853407358054
   # model_accuracy ---> 93% 