   #Hello wrold ... 
   #LETS CODE THE PROGRAM ---> 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os,time,sys,warnings
from pathlib import Path 
from icecream import ic 

  #LETS load the data --> 
data_path = Path('/content/model_dataset.xlsx')
if data_path.exists():
  data =pd.read_excel(data_path)
  ic('data load sucessfully...')
else : 
  raise FileNotFoundError(f'This file path{data_path}doest exists...')

  #LETS check the data strcuture --> 
ic(data.isnull().sum()) #NO NaN values 
ic(data.info()) # 6109 entries and float64(7), int64(13) also y= Exam_Score
ic(data.describe())
  
  #LETS Create a function to detact the outliers 
def detect_outliers(data):
    outlier_columns = []
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        
        if not outliers.empty:  # If outliers exist, add column name
            outlier_columns.append(col)
    
    return outlier_columns

# Example usage
outlier_cols = detect_outliers(data)
print("Columns with outliers:", outlier_cols)

 # Columns with outliers: ['Exam_Score', 'Internet_Access', 'Learning_Disabilities'] 
plt.figure(figsize=(5,8))
sns.boxenplot(data=data[outlier_cols])
plt.show()
 #LETS remove the outliers ---> 

Q1= data['Exam_Score'].quantile(0.25)
Q3= data['Exam_Score'].quantile(0.95)  #95% is good for 8.1772 
IQR = Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Exam_Score'] >= lower_bound) & (data['Exam_Score'] <= upper_bound)]
ic(data['Exam_Score'].describe())

  #LETS DO encoding ---> 
from sklearn.preprocessing import MinMaxScaler
scalor_01 = MinMaxScaler()
data_columns_01 = data.select_dtypes(include=['float64']).columns
output_01 = pd.DataFrame(index=data.index) 
for col in data_columns_01:
  output_01[col] = scalor_01.fit_transform(data[[col]]).reshape(-1,1)
output_01['Exam_Score'] = data['Exam_Score']
ic(output_01.head(5))


from sklearn.preprocessing import StandardScaler
scalor_02 = StandardScaler()
data_columns_02 = data.select_dtypes(include=['int64']).columns
output_02 = pd.DataFrame(index=data.index) 
for col in data_columns_02:
  output_02[col] = scalor_02.fit_transform(data[[col]])
ic(output_02.head(5))

concat_data=pd.concat([output_02,output_01],axis=1)
concat_data.to_csv('student_model_dataset.csv',index=False)
concat_data

   #LETS LOAD THE FRESH DATA --> 
df = pd.read_csv('/content/student_model_dataset.csv')
df.head(5)
ic(df.describe())  #NO outlies
ic(df.info()) # 6074 entries,float64(20),memory usage: 949.2 KB 
ic(df.info()) #NO overfit data . 
   
   #LETS select the features ---> 
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y = df['Exam_Score']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
ic(X_train.shape,X_test.shape,y_train.shape,y_test.shape)  
  
    #X_train.shape: (4859, 19)
    #X_test.shape : (1215, 19)
    #y_train.shape: (4859,)
    #y_test.shape : (1215,0 ) 

  #LETS check the datas corr relation --> 
sub_data = data.corr() 
ic(sub_data) 
  
  # Given correlation values
corr_values = [
    0.494744, 0.671425, -0.016597, 0.200739, 0.144653, 0.032166, 1.000000, 
    -0.092112, -0.104253, 0.074095, -0.015624, 0.068067, -0.017811, -0.064297, 
    -0.011560, 0.118478, -0.112164, 0.063053, 0.101894, 0.016724
]

# Compute the mean of absolute values and convert to percentage
linearity_percentage = (sum(abs(val) for val in corr_values) / len(corr_values)) * 100
linearity_percentage  #17.102279999999997 

  #LETS build the model --> 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ic("Mean Squared Error:", mse)
ic("R-squared:", r2)
test_r2 = model.score(X_test, y_test)
print("Testing R-squared:", test_r2)  #83% 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

   # Define the model
neural_network = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

  # Compile the model
neural_network.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = neural_network.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],verbose=1)

   # Evaluate the model
y_pred_01 = neural_network.predict(X_test)
mse_01 = mean_squared_error(y_test, y_pred_01)
r2_01 = r2_score(y_test, y_pred_01)

print("Mean Squared Error:", mse_01)
print("R-squared:", r2_01) #88% 

   #LETS load the model in file ---> 
neural_network.save('model.h5')  # Save

  

