 #Hello world ....
  # lets code the program ---> 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score
import joblib  

  # Load the dataset --> 
data_path = Path('/content/updated_version.csv')
if data_path.exists():
    data = pd.read_csv(data_path)
    ic('Data loaded successfully...')
else:
    raise FileNotFoundError(f'The file path {data_path} does not exist...')

  # Check the data --> 
ic(data.info())
ic(data.describe())

  # Detect and remove outliers --> 
def detect_outliers(data):
    outlier_columns = []
    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if data[column].lt(lower_bound).sum() > 0 or data[column].gt(upper_bound).sum() > 0:
            outlier_columns.append(column)
    return outlier_columns

outliers_columns = detect_outliers(data)
ic(outliers_columns)

def remove_outliers(data):
    data = data.copy()
    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

data = remove_outliers(data)

  # Encode the data in baianry formate and save to another file --> 
scalor_01 = MinMaxScaler()
data_columns_01 = data.select_dtypes(include=['float64']).columns
output_01 = pd.DataFrame(index=data.index)
for col in data_columns_01:
    output_01[col + 'en'] = scalor_01.fit_transform(data[[col]])
    
scalor_02 = StandardScaler()
data_columns_02 = data.select_dtypes(include=['int64']).columns
output_02 = pd.DataFrame(index=data.index)
for col in data_columns_02:
    output_02[col + 'en'] = scalor_02.fit_transform(data[[col]])

  # Combine the scaled data..
concat_data = pd.concat([output_01, output_02], axis=1)
concat_data.to_csv('Systemetical_data.csv')

   # Train the model for model --> 
df = pd.read_csv('Systemetical_data.csv')
x = df.iloc[:, :-1]
y = df['heart_attacken']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

   # Checking data features linearity --> 
sub_data = x_train.copy()
plt.figure(figsize=(6, 6))
sns.pairplot(sub_data)
plt.show()

   # Build the Neural Network model --> 
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2,)

# Evaluate the model
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}%")

# Save the model using joblib (for TensorFlow model, usually 'save' is preferred but here is how to do it with joblib)
joblib.dump(model, 'Neural_network_model.joblib')  # Save the model using joblib

# Load the model using joblib (for TensorFlow, using joblib is not typical, but for other models it's common)
loaded_model = joblib.load('Neural_network_model.joblib')

# Evaluate the loaded model
y_pred_loaded = loaded_model.predict(x_test)
y_pred_loaded = (y_pred_loaded > 0.5)  # Convert probabilities to binary labels

accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"Accuracy of the loaded model: {accuracy_loaded:.4f}")

# Optionally, save the TensorFlow model as well (using .h5 format)
model.save('Neural_network_model.h5')
