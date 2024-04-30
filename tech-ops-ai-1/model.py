# model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

origins = ['Nairobi', 'Mombasa', 'Mandera']
destinations = ['Miami', 'Dallas', 'Atlanta', 'Seattle', 'Boston']
parcel_type=["Delicate", "Normal"]
delivery_services=["Express", "Standard"]

data = {
    'Origin': np.random.choice(origins, size=1000),
    'Destination': np.random.choice(origins, size=1000),
    'parcel_weight': np.random.randint(0,10, size=1000),
    'Parcel_type': np.random.choice(parcel_type, size=1000),
    'Delivery_Service': np.random.choice(delivery_services, size=1000),
    'is_holiday_or_weekend': np.random.randint(0,1, size=1000),
    'previous_orders': np.random.randint(500, size=1000),
    'price': np.random.randint(3000, size=1000)
}

distances={
    'Nairobi_Mombasa':488.1,
    'Mombasa_Nairobi':488.1,
    'Mombasa_Mandera':1121.4,
    'Mandera_Mombasa':1121.4,
    'Nairobi_Mandera':1025.6 ,
    'Mandera_Nairobi':1025.6,
    'Nairobi_Nairobi':0,
    'Mombasa_Mombasa':0,
    'Mandera_Mandera':0
}

df = pd.DataFrame(data)

df['Parcel_type'] = label_encoder.fit_transform(df['Parcel_type'])
df['Delivery_Service'] = label_encoder.fit_transform(df['Delivery_Service'])


df['Distance']=[0 for x in range(len(df))]

for index,row in df.iterrows():
  name=f"{row['Origin']}_{row['Destination']}"
  df['Distance'].iloc[index] = distances[name]

df.drop(inplace=True,columns=['Origin','Destination'])

# df.to_csv('./data.csv')


# df = pd.read_csv('./data.csv')

# Split features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Train a linear regression model
model = LinearRegression()
model.fit(X.values, y.values)

# Save the model
joblib.dump(model, 'model.joblib')
