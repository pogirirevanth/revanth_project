
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset from your data folder
# Note: Using the path shown in your screenshot
df = pd.read_csv('data/startup_data.csv')
# 2. Data Cleaning & Pre-processing (Based on your analysis)
# Drop redundant columns like state_code.1 and irrelevant industry flags
cols_to_drop = ['Unnamed: 0', 'state_code.1', 'object_id', 'id', 'zip_code', 'city', 'name', 'labels']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Reduce State Categories to Top 5 + 'other'
top_5_states = ['CA', 'NY', 'MA', 'TX', 'WA']
df['state_code'] = df['state_code'].apply(lambda x: x if x in top_5_states else 'other')

# Convert status (Acquired/Closed) to 1/0
df['status'] = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)

# 3. Feature Selection
# Choosing the specific features you analyzed (milestones, relationships, funding)
features = ['milestones', 'relationships', 'funding_rounds', 'funding_total_usd']
X = df[features]
y = df['status']

# 4. Train-Test Split (70/30 split as per your plan)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=116)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=116)
model.fit(X_train_scaled, y_train)

# 7. Save the Model and Scaler (Serialization)
# These files will appear in your main folder
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Success! model.pkl and scaler.pkl have been created.")