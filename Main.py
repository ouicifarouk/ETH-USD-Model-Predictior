import numpy as np
import pandas as pd
import yfinance as yf
import joblib 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# 1. Getting The Data Ready
# Fetching live data for the last 23 months with 1-hour interval
print("Fetching data from Yahoo Finance... ⏳")
ETH_data = yf.download('ETH-USD', period='23mo', interval='1h')
cdf = pd.DataFrame(ETH_data)

# 2. Splitting Data
# Using shuffle=False because this is time-series data (past predicts future)
train_set, test_set = train_test_split(cdf, test_size=0.2, shuffle=False)
print(f"Data split completed. Training samples: {len(train_set)}")

# 3. Prepare Features (X) and Labels (y)
# We drop 'Close' to avoid future leakage
features = train_set.drop(['High', 'Low', 'Close'], axis=1)
labels = train_set[['High', 'Low']]

# 4. Building the Pipeline (Cleaning + Scaling)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# Process the training data
features_prepared = num_pipeline.fit_transform(features)

# 5. Training the Model
print("Training Linear Regression Model... 🤖")
lin_reg = LinearRegression()
lin_reg.fit(features_prepared, labels)

# 6. Evaluate on Training Data
lin_pred = lin_reg.predict(features_prepared)
lin_mse = mean_squared_error(labels, lin_pred)
lin_rmse = np.sqrt(lin_mse)
print(f'Training RMSE: {lin_rmse}')

# 7. Cross-Validation (Checking for Overfitting)
print("Running Cross-Validation... ⚔️")
scores = cross_val_score(lin_reg, features_prepared, labels, 
                         scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)

print(f'Cross-Validation Mean RMSE: {rmse_scores.mean()}')
print(f'Standard Deviation: {rmse_scores.std()}')

# 8. Saving the Model
joblib.dump(lin_reg, 'ETH_Model.pkl')
print("Model saved successfully as 'ETH_Model.pkl' ✅")
