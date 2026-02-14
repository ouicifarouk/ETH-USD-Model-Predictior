# 📈 ETH-USD Price Predictor (High/Low)

This is a supervised machine learning model built with **Linear Regression** to predict the hourly High and Low prices of ETH against USD.

## 🚀 Model Specifications
* **Data Source:** Real-time hourly data for the last 23 months fetched via `yfinance`.
* **Update Frequency:** Hourly.
* **Accuracy:** * **RMSE (Average Squared Error):** ~15
    * **Standard Deviation:** ~3

## ⚠️ Disclaimer
**Despite its accuracy, this model cannot be relied upon for managing financial risks or predicting prices for trading.** Anyone using this model bears full responsibility for any financial losses. It is recommended for educational purposes only.

## 🧪 Alternative Models Tested
We included tests for other models that yielded worse results for this specific time-series task:
* **Random Forest Regressor**
* **Decision Tree Regressor**
*Note: You can swap `LinearRegression()` with these models in the code to compare performance.*
