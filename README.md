# 📈 Time Series Forecasting of AAL Stock Prices Using ARIMA and Deep Learning Models (2010–2025)

A comprehensive time series analysis and forecasting project comparing traditional statistical methods (ARIMA, SARIMA, SARIMAX) with modern deep learning architectures (RNN, LSTM, GRU) for predicting American Airlines (AAL) stock opening prices.


## 📋 Table of Contents

Overview
Features
Installation
Dataset
Methodology
Results
Project Structure
Usage
Key Findings
Future Improvements
Contributing
License

# 🎯 Overview
This project implements and compares multiple forecasting approaches for AAL stock prices spanning from 2010 to 2025. It demonstrates the complete workflow of time series analysis including:

``` Data acquisition and preprocessing
Statistical testing and validation
Classical decomposition and STL analysis
ARIMA family models (ARIMA, SARIMA, SARIMAX)
Deep learning models (RNN, LSTM, GRU)
90-day future forecasting
Comprehensive model evaluation
```
# ✨ Features

```
Data Collection: Automated stock data retrieval using Yahoo Finance API
Statistical Analysis:

Stationarity tests (ADF, KPSS)
Autocorrelation analysis (ACF, PACF)
Ljung-Box test
Classical and STL decomposition


Traditional Models: ARIMA, SARIMA, SARIMAX implementations
Deep Learning Models:

Simple RNN with 128-64-32 architecture
LSTM with dropout regularization
GRU with optimized gates


Evaluation Metrics: RMSE, MAPE, SMAPE, MSLE
Visualization: Comprehensive plots for all stages of analysis
Future Forecasting: 90-day ahead predictions
```

# 🔧 Installation
Prerequisites
bashpython >= 3.8
pip >= 21.0
Required Libraries
bashpip install pandas numpy matplotlib seaborn scipy
pip install statsmodels scikit-learn tensorflow keras
pip install yfinance yahooquery
Or install all dependencies at once:
bashpip install -r requirements.txt
```

### Requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
tensorflow>=2.6.0
keras>=2.6.0
yfinance>=0.1.70
yahooquery>=2.3.0

# 📊 Dataset

Source: Yahoo Finance via yahooquery API
Ticker: AAL (American Airlines Group Inc.)
Period: 1990-01-01 to 2025 (dynamically updated)
Feature: Opening price (normalized using MinMaxScaler)
Split: 75% training, 25% testing
Validation: 20% of training data used for validation

# 🔬 Methodology
1. Data Preprocessing
```
Data extraction from Yahoo Finance
MinMax normalization (0-1 range)
Train-test split with temporal ordering preserved
```
2. Statistical Analysis
```
python- Descriptive statistics (mean, std, skewness, kurtosis)
- Stationarity tests (ADF and KPSS)
- Differencing for non-stationary series
- ACF/PACF analysis for parameter selection
```

### 3. Classical Models
- **ARIMA(2,1,2)**: Basic autoregressive integrated moving average
- **SARIMA(2,1,1)x(2,1,1,s)**: Seasonal extension with period s
- **SARIMAX**: Includes exogenous variables (time trend, moving averages)

### 4. Deep Learning Architecture

**Common Configuration:**
- Lookback window: 60 days
- Training epochs: 100 (with early stopping)
- Batch size: 32
- Optimizer: Adam
- Loss function: Mean Squared Error

**Simple RNN:**
```
SimpleRNN(128) → Dropout(0.2) → SimpleRNN(64) → Dropout(0.2) 
→ SimpleRNN(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)
```

**LSTM:**
```
LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) 
→ LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)
```

**GRU:**
```
GRU(128) → Dropout(0.2) → GRU(64) → Dropout(0.2) 
→ GRU(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)
```

## 📈 Results

### Model Performance Comparison (Test Set)

| Model   | RMSE ↓     | MAPE (%) ↓ | SMAPE (%) ↓ | MSLE ↓       |
|---------|------------|------------|-------------|--------------|
| RNN     | 0.0215     | 8.90       | 8.83        | 0.000316     |
| LSTM    | 0.0124     | 5.05       | 4.87        | 0.000106     |
| **GRU** | **0.0113** | **4.38**   | **4.35**    | **0.000087** |

### Key Insights

✅ **GRU emerged as the best performer** with:
- Lowest error across all metrics
- Best balance between complexity and performance
- Faster training compared to LSTM

✅ **LSTM performed competitively** with:
- Marginally higher error than GRU
- Better long-term dependency capture than RNN
- More parameters leading to slightly higher complexity

✅ **Simple RNN lagged behind** due to:
- Vanishing gradient problems
- Limited ability to capture long-term patterns
- Not recommended for financial forecasting

## 📁 Project Structure
```
aal-stock-forecasting/
│
├── data/
│   └── aal_stock_data.csv          # Downloaded stock data
│
├── models/
│   ├── rnn_model.h5                # Trained RNN model
│   ├── lstm_model.h5               # Trained LSTM model
│   └── gru_model.h5                # Trained GRU model
│
├── plots/
│   ├── classical_decomposition.png
│   ├── stl_decomposition.png
│   ├── stationarity_transformation.png
│   ├── acf_pacf_plots.png
│   ├── moving_average.png
│   ├── arima_forecast.png
│   ├── sarima_forecast.png
│   ├── sarimax_forecast.png
│   └── model_comparison.png
│
├── notebooks/
│   └── AAL_Stock_Forecasting.ipynb # Main Jupyter notebook
│
├── src/
│   ├── data_loader.py              # Data acquisition functions
│   ├── preprocessing.py            # Data preprocessing utilities
│   ├── statistical_tests.py        # Stationarity tests
│   ├── models.py                   # Model architectures
│   └── evaluation.py               # Metrics calculation
│
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```
# 🚀 Usage
Quick Start
python# 1. Clone the repository
git clone https://github.com/yourusername/aal-stock-forecasting.git
cd aal-stock-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Jupyter notebook
jupyter notebook notebooks/AAL_Stock_Forecasting.ipynb
Running Individual Models
python# Load and preprocess data
from src.data_loader import load_aal_data
from src.preprocessing import preprocess_data

df = load_aal_data(start='2010-01-01')
train, test = preprocess_data(df, split=0.75)

# Train GRU model (best performer)
from src.models import build_gru_model

model = build_gru_model(input_shape=(60, 1))
model.fit(x_train, y_train, validation_split=0.2, epochs=100)

# Make predictions
predictions = model.predict(x_test)
Forecasting Future Values
python# Generate 90-day forecast
from src.models import forecast_future

forecast = forecast_future(model, last_window=test_data[-60:], steps=90)
🔑 Key Findings

GRU is the optimal choice for AAL stock forecasting

4.38% MAPE indicates high accuracy
Computationally efficient with fewer parameters
Robust to overfitting


Deep learning outperforms traditional methods

Better captures non-linear patterns
More effective with large datasets
Handles complex temporal dependencies


Feature engineering opportunities

Current models use only opening price
Adding technical indicators could improve performance
Sentiment analysis integration potential



# 🔮 Future Improvements
```
 Incorporate additional features (volume, technical indicators)
 Implement attention mechanisms
 Add ensemble methods combining multiple models
 Include sentiment analysis from news/social media
 Real-time prediction pipeline
 Hyperparameter optimization using Optuna/Ray Tune
 Multi-step ahead forecasting
 Confidence intervals for predictions
 Web dashboard for interactive visualization
```
# 🤝 Contributing
```Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
```

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
👥 Authors


⭐ If you find this project useful, please consider giving it a star! ⭐
