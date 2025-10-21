# Time Series Forecasting - Pure Python + Streamlit

This is a pure Python conversion of the Jupyter notebook-based time series forecasting project, featuring an interactive Streamlit web application.

## Features

- **Data Processing**: Load and preprocess various time series datasets
- **Traditional Methods**: ARIMA, Holt-Winters, seasonal decomposition
- **Deep Learning**: LSTM and CNN-based forecasting
- **Prophet**: Facebook Prophet forecasting with cross-validation
- **Interactive UI**: Streamlit-based web interface
- **Visualization**: Plotly charts for better interactivity

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Direct Streamlit
```bash
streamlit run streamlit_app.py
```

### Option 2: Using the launcher
```bash
python run_app.py
```

## Available Datasets

- **Airline Passengers**: Monthly international airline passengers (1949-1960)
- **Alcohol Sales**: Monthly alcohol sales data
- **Female Births**: Daily female births in California (1959)
- **Restaurant Visitors**: Daily restaurant visitor counts
- **Superstore Sales**: Furniture sales data with time series analysis

## Forecasting Methods

### Traditional Methods
- **Holt-Winters**: Triple exponential smoothing
- **ARIMA**: Autoregressive Integrated Moving Average
- **Seasonal Decomposition**: Trend, seasonal, and residual analysis

### Deep Learning
- **LSTM**: Long Short-Term Memory networks (PyTorch)
- **CNN**: Convolutional Neural Networks (TensorFlow)

### Advanced Methods
- **Prophet**: Facebook's forecasting tool with holiday effects

## Project Structure

```
├── data_processor.py          # Data loading and preprocessing
├── traditional_forecasting.py # Classical forecasting methods
├── deep_learning_forecasting.py # LSTM and CNN implementations
├── prophet_forecasting.py     # Prophet forecasting
├── streamlit_app.py          # Main Streamlit application
├── run_app.py                # Launcher script
├── requirements.txt           # Python dependencies
└── data/                     # Dataset files
```

## Key Benefits of Python Conversion

1. **Modularity**: Clean separation of concerns
2. **Reusability**: Functions can be imported and used elsewhere
3. **Testing**: Easier to write unit tests
4. **Deployment**: Can be deployed as a web service
5. **Integration**: Easier to integrate with other Python applications

## Customization

- Add new datasets by updating `data_processor.py`
- Implement new forecasting methods in separate modules
- Customize the Streamlit interface in `streamlit_app.py`
- Modify model parameters and hyperparameters

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed
- **Memory Issues**: Reduce sequence length for deep learning models
- **Slow Training**: Reduce epochs or use smaller datasets for testing

## Contributing

Feel free to add new forecasting methods, improve the UI, or enhance the data processing capabilities.
