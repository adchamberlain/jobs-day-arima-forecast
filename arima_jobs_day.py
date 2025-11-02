# SIMPLE ARIMA FORECAST OF BLS NONFARM PAYROLLS.
# Note: Produces ARIMA forecasts of BLS NSA payrolls, as well as implied
# BLS seasonal adjustment factors. Combines into a 12-month-ahead forecast
# of monthly SA nonfarm payrolls.
#
# Andrew Chamberlain, Ph.D.
# andrewchamberlain.com 
# ORIGINAL: July 2018
# LAST UPDATE: November 2, 2025 (Modernized to use BLS API)

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import os
import requests
import json

# Set working directory to the script's location
# Handle both regular execution and iPython/interactive sessions
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined in iPython interactive sessions
    script_dir = os.getcwd()
os.chdir(script_dir)

##############################################
# CONFIGURATION: Update these dates monthly
##############################################
# Training period for ARIMA model
start = '2010-01-01'  # Start of training data
end = '2025-08-01'    # End of training data (last historical month)

# Automatically calculate 12-month forecast period
end_date = pd.to_datetime(end)
start_f = (end_date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
end_f = (end_date + pd.DateOffset(months=12)).strftime('%Y-%m-%d')


def get_bls_api_key():
    """
    Get BLS API key from environment variable or .zshrc file.
    
    Returns:
    - API key string
    """
    # First, try to get from environment variable
    api_key = os.getenv('BLS_API_KEY')
    
    if not api_key:
        # Try to read from .zshrc file
        zshrc_path = os.path.expanduser('~/.zshrc')
        if os.path.exists(zshrc_path):
            try:
                with open(zshrc_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Look for export BLS_API_KEY="..." or export BLS_API_KEY='...'
                        if line.startswith('export BLS_API_KEY'):
                            # Extract the key from the line
                            if '=' in line:
                                key_part = line.split('=', 1)[1].strip()
                                # Remove quotes
                                api_key = key_part.strip('"').strip("'")
                                print("API key loaded from ~/.zshrc")
                                break
            except Exception as e:
                print(f"Warning: Could not read .zshrc file: {e}")
    
    if not api_key:
        error_msg = (
            "BLS_API_KEY not found.\n"
            "Please add to ~/.zshrc:\n"
            "  export BLS_API_KEY='your_key_here'\n"
            "Or set it in your Python session:\n"
            "  import os\n"
            "  os.environ['BLS_API_KEY'] = 'your_key_here'"
        )
        raise ValueError(error_msg)
    
    return api_key


def fetch_bls_data(series_ids, start_year, end_year):
    """
    Fetches data from the BLS API for given series IDs and date range.
    
    Parameters:
    - series_ids: List of BLS series IDs (e.g., ['CEU0000000001', 'CES0000000001'])
    - start_year: Start year for data retrieval (int)
    - end_year: End year for data retrieval (int)
    
    Returns:
    - DataFrame with datetime index and columns for each series
    """
    # Get API key
    api_key = get_bls_api_key()
    
    # BLS API has a limit of 20 years per request with registered key
    # We'll need to batch requests if the date range is larger
    all_data = {}
    
    # Process in 20-year batches
    for batch_start in range(start_year, end_year + 1, 20):
        batch_end = min(batch_start + 19, end_year)
        
        headers = {'Content-type': 'application/json'}
        data = {
            "seriesid": series_ids,
            "startyear": str(batch_start),
            "endyear": str(batch_end),
            "registrationkey": api_key
        }
        
        print(f"Fetching BLS data for years {batch_start}-{batch_end}...")
        response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', 
                                json=data, headers=headers)
        response_data = response.json()
        
        if response_data['status'] != 'REQUEST_SUCCEEDED':
            raise Exception(f"BLS API request failed: {response_data.get('message', 'No error message provided')}")
        
        # Parse the response
        for series in response_data['Results']['series']:
            series_id = series['seriesID']
            if series_id not in all_data:
                all_data[series_id] = {}
            
            for item in series['data']:
                year = item['year']
                period = item['period']
                
                # Only process monthly data (M01-M12)
                if period.startswith('M') and len(period) == 3:
                    month = period[1:]
                    date_str = f"{year}-{month}-01"
                    value = item['value']
                    
                    # Remove commas and convert to float
                    all_data[series_id][date_str] = float(value.replace(',', ''))
    
    # Convert to DataFrame
    df_list = []
    for series_id, data_dict in all_data.items():
        series_df = pd.DataFrame(list(data_dict.items()), columns=['date', series_id])
        series_df['date'] = pd.to_datetime(series_df['date'])
        series_df = series_df.sort_values('date')
        df_list.append(series_df)
    
    # Merge all series into one DataFrame
    df = df_list[0]
    for i in range(1, len(df_list)):
        df = df.merge(df_list[i], on='date', how='outer')
    
    df = df.set_index('date').sort_index()
    
    return df


# Fetch BLS payrolls data (NSA and SA) using API
# NSA Payrolls: CEU0000000001 (these are levels in thousands)
# SA Payrolls: CES0000000001 (these are levels in thousands)
print("Fetching data from BLS API...")
series_ids = ['CEU0000000001', 'CES0000000001']

# Determine the year range needed (from 1990 to forecast end year)
start_year = 1990
end_year = int(end_f.split('-')[0])

df_levels = fetch_bls_data(series_ids, start_year, end_year)

# The CSV contains month-over-month CHANGES in payrolls (in thousands)
# BLS API returns LEVELS, so we need to compute the month-over-month changes
print("Computing month-over-month changes...")
df_changes = df_levels.diff()

# Calculate implied SA factors (NSA change - SA change)
df = pd.DataFrame({
    'payrolls_nsa': df_changes['CEU0000000001'],
    'payrolls_sa': df_changes['CES0000000001']
})
df['implied_sa_factors'] = df['payrolls_nsa'] - df['payrolls_sa']

# Drop the first row (which will be NaN due to diff())
df = df.dropna()

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# VALIDATION: Compare API data with CSV data (if CSV exists)
try:
    df_csv = pd.read_csv('data.csv', index_col=0)
    df_csv.index = pd.to_datetime(df_csv.index)
    df_csv.columns = ['payrolls_nsa','payrolls_sa','implied_sa_factors']
    
    # Find common date range
    common_dates = df.index.intersection(df_csv.index)
    
    if len(common_dates) > 0:
        # Compare the data
        api_subset = df.loc[common_dates]
        csv_subset = df_csv.loc[common_dates]
        
        # Calculate differences
        diff_nsa = (api_subset['payrolls_nsa'] - csv_subset['payrolls_nsa']).abs()
        diff_sa = (api_subset['payrolls_sa'] - csv_subset['payrolls_sa']).abs()
        
        max_diff_nsa = diff_nsa.max()
        max_diff_sa = diff_sa.max()
        
        print(f"\nValidation against CSV data:")
        print(f"  Common dates: {len(common_dates)}")
        print(f"  Max difference in NSA payrolls: {max_diff_nsa:.1f} thousand")
        print(f"  Max difference in SA payrolls: {max_diff_sa:.1f} thousand")
        
        # Flag if differences are significant (> 1 thousand jobs)
        if max_diff_nsa > 1 or max_diff_sa > 1:
            print("  WARNING: Significant differences detected between API and CSV data!")
            print("  This may be due to data revisions by BLS.")
        else:
            print("  Data validation PASSED - API data matches CSV data.")
except FileNotFoundError:
    print("\nNo CSV file found for validation. Proceeding with API data only.")
except Exception as e:
    print(f"\nWarning: Could not validate against CSV data: {e}")

print("\nProceeding with API data...\n")

# Perform seasonal and trend decomposition for NSA payrolls.
decomp = seasonal_decompose(df.loc[start:end,'payrolls_nsa'], model='additive', period=12, extrapolate_trend=12)
fig = decomp.plot()
fig.savefig('decomp.png')

# Examine components of the NSA payrolls time series decomposition.
trend = decomp.trend
seasonal = decomp.seasonal
resid = decomp.resid

##########################################################################
# Run auto ARIMA proceedure for NSA payrolls and implied BLS SA factors.
##########################################################################

# Auto ARIMA for NSA payrolls.
model_nsa = auto_arima(df.loc[start:end,'payrolls_nsa'], exogenous=None, start_p=1, start_q=1,
                           max_p=6, max_q=6, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
model_nsa.summary()

# Auto ARIMA for implied BLS SA factors.
model_sa_factors = auto_arima(df.loc[start:end,'implied_sa_factors'], exogenous=None, start_p=1, start_q=1,
                           max_p=6, max_q=6, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
model_sa_factors.summary()


#################################################
# Create 12-month ahead forecast of NSA payrolls.
#################################################

# Create training data frame.
train = df.loc[start:end,'payrolls_nsa']

# Create forecast date range (12 months ahead)
forecast_dates = pd.date_range(start=start_f, end=end_f, freq='MS')

# Fit arima model on NSA payrolls time series.
model_nsa.fit(train)

# Forecast 12 months ahead.
forecast_nsa = model_nsa.predict(n_periods = 12)

# Create forecast dataframe.
forecast_nsa_df = pd.DataFrame(forecast_nsa,
                               index=forecast_dates,
                               columns=['prediction_nsa'])


###########################################################
# Create 12-month ahead forecast of BLS implied SA factors.
###########################################################

# Create training data frame.
train_sa_factors = df.loc[start:end,'implied_sa_factors']

# Fit arima model on BLS implied SA factors time series.
model_sa_factors.fit(train_sa_factors)

# Forecast 12 months ahead.
forecast_sa_factors = model_sa_factors.predict(n_periods = 12)

# Create forecast dataframe.
forecast_sa_factors_df = pd.DataFrame(forecast_sa_factors, index=forecast_dates, columns=['prediction_sa_factors'])


###########################################
# Merge full forecast data together.
###########################################

# Merge together forecasts of NSA payrolls and implied BLS SA factors.
df_forecast = pd.concat([df,forecast_nsa_df,forecast_sa_factors_df],axis=1)

# Generate new column for 12-month ahead forecast of SA payrolls (NSA forecast minus forecasted SA factors).
df_forecast['forecast_sa'] = df_forecast['prediction_nsa'] - df_forecast['prediction_sa_factors']

# Push out 12-month SA nonfarm payrolls forecast to CSV.
df_forecast.to_csv('forecast.csv')

###########################################
# Display SA forecast results in console
###########################################
print("\n" + "="*70)
print("12-MONTH SEASONALLY ADJUSTED (SA) NONFARM PAYROLLS FORECAST")
print("="*70)
print("\nForecast Period: {} to {}\n".format(start_f, end_f))

# Create a clean display table
forecast_display = df_forecast.loc[start_f:end_f, ['forecast_sa']].copy()
forecast_display['forecast_sa'] = forecast_display['forecast_sa'].round(0)
forecast_display.columns = ['SA Payrolls Change (000s)']

print(forecast_display.to_string())

print("\n" + "="*70)
print("Summary Statistics:")
print("  Mean monthly change: {:,.0f} thousand jobs".format(forecast_display['SA Payrolls Change (000s)'].mean()))
print("  Median monthly change: {:,.0f} thousand jobs".format(forecast_display['SA Payrolls Change (000s)'].median()))
print("  Min monthly change: {:,.0f} thousand jobs".format(forecast_display['SA Payrolls Change (000s)'].min()))
print("  Max monthly change: {:,.0f} thousand jobs".format(forecast_display['SA Payrolls Change (000s)'].max()))
print("  Total 12-month change: {:,.0f} thousand jobs".format(forecast_display['SA Payrolls Change (000s)'].sum()))
print("="*70)
print("\nForecast saved to: forecast.csv")
print("Decomposition plot saved to: decomp.png\n")

### end ###
