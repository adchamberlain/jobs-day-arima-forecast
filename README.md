# jobs-day-arima-forecast

<b>SIMPLE ARIMA FORECAST OF BLS NONFARM PAYROLLS.</b>

This script (and accompanying data file) produces auto-ARIMA forecasts of BLS NSA payrolls, as well as implied BLS seasonal adjustment factors. Combines into a 12-month-ahead forecast of monthly SA nonfarm payrolls.

## Running Instructions

1. **Set up BLS API Key**: Ensure your BLS API key is stored as an environment variable `BLS_API_KEY` in your `.zshrc` file:
   ```bash
   export BLS_API_KEY='your_api_key_here'
   ```
   Then reload: `source ~/.zshrc`

2. Update the monthly start/end date in the script. `start_f` should be the start of the forecast period (generally the current month). `end_f` must be 11 months after `start_f`.

3. Make sure the working directory is pointing at the right folder.

4. Run from command line: `python arima_jobs_day.py`

**Note**: The script now automatically fetches the latest data from the BLS API, so manual CSV updates are no longer required. The data is fetched from:
- Source (NSA Payrolls): https://data.bls.gov/timeseries/CEU0000000001
- Source (SA Payrolls): https://data.bls.gov/timeseries/CES0000000001

## Installation Instructions

<b>Note:</b> First time use may require installation of pyramid.arima, matplotlib, statsmodels, and requests Python packages. Requires Python 3.5. Mac Terminal commands:

<code>pip install pyramid-arima</code>

<code>pip install -U statsmodels</code>

<code>pip install matplotlib</code>

<code>pip install requests</code>

See `requirements.txt` for version requirements. Newer versions of statsmodels appear to be incompatible with `pyramid.arima`. Installing the exact versions required can be accomplished with the following code:

```pip install -r requirements.txt```


Author: Andrew Chamberlain, Ph.D.
andrewchamberlain.com