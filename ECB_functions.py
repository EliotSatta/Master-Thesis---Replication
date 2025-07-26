import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from ecbdata import ecbdata
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

######### Section 1. #########
def quarter_to_month(time_period: str) -> str:
    """
    Convert a quarterly label to its corresponding month‐end string.

    This function maps strings of the form 'YYYY-Qx' to the month ending
    that quarter:
      - 'YYYY-Q1' → 'YYYY-03'
      - 'YYYY-Q2' → 'YYYY-06'
      - 'YYYY-Q3' → 'YYYY-09'
      - 'YYYY-Q4' → 'YYYY-12'
    If the input is not in the expected quarterly format, it is returned unchanged.

    Parameters
    ----------
    time_period : str
        A date label, expected as 'YYYY-Q1', 'YYYY-Q2', etc., or any other
        string for which no conversion is required.

    Returns
    -------
    str
        The converted 'YYYY-MM' string corresponding to quarter‐end, or the
        original input if it does not match 'YYYY-Qx'.
    """
    # Only attempt conversion if the pattern '-Q' exists in the string
    if '-Q' in time_period:
        # Split the input into year and quarter number
        year, quarter = time_period.split('-Q')  # e.g. '2023-Q1' → ['2023', '1']

        # Define mapping from quarter number to quarter‐end month
        quarter_end_months = {
            '1': '03',  # Q1 → March
            '2': '06',  # Q2 → June
            '3': '09',  # Q3 → September
            '4': '12',  # Q4 → December
        }

        # If the quarter is recognized, format the output string
        if quarter in quarter_end_months:
            return f"{year}-{quarter_end_months[quarter]}"

    # For any non‐quarterly label, return the input unchanged
    return time_period

def get_ecb_series(series_code: str, col_name: str, frequency: str) -> pd.DataFrame:
    """
    Retrieve and process an ECB time series, returning quarterly values.

    Parameters
    ----------
    series_code : str
        The ECB’s unique code for the desired time series.
    col_name : str
        The name to assign to the series values in the output.
    frequency : {'Q', 'A', 'M'}
        The original frequency of the data:
        - 'Q': already quarterly (no aggregation required).
        - 'A': annual (mapped to March of each year).
        - 'M': monthly (aggregated to quarterly by mean).

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - TIME_PERIOD : str
            Quarter‐end label in 'YYYY-MM' format.
        - {col_name} : float
            The series values, aggregated to quarters if needed.
    """
    # 1. Fetch raw data (TIME_PERIOD, OBS_VALUE) from the ECB API
    df = ecbdata.get_series(series_code, detail='dataonly')[['TIME_PERIOD', 'OBS_VALUE']]

    # 2. Handle quarterly data: convert 'YYYY-Qx' → 'YYYY-MM'
    if frequency.upper() == 'Q':
        df['TIME_PERIOD'] = df['TIME_PERIOD'].apply(quarter_to_month)

    # 3. Handle annual data: map each year to March ('YYYY-03')
    elif frequency.upper() == 'A':
        df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(str) + '-03'

    # 4. Handle monthly data: compute quarterly means
    elif frequency.upper() == 'M':
        # 4.1 Parse strings to datetime at month precision
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'],
                                           format='%Y-%m',
                                           errors='coerce')

        # 4.2 Resample index from monthly to quarterly, taking the mean
        df.set_index('TIME_PERIOD', inplace=True)
        df = df.resample('Q').mean()

        # 4.3 Restore TIME_PERIOD as 'YYYY-MM' strings at quarter‐end
        df.reset_index(inplace=True)
        df['TIME_PERIOD'] = df['TIME_PERIOD'].dt.strftime('%Y-%m')

    # 5. Rename the value column to the user‐provided name
    df.rename(columns={'OBS_VALUE': col_name}, inplace=True)

    return df

def merge_ecb_series(series_list: list[tuple[str, str, str]]) -> pd.DataFrame:
    """
    Retrieve and combine multiple ECB time series into a single DataFrame.

    Each series is fetched and, if necessary, aggregated to quarterly frequency.
    All series are then merged on the TIME_PERIOD column using an outer join,
    preserving any periods present in any series.

    Parameters
    ----------
    series_list : list of (series_code, col_name, frequency) tuples
        A list where each element specifies:
        - series_code : str
            The ECB’s code for the time series.
        - col_name : str
            The desired column name for the series values.
        - frequency : {'Q', 'A', 'M'}
            The original frequency of the data:
            * 'Q' for quarterly (no aggregation).
            * 'A' for annual (mapped to March of each year).
            * 'M' for monthly (aggregated to quarterly means).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by an integer index with columns:
        - TIME_PERIOD : str
            Quarter‐end labels in 'YYYY-MM' format.
        - One column per series, named according to each tuple’s `col_name`.
    """
    # 1. Initialize by fetching the first series
    base_code, base_col_name, base_freq = series_list[0]
    merged_df = get_ecb_series(base_code, base_col_name, base_freq)

    # 2. Iteratively fetch and merge each remaining series
    for code, col_name, freq in series_list[1:]:
        # 2.1 Fetch and process the next series
        df = get_ecb_series(code, col_name, freq)
        # 2.2 Merge with the accumulated DataFrame on TIME_PERIOD (outer join)
        merged_df = pd.merge(merged_df, df, on='TIME_PERIOD', how='outer')

    # 3. Order the result by TIME_PERIOD for readability
    merged_df.sort_values(by='TIME_PERIOD', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

def build_ecb_dataset(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch and merge a fixed list of 11 ECB series into a quarterly DataFrame.

    Parameters
    ----------
    start_date : str, optional
        Earliest TIME_PERIOD to keep (ISO 'YYYY-MM' or 'YYYY-MM-DD').
    end_date : str, optional
        Latest TIME_PERIOD to keep (ISO 'YYYY-MM' or 'YYYY-MM-DD').

    Returns
    -------
    pd.DataFrame
        A DataFrame with:
          - TIME_PERIOD as datetime (quarter-end),
          - the 11 pre-specified series as columns,
          - filtered to [start_date, end_date] if provided.
    """

    # ─────────────────────────────────────────────────────────────────────
    # 1) Hard-coded series list: (ECB_code, column_name, frequency)
    series_list = [
        ('MNA.Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N', 'Nominal GDP (millions of €)', 'Q'),
        ('MNA.Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.IX.D.N', 'GDP Deflator Index', 'Q'),
        ('ENA.A.N.I9.W0.S1.S1._Z.POP._Z._Z._Z.PS._Z.N', 'Total Population (Thousands)', 'A'),
        ('MNA.Q.Y.I9.W0.S1M.S1.D.P31._Z._Z._T.EUR.V.N', 'Private Final Consumption (millions of €)', 'Q'),
        ('MNA.Q.Y.I9.W0.S1.S1.D.P51G.N11G._T._Z.EUR.V.N', 'Total Investment (millions of €)', 'Q'),
        ('QSA.Q.Y.I9.W0.S1M.S1.N.D.P51G._Z._Z._Z.XDC._T.S.V.N._T', 'Households & NPISH Investment (millions of €)', 'Q'),
        ('QSA.Q.Y.I9.W0.S11.S1.N.D.P51G._Z._Z._Z.XDC._T.S.V.N._T', 'Non-Financial Corporations Investment (millions of €)', 'Q'),
        ('QSA.Q.Y.I9.W0.S12.S1.N.D.P51G._Z._Z._Z.XDC._T.S.V.N._T', 'Financial Corporations Investment (millions of €)', 'Q'),
        ('LFSI.M.I9.S.UNEHRT.TOTAL0.15_74.T', 'Unemployment rate (%)', 'M'),
        ('MNA.Q.Y.I9.W2.S1.S1._Z.COM_PS._Z._T._Z.IX.V.N', 'Nominal Compensation per Employee Index', 'Q'),
        ('GFS.Q.N.I9.W0.S13.S1.C.L.LE.GD.T._Z.XDC_R_B1GQ_CY._T.F.V.N._T', 'Government Debt (as % of GDP)', 'Q'),
        ('MNA.Q.Y.I9.W0.S13.S1.D.P3._Z._Z._T.EUR.V.N', 'Government Final Consumption (millions of €)', 'Q'),
        ('QSA.Q.Y.I9.W0.S1M.S1.N.B.D62._Z._Z._Z.XDC._T.S.V.N._T', 'Social Benefits (millions of €)', 'Q'),
        ('QSA.Q.Y.I9.W0.S1M.S1.N.B.D63._Z._Z._Z.XDC._T.S.V.N._T', 'Social Transfers in Kind (millions of €)', 'Q'),
        ('FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA', 'Euribor 3-months - average of observations (%)', 'M')
          ]
    # ─────────────────────────────────────────────────────────────────────

    # 2) Fetch & merge
    df = merge_ecb_series(series_list)

    # 3) Convert TIME_PERIOD → datetime (quarter-end):
    #    append '-01' to get YYYY-MM-DD, then parse
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'] + '-01',
                                       format='%Y-%m-%d',
                                       errors='raise')

    # 4) Apply optional date filtering
    if start_date:
        start = pd.to_datetime(start_date, errors='raise')
        df = df[df['TIME_PERIOD'] >= start]
    if end_date:
        end = pd.to_datetime(end_date, errors='raise')
        df = df[df['TIME_PERIOD'] <= end]

    # 5) Reset index for cleanliness
    return df.reset_index(drop=True)

######### Section 2. #########
def per_capita(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per capita versions of selected macroeconomic variables,
    applying appropriate scaling factors for both the variable and the population.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with macroeconomic variables and population.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, updated with new per capita columns.
    """
    # Hardcoded list of variable definitions
    per_capita_variables = [
        ('Nominal GDP (millions of €)', 'Total Population (Thousands)', 'Nominal GDP per capita', 1_000_000, 1_000),
        ('Private Final Consumption (millions of €)', 'Total Population (Thousands)', 'Private Final Consumption per capita', 1_000_000, 1_000),
        ('Total Investment (millions of €)', 'Total Population (Thousands)', 'Total Investment per capita', 1_000_000, 1_000),
        ('Government Final Consumption (millions of €)', 'Total Population (Thousands)', 'Government Final Consumption per capita', 1_000_000, 1_000),
        ('Government Transfers (millions of €)', 'Total Population (Thousands)', 'Government Transfers per capita', 1_000_000, 1_000)
    ]

    # Compute per capita for each variable
    for var_col, pop_col, new_col_name, var_factor, pop_factor in per_capita_variables:
        numerator = df[var_col] * var_factor
        denominator = df[pop_col] * pop_factor
        df[new_col_name] = numerator / denominator

    return df

def rebase_index(df: pd.DataFrame, date_col: str, new_base_date) -> pd.DataFrame:
    """
    Rebase selected index columns so that their value at `new_base_date` equals 100.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the indices and time column.
    date_col : str
        The name of the date column (can also be the index).
    new_base_date : str or datetime-like
        The date at which all indices will be set to 100.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with selected indices rebased.
    """
    # Hardcoded list of index columns to rebase
    index_columns = [
        "GDP Deflator Index",
        "Nominal Compensation per Employee Index"
    ]

    df_out = df.copy()

    # Handle date column if it's in the index
    if date_col == df_out.index.name:
        df_out = df_out.reset_index()
    elif date_col not in df_out.columns:
        raise KeyError(f"Date column '{date_col}' not found.")

    # Convert date column and base date
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    base_date = pd.to_datetime(new_base_date)

    # Check base date is present
    mask = df_out[date_col] == base_date
    if not mask.any():
        raise ValueError(f"Base date {new_base_date} not found in column '{date_col}'.")

    # Rebase each index column
    for index_col in index_columns:
        base_value = df_out.loc[mask, index_col].iloc[0]
        if not pd.api.types.is_numeric_dtype(type(base_value)) or base_value == 0:
            raise ValueError(f"Invalid base value {base_value} at {new_base_date} in '{index_col}'.")
        df_out[index_col] = df_out[index_col] * (100.0 / base_value)

    return df_out

def interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the 'Total Population (Thousands)' column using linear interpolation,
    filling only the gaps inside the data (no extrapolation).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the population column.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the interpolated population column.
    """
    df_out = df.copy()
    df_out["Total Population (Thousands)"] = (
        df_out["Total Population (Thousands)"]
        .interpolate(method="linear", limit_area="inside")
    )
    return df_out

def real_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes log-difference growth rates (real or nominal) for a hardcoded list of macroeconomic variables,
    optionally deflating by an index before computing the growth.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with relevant variables.

    Returns
    -------
    pd.DataFrame
        The same DataFrame updated with the new growth rate columns.
    """
    # Hardcoded list of (input_column, output_column, scale_boolean, deflator_column)
    growth_specs = [
        ('Nominal GDP per capita', 'Real per capita GDP Growth (%)', 'GDP Deflator Index'),
        ('GDP Deflator Index', 'Inflation (%)', None),
        ('Private Final Consumption per capita', 'Real per capita Consumption Growth (%)', 'GDP Deflator Index'),
        ('Total Investment per capita', 'Real per capita Investment Growth (%)', 'GDP Deflator Index'),
        ('Government Final Consumption per capita', 'Real per capita Governemnt Consumption Growth (%)', 'GDP Deflator Index'),
        ('Government Transfers per capita', 'Real per capita Government Transfers Growth (%)', 'GDP Deflator Index'),
        ('Nominal Compensation per Employee Index', 'Real Wage Growth (%)', 'GDP Deflator Index')
    ]

    for col_name, new_col_name, deflator_col in growth_specs:
        # Select series (real or nominal)
        if deflator_col:
            if (df[deflator_col] <= 0).any():
                raise ValueError(f"'{deflator_col}' contains non-positive values.")
            series = df[col_name] / df[deflator_col]
        else:
            series = df[col_name]

        if (series <= 0).any():
            raise ValueError(f"'{col_name}' (or its real version) contains non-positive values.")

        log_diff = np.log(series).diff()
        df[new_col_name] = log_diff

    return df

######### Section 3. #########
def build_estimation_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the final dataset for estimation by selecting, renaming,
    and cleaning the necessary variables from ECB_data, and scaling
    interest-rate, unemployment and debt series to model‐friendly units.

    Parameters
    ----------
    df : pd.DataFrame
        Original ECB_data DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned, renamed, and rescaled DataFrame ready for estimation.
    """
    # 1) Select only the raw columns we care about
    selected = df.loc[:, [
        'TIME_PERIOD',
        'Real per capita GDP Growth (%)',
        'Real per capita Consumption Growth (%)',
        'Real per capita Investment Growth (%)',
        'Real per capita Governemnt Consumption Growth (%)',
        'Real per capita Government Transfers Growth (%)',
        'Real Wage Growth (%)',
        'Inflation (%)',
        'Unemployment rate (%)',
        'Government Debt (as % of GDP)',
        'Euribor 3-months - average of observations (%)'
    ]]

    # 2) Rename to DSGE-friendly observation names
    renamed = selected.rename(columns={
        'TIME_PERIOD': 't',
        'Real per capita GDP Growth (%)': 'dy_obs',
        'Real per capita Consumption Growth (%)': 'dc_obs',
        'Real per capita Investment Growth (%)': 'dinvest_obs',
        'Real per capita Governemnt Consumption Growth (%)': 'dg_obs',
        'Real per capita Government Transfers Growth (%)': 'dtau_obs',
        'Real Wage Growth (%)': 'pi_w_obs',
        'Inflation (%)': 'pi_p_obs',
        'Unemployment rate (%)': 'u_obs',
        'Government Debt (as % of GDP)': 'sB_obs',
        'Euribor 3-months - average of observations (%)': 'rS_obs'
    })

    # 3) Rescale selected columns directly
    renamed['rS_obs'] = renamed['rS_obs'] / 400.0  # Annualized % → quarterly decimal
    renamed['u_obs']  = renamed['u_obs']  / 100.0  # % → fraction
    renamed['sB_obs'] = renamed['sB_obs'] / 100.0  # % → fraction

    return renamed.dropna()

def seasonally_adjustment(df, series_name, period=4):
    """
    Seasonally adjusts a given series in a DataFrame, plots original and adjusted series,
    and returns the modified DataFrame with the adjusted series replacing the original.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - series_name: str, name of the column to seasonally adjust
    - period: int, seasonal period (default=4 for quarterly data)
    
    Returns:
    - Modified DataFrame with the adjusted series
    """
    # Perform seasonal decomposition
    result = seasonal_decompose(df[series_name], model='additive', period=period)

    # Compute seasonally adjusted series
    adjusted_series = df[series_name] - result.seasonal

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df[series_name], label='Original Series', color='blue', linestyle='dashed', linewidth=0.75)
    plt.plot(adjusted_series, label='Seasonally Adjusted Series', color='red', linewidth=0.75)
    plt.title(f'Original and Seasonally Adjusted {series_name}')
    plt.legend()
    plt.grid()
    plt.show()

    # Replace the original series in the DataFrame
    df[series_name] = adjusted_series

    return df

def plot(df, time_col='t'):
    """
    Plot each column of df (except time_col) against time_col,
    adding a y=0 line for selected variables.
    """
    center_zero_vars = {'dy_obs','dc_obs','dinvest_obs','dg_obs','dtau_obs','pi_w_obs','pi_p_obs'}

    for col in df.columns:
        if col == time_col:
            continue

        plt.figure(figsize=(8,4))
        plt.plot(df[time_col], df[col], label=col, color='blue' ,linewidth=1)

        if col in center_zero_vars:
            plt.axhline(0, color='black', linestyle='--', linewidth=0.75)

        plt.title(col)
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        import os

def save_df(
    df: pd.DataFrame,
    directory: str,
    filename: str
) -> str:
    """
    Save a DataFrame to CSV, creating the target directory if it doesn't exist.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    directory : str
        Path to the directory where the CSV will be saved.
    filename : str
        Name of the CSV file (should include '.csv' extension).

    Returns
    -------
    str
        Full file path of the saved CSV.

    Example
    -------
    path = save_dataframe_csv(
        df=estimation_df,
        directory='/Users/eliotsatta/Documents/Master Thesis',
        filename='estimation_df.csv'
    )
    # Prints: "Saved CSV to: /Users/.../Master Thesis/estimation_df.csv"
    """
    # 1) Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    # 2) Build full file path
    file_path = os.path.join(directory, filename)

    # 3) Save DataFrame to CSV without the index
    df.to_csv(file_path, index=False)

    # 4) Notify the user
    print(f"Saved CSV to: {file_path}")

    return file_path
