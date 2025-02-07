import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import logging
from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Logger & Paths Setup
# ------------------------------------------------------------------------------
# Initialize the logger for the preprocessing module.
logging = configure_logger("preprocessing", "preprocessing.log")

# Define directories for raw (scraped) and cleaned data.
RAW_DATA_FOLDER: str = './data/scraped'
CLEANED_DATA_FOLDER: str = './data/cleaned'
os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# Expected Columns (Normalized)
# ------------------------------------------------------------------------------
# List of columns expected after normalization; they exactly match the columns available.
EXPECTED_COLUMNS = [
    'rank', 'player', 'country_code', 'country', 'position', 'squad',
    'age', 'born', 'matches_played', 'starts', 'minutes_played', '90s_played',
    'goals', 'assists', 'goals_+_assists', 'goals_-_penalties', 'penalty_kicks_made',
    'penalty_kicks_attempted', 'yellow_cards', 'red_cards', 'expected_goals',
    'non-penalty_xg', 'expected_assists', 'non-penalty_xg_+_xag', 'progressive_carries',
    'progressive_passes', 'progressive_receives', 'goals_per_90', 'assists_per_90',
    'goals_+_assists_per_90', 'goals_-_penalties_per_90', 'goals_+_assists_-_penalties_per_90',
    'xg_per_90', 'xag_per_90', 'xg_+_xag_per_90', 'non-penalty_xg_per_90',
    'non-penalty_xg_+_xag_per_90'
]
# Normalize expected column names for validation.
NORMALIZED_EXPECTED_COLUMNS = set(EXPECTED_COLUMNS)

# ------------------------------------------------------------------------------
# Country Code Mapping
# ------------------------------------------------------------------------------
# Mapping from country code abbreviations to full country names.
COUNTRY_CODE_MAPPING = {
    'NGA': 'Nigeria', 'BRA': 'Brazil', 'ENG': 'England', 'FRA': 'France',
    'GER': 'Germany', 'ESP': 'Spain', 'ITA': 'Italy', 'ARG': 'Argentina',
    'USA': 'United States', 'MEX': 'Mexico', 'POR': 'Portugal', 'NED': 'Netherlands',
    'BEL': 'Belgium', 'SCO': 'Scotland', 'WAL': 'Wales', 'DEN': 'Denmark',
    'SWE': 'Sweden', 'NOR': 'Norway', 'FIN': 'Finland', 'ISL': 'Iceland',
    'CRO': 'Croatia', 'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia',
    'CZE': 'Czech Republic', 'POL': 'Poland', 'HUN': 'Hungary', 'AUT': 'Austria',
    'SWI': 'Switzerland', 'GRE': 'Greece', 'TUR': 'Turkey', 'ROU': 'Romania',
    'RUS': 'Russia', 'UKR': 'Ukraine', 'BLR': 'Belarus', 'KAZ': 'Kazakhstan',
    'CHN': 'China', 'JPN': 'Japan', 'KOR': 'South Korea', 'AUS': 'Australia',
    'NZL': 'New Zealand', 'IRN': 'Iran', 'IRQ': 'Iraq', 'SAU': 'Saudi Arabia',
    'EGY': 'Egypt', 'ALG': 'Algeria', 'TUN': 'Tunisia', 'MAR': 'Morocco',
    'ZAF': 'South Africa', 'GHA': 'Ghana', 'CIV': 'Ivory Coast', 'SEN': 'Senegal',
    'CMR': 'Cameroon', 'KEN': 'Kenya', 'UGA': 'Uganda', 'TAN': 'Tanzania',
    'COL': 'Colombia', 'CHI': 'Chile', 'URU': 'Uruguay', 'PAR': 'Paraguay',
    'PER': 'Peru', 'VEN': 'Venezuela', 'ECU': 'Ecuador', 'BOL': 'Bolivia',
    'CRC': 'Costa Rica', 'PAN': 'Panama', 'SLV': 'El Salvador', 'HON': 'Honduras',
    'CAN': 'Canada', 'CUB': 'Cuba', 'TRI': 'Trinidad and Tobago', 'JAM': 'Jamaica',
    'HTI': 'Haiti', 'DOM': 'Dominican Republic', 'ALB': 'Albania', 'ARM': 'Armenia',
    'SUI': 'Switzerland', 'LUX': 'Luxembourg', 'TOG': 'Togo', 'SUR': 'Suriname',
    'KVX': 'Kosovo', 'BIH': 'Bosnia and Herzegovina', 'MKD': 'North Macedonia',
    'ISR': 'Israel', 'PHI': 'Philippines', 'CUW': 'Curaçao', 'EQG': 'Equatorial Guinea',
    'MLI': 'Mali', 'BFA': 'Burkina Faso', 'COD': 'Democratic Republic of Congo',
    'GUI': 'Guinea', 'BUL': 'Bulgaria', 'ANG': 'Angola', 'FRO': 'Faroe Islands',
    'BEN': 'Benin', 'CGO': 'Congo', 'MNE': 'Montenegro', 'COM': 'Comoros',
    'IRL': 'Ireland', 'GAM': 'Gambia', 'CTA': 'Central African Republic',
    'MTN': 'Mauritania', 'MTQ': 'Martinique', 'GEO': 'Georgia', 'GAB': 'Gabon',
    'CPV': 'Cape Verde', 'PLE': 'Palestine', 'MOZ': 'Mozambique', 'ZIM': 'Zimbabwe',
    'GNB': 'Guinea-Bissau', 'ZAM': 'Zambia', 'SYR': 'Syria', 'GLP': 'Guadeloupe',
    'GUF': 'French Guiana', 'RSA': 'South Africa', 'HAI': 'Haiti', 'MAD': 'Madagascar',
    'NCL': 'New Caledonia', 'CHA': 'Chad', 'BDI': 'Burundi', 'UZB': 'Uzbekistan',
    'JOR': 'Jordan', 'MLT': 'Malta', 'NIR': 'Northern Ireland', 'SKN': 'Saint Kitts and Nevis',
    'GRN': 'Grenada', 'LTU': 'Lithuania', 'MDA': 'Moldova', 'EST': 'Estonia',
    'LBY': 'Libya', 'SLE': 'Sierra Leone', 'CYP': 'Cyprus',
    'N/A': 'Not Available',
}


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten multi-level headers if present by joining the levels with an underscore.

    Args:
        df (pd.DataFrame): DataFrame that may have multi-level column headers.

    Returns:
        pd.DataFrame: DataFrame with flattened column headers.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join([str(item).strip() for item in col if item and str(item) != 'nan'])
            for col in df.columns.values
        ]
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by converting to lower-case and replacing spaces with underscores.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=True)
    return df


def standardize_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize player names by removing unwanted characters and converting them to lower-case.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardized player names.
    """
    if 'player' in df.columns:
        df['player'] = df['player'].apply(
            lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x).lower() if isinstance(x, str) else x
        )
    return df


def ensure_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns with few unique values to categorical data type and
    ensure that numeric columns are properly converted.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with corrected data types.
    """
    # Convert suitable object columns to categorical type.
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:
            df[col] = df[col].astype('category')
    # Convert numeric columns ensuring coercion of errors.
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with excessive missing data and impute remaining missing values.

    - Drops columns that do not meet the threshold of non-missing values.
    - For numeric columns, missing values are replaced with the median.
    - For object or categorical columns, missing values are replaced with the mode.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    threshold = 0.3 * len(df)
    df.dropna(thresh=threshold, axis=1, inplace=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if not df[col].isnull().all():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic feature engineering using available columns.

    Example:
      - Compute xAG overperformance as the difference between assists and expected assists.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    if {'assists', 'expected_assists'}.issubset(df.columns):
        df['xag_overperformance'] = df['assists'] - df['expected_assists']
    return df


def additional_enhancements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply additional cleaning steps and simple enhancements.

    Enhancements include:
      - Removing rows with unrealistic ages, negative minutes played, or negative goals.
      - Computing a performance consistency score (standard deviation of goals per 90) grouped by player.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame after applying additional enhancements.
    """
    # Remove rows with unrealistic values.
    if 'age' in df.columns:
        df = df[(df['age'] >= 16) & (df['age'] <= 45)]
    if 'minutes_played' in df.columns:
        df = df[df['minutes_played'] >= 0]
    if 'goals' in df.columns:
        df = df[df['goals'] >= 0]

    # Compute performance consistency as the standard deviation of goals_per_90 for each player.
    if 'goals_per_90' in df.columns and 'player' in df.columns:
        df['performance_consistency'] = df.groupby('player')['goals_per_90'].transform(np.std)

    return df


def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an in-depth set of features derived from available data.

    The function adds multiple new metrics:
      - Playing Time Metrics (e.g., minutes per match, start rate)
      - Offensive Contribution Metrics (e.g., assist-to-goal ratio, penalty conversion rate)
      - Efficiency and Overperformance Metrics (comparing actual vs. expected values)
      - Per 90 Comparison Metrics
      - Non-Penalty and Progressive Metrics
      - Age and Birth Year related metrics

    Returns:
        pd.DataFrame: DataFrame enriched with advanced features.
    """
    # ---- Playing Time Metrics ----
    df['minutes_per_match'] = df['minutes_played'] / df['matches_played'].replace(0, np.nan)
    df['start_rate'] = df['starts'] / df['matches_played'].replace(0, np.nan)
    df['minutes_played_ratio'] = df['minutes_played'] / (df['matches_played'] * 90).replace(0, np.nan)
    df['nineties_per_match'] = df['90s_played'] / df['matches_played'].replace(0, np.nan)

    # ---- Offensive Contribution Metrics ----
    df['assist_to_goal_ratio'] = df['assists'] / df['goals'].replace(0, np.nan)
    df['penalty_conversion_rate'] = df['penalty_kicks_made'] / df['penalty_kicks_attempted'].replace(0, np.nan)
    df['discipline_score'] = df['yellow_cards'] + 3 * df['red_cards']
    df['finishing_efficiency'] = df['goals'] / df['expected_goals'].replace(0, np.nan)
    df['assisting_efficiency'] = df['assists'] / df['expected_assists'].replace(0, np.nan)
    df['overall_efficiency'] = (df['goals'] + df['assists']) / (df['expected_goals'] + df['expected_assists']).replace(
        0, np.nan)

    df['goal_overperformance'] = df['goals'] - df['expected_goals']
    df['assist_overperformance'] = df['assists'] - df['expected_assists']
    df['overall_overperformance'] = (df['goals'] + df['assists']) - (df['expected_goals'] + df['expected_assists'])
    df['goal_ratio'] = df['goals'] / (df['goals'] + df['assists']).replace(0, np.nan)

    # ---- Per 90 Comparison Metrics ----
    df['per90_goal_diff'] = df['goals_per_90'] - df['xg_per_90']
    df['per90_assist_diff'] = df['assists_per_90'] - df['xag_per_90']
    df['per90_overall_diff'] = df['goals_+_assists_per_90'] - df['xg_+_xag_per_90']
    df['conversion_rate_per90'] = df['goals_per_90'] / df['xg_per_90'].replace(0, np.nan)
    df['assist_conversion_per90'] = df['assists_per_90'] / df['xag_per_90'].replace(0, np.nan)

    # ---- Non-Penalty Metrics ----
    df['non_penalty_overperformance'] = df['goals_+_assists_-_penalties_per_90'] - df['non-penalty_xg_+_xag_per_90']
    df['penalty_impact'] = df['goals_+_assists_per_90'] - df['goals_+_assists_-_penalties_per_90']

    # ---- Progressive Metrics ----
    df['progressive_total'] = df['progressive_carries'] + df['progressive_passes'] + df['progressive_receives']
    df['progressive_actions_per_match'] = df['progressive_total'] / df['matches_played'].replace(0, np.nan)
    df['progressive_carries_ratio'] = df['progressive_carries'] / df['progressive_total'].replace(0, np.nan)
    df['progressive_passes_ratio'] = df['progressive_passes'] / df['progressive_total'].replace(0, np.nan)
    df['progressive_receives_ratio'] = df['progressive_receives'] / df['progressive_total'].replace(0, np.nan)

    # ---- Age & Born Metrics ----
    df['age_squared'] = df['age'] ** 2
    try:
        df['birth_year'] = pd.to_datetime(df['born'], errors='coerce').dt.year
    except Exception as e:
        logging.warning("Could not parse 'born' column to extract birth_year: " + str(e))

    return df


def data_integrity_checks(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Perform data integrity checks on the DataFrame.

    Checks include:
      - Logging any unmapped country codes.
      - Ensuring all critical columns (as defined in NORMALIZED_EXPECTED_COLUMNS) are present.
      - Warning about duplicate player records based on player, squad, and season.

    Args:
        df (pd.DataFrame): DataFrame to check.
        file_path (str): Path of the file being processed (used for logging).

    Returns:
        pd.DataFrame: The original DataFrame if integrity checks pass.

    Raises:
        ValueError: If any essential columns are missing.
    """
    # Check for unmapped country codes.
    if 'country_code' in df.columns:
        unmapped = df[~df['country_code'].isin(COUNTRY_CODE_MAPPING.keys())]
        if not unmapped.empty:
            logging.warning(
                f"Unmapped country codes in {file_path}: {unmapped['country_code'].unique()}"
            )

    # Validate column completeness.
    missing_cols = NORMALIZED_EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        logging.error(f"Missing critical columns in {file_path}: {missing_cols}")
        raise ValueError("Essential columns missing, stopping processing.")

    # Check for duplicate player records by player, squad, and season.
    if {'player', 'squad', 'season'}.issubset(df.columns):
        duplicates = df[df.duplicated(subset=['player', 'squad', 'season'], keep=False)]
        if not duplicates.empty:
            dup_list = duplicates[['player', 'squad', 'season']].drop_duplicates()
            logging.warning(f"Duplicate player records in {file_path}: {dup_list.to_dict(orient='records')}")

    return df


def preprocess_file(file_path: str, league: str, season: str) -> None:
    """
    Process and clean a single CSV file.

    Steps performed:
      1. Read the CSV file and flatten any multi-level headers.
      2. Normalize column names and standardize player names.
      3. Convert data types and handle missing values.
      4. Apply basic and advanced feature engineering.
      5. Append metadata (league and season).
      6. Run data integrity checks.
      7. Export the cleaned data to a gzip-compressed CSV and a Parquet file.

    Args:
        file_path (str): Path to the CSV file to process.
        league (str): League name extracted from the file name.
        season (str): Season extracted from the file name.
    """
    try:
        logging.info(f"Processing file: {file_path}")

        # Read the CSV file with a multi-level header and flatten it.
        df = pd.read_csv(file_path, header=[0, 1])
        df = flatten_columns(df)

        # Normalize column names and standardize player names.
        df = normalize_column_names(df)
        df = standardize_player_names(df)

        # Ensure that data types are correct.
        df = ensure_data_types(df)

        # Handle missing data.
        df = handle_missing_data(df)

        # Apply basic feature engineering.
        df = feature_engineering(df)

        # Apply additional enhancements.
        df = additional_enhancements(df)

        # Apply advanced feature engineering.
        df = advanced_feature_engineering(df)

        # If 'country' column is missing but 'country_code' exists, add the mapping.
        if 'country_code' in df.columns and 'country' not in df.columns:
            df['country'] = df['country_code'].map(COUNTRY_CODE_MAPPING).fillna('Not Available')

        # Append metadata about league and season.
        df['league'] = league
        df['season'] = season

        # Perform data integrity checks.
        df = data_integrity_checks(df, file_path)

        # ------------------------------------------------------------------------------
        # Export the cleaned DataFrame.
        # ------------------------------------------------------------------------------
        # Save as a gzip-compressed CSV.
        cleaned_csv_path = os.path.join(CLEANED_DATA_FOLDER, f"cleaned_{league}_{season}.csv.gz")
        df.to_csv(cleaned_csv_path, index=False, float_format="%.4f", compression='gzip')
        logging.info(f"Cleaned CSV saved to: {cleaned_csv_path}")

        # Save as a Parquet file.
        cleaned_parquet_path = os.path.join(CLEANED_DATA_FOLDER, f"cleaned_{league}_{season}.parquet")
        df.to_parquet(cleaned_parquet_path, index=False)
        logging.info(f"Cleaned Parquet saved to: {cleaned_parquet_path}")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")


def process_single_file(file_name: str) -> None:
    """
    Process a single CSV file based on its naming convention.

    The file name is expected to have the format: <league>_<season>_...

    Args:
        file_name (str): Name of the CSV file to process.
    """
    if file_name.endswith('.csv'):
        try:
            # Extract league and season from the file name.
            parts = Path(file_name).stem.split('_')
            if len(parts) >= 2:
                league, season = parts[0], parts[1]
            else:
                league, season = "unknown", "unknown"
            file_path = os.path.join(RAW_DATA_FOLDER, file_name)
            preprocess_file(file_path, league, season)
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")


def process_all_files(input_folder: str) -> None:
    """
    Process all CSV files in the specified input folder using multi-core processing.

    Args:
        input_folder (str): Path to the folder containing raw CSV files.
    """
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    with Pool(processes=4) as pool:
        pool.map(process_single_file, files)


# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_all_files(RAW_DATA_FOLDER)
