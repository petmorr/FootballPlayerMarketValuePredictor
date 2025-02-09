import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Logger & Paths Setup
# ------------------------------------------------------------------------------
logger = configure_logger("preprocessing", "preprocessing.log")

RAW_DATA_FOLDER: str = './data/scraped'
CLEANED_DATA_FOLDER: str = './data/cleaned'
os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)

# ------------------------------------------------------------------------------
# Expected Standardized Column Names (in order)
# ------------------------------------------------------------------------------
EXPECTED_COLUMNS_ORDER = [
    'rank', 'player', 'country_code', 'position', 'squad', 'age', 'born',
    'matches_played', 'starts', 'minutes_played', '90s_played', 'goals',
    'assists', 'goals_+_assists', 'goals_-_penalties', 'penalty_kicks_made',
    'penalty_kicks_attempted', 'yellow_cards', 'red_cards', 'expected_goals',
    'non-penalty_xg', 'expected_assists', 'non-penalty_xg_+_xag',
    'progressive_carries', 'progressive_passes', 'progressive_receives',
    'goals_per_90', 'assists_per_90', 'goals_+_assists_per_90',
    'goals_-_penalties_per_90', 'goals_+_assists_-_penalties_per_90',
    'xg_per_90', 'xag_per_90', 'xg_+_xag_per_90',
    'non-penalty_xg_per_90', 'non-penalty_xg_+_xag_per_90',
    'matches'
]

# Define the set of essential columns (used in integrity checks)
ESSENTIAL_COLUMNS = {
    'rank', 'player', 'country_code', 'position', 'squad', 'age',
    'born', 'matches_played', 'minutes_played', 'goals'
}

# ------------------------------------------------------------------------------
# Country Code Mapping (updated to include SUI)
# ------------------------------------------------------------------------------
COUNTRY_CODE_MAPPING = {
    'NGA': 'Nigeria', 'BRA': 'Brazil', 'ENG': 'England', 'FRA': 'France',
    'GER': 'Germany', 'ESP': 'Spain', 'ITA': 'Italy', 'ARG': 'Argentina',
    'USA': 'United States', 'MEX': 'Mexico', 'POR': 'Portugal', 'NED': 'Netherlands',
    'BEL': 'Belgium', 'SCO': 'Scotland', 'WAL': 'Wales', 'DEN': 'Denmark',
    'SWE': 'Sweden', 'NOR': 'Norway', 'FIN': 'Finland', 'ISL': 'Iceland',
    'CRO': 'Croatia', 'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia',
    'CZE': 'Czech Republic', 'POL': 'Poland', 'HUN': 'Hungary', 'AUT': 'Austria',
    'SWI': 'Switzerland', 'SUI': 'Switzerland',  # both variants now
    'GRE': 'Greece', 'TUR': 'Turkey', 'ROU': 'Romania',
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
    'LUX': 'Luxembourg', 'TOG': 'Togo', 'SUR': 'Suriname',
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
    'LBY': 'Libya', 'SLE': 'Sierra Leone', 'CYP': 'Cyprus', 'LVA': 'Latvia',
    'N/A': 'Not Available', 'NAN': 'Not Available', 'NA': 'Not Available'
}

# ------------------------------------------------------------------------------
# Helper Functions for Data Processing
# ------------------------------------------------------------------------------

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a multi-index header by joining levels with an underscore."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join([str(item).strip() for item in col if item and str(item).lower() != 'nan'])
            for col in df.columns.values
        ]
    return df

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lower-case and replace spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=True)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the DataFrame columns to the standardized expected names.
    Expects that the DataFrame contains at least len(EXPECTED_COLUMNS_ORDER) columns,
    and that the last two columns are 'league' and 'season'.
    """
    total_expected = len(EXPECTED_COLUMNS_ORDER) + 2  # plus league and season
    if df.shape[1] < total_expected:
        raise ValueError("Not enough columns to rename")
    df = df.iloc[:, :total_expected]
    df.columns = EXPECTED_COLUMNS_ORDER + ['league', 'season']
    return df

def standardize_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unwanted characters and convert player names to lower-case."""
    if 'player' in df.columns:
        df['player'] = df['player'].apply(
            lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x).lower() if isinstance(x, str) else x
        )
    return df

def ensure_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns expected to be numeric into proper numeric types.
    Also converts object columns with few unique values to categorical.
    """
    non_numeric = {'player', 'country_code', 'position', 'squad', 'born', 'league', 'season'}
    numeric_cols = set(EXPECTED_COLUMNS_ORDER) - non_numeric

    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:
            df[col] = df[col].astype('category')
    return df

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with too many missing values and fill missing values.
    Numeric columns are filled with median and categorical with mode.
    """
    threshold = 0.3 * len(df)
    df.dropna(thresh=threshold, axis=1, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if not df[col].isnull().all():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple new features such as xAG overperformance."""
    if {'assists', 'expected_assists'}.issubset(df.columns):
        df['assists'] = pd.to_numeric(df['assists'], errors='coerce')
        df['expected_assists'] = pd.to_numeric(df['expected_assists'], errors='coerce')
        df['xag_overperformance'] = df['assists'] - df['expected_assists']
    return df

def additional_enhancements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unrealistic values and compute performance consistency.
    """
    if 'age' in df.columns:
        df = df[(df['age'] >= 16) & (df['age'] <= 45)]
    if 'minutes_played' in df.columns:
        df = df[df['minutes_played'] >= 0]
    if 'goals' in df.columns:
        df = df[df['goals'] >= 0]
    if 'goals_per_90' in df.columns and 'player' in df.columns:
        df['performance_consistency'] = df.groupby('player')['goals_per_90'].transform('std')
    return df

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features such as playing time metrics, offensive metrics,
    per-90 comparisons, non-penalty metrics, progressive metrics, and age features.
    """
    # Playing Time Metrics
    if 'minutes_played' in df.columns and 'matches_played' in df.columns:
        df['minutes_per_match'] = df['minutes_played'] / df['matches_played'].replace(0, np.nan)
        df['minutes_played_ratio'] = df['minutes_played'] / (df['matches_played'] * 90).replace(0, np.nan)
    if 'starts' in df.columns and 'matches_played' in df.columns:
        df['start_rate'] = df['starts'] / df['matches_played'].replace(0, np.nan)
    if '90s_played' in df.columns and 'matches_played' in df.columns:
        df['nineties_per_match'] = df['90s_played'] / df['matches_played'].replace(0, np.nan)

    # Offensive Contribution Metrics
    if 'assists' in df.columns and 'goals' in df.columns:
        df['assist_to_goal_ratio'] = df['assists'] / df['goals'].replace(0, np.nan)
    if 'penalty_kicks_made' in df.columns and 'penalty_kicks_attempted' in df.columns:
        df['penalty_conversion_rate'] = df['penalty_kicks_made'] / df['penalty_kicks_attempted'].replace(0, np.nan)
    if 'yellow_cards' in df.columns and 'red_cards' in df.columns:
        df['discipline_score'] = df['yellow_cards'] + 3 * df['red_cards']
    if 'goals' in df.columns and 'expected_goals' in df.columns:
        df['finishing_efficiency'] = df['goals'] / df['expected_goals'].replace(0, np.nan)
    if 'assists' in df.columns and 'expected_assists' in df.columns:
        df['assisting_efficiency'] = df['assists'] / df['expected_assists'].replace(0, np.nan)
    if {'goals', 'assists', 'expected_goals', 'expected_assists'}.issubset(df.columns):
        df['overall_efficiency'] = (df['goals'] + df['assists']) / (
                df['expected_goals'] + df['expected_assists']).replace(0, np.nan)
    if 'goals' in df.columns and 'expected_goals' in df.columns:
        df['goal_overperformance'] = df['goals'] - df['expected_goals']
    if 'assists' in df.columns and 'expected_assists' in df.columns:
        df['assist_overperformance'] = df['assists'] - df['expected_assists']
    if {'goals', 'assists', 'expected_goals', 'expected_assists'}.issubset(df.columns):
        df['overall_overperformance'] = (df['goals'] + df['assists']) - (
                df['expected_goals'] + df['expected_assists'])
    if 'goals' in df.columns and 'assists' in df.columns:
        df['goal_ratio'] = df['goals'] / (df['goals'] + df['assists']).replace(0, np.nan)

    # Per 90 Comparison Metrics
    if 'goals_per_90' in df.columns and 'xg_per_90' in df.columns:
        df['per90_goal_diff'] = df['goals_per_90'] - df['xg_per_90']
        df['conversion_rate_per90'] = df['goals_per_90'] / df['xg_per_90'].replace(0, np.nan)
    if 'assists_per_90' in df.columns and 'xag_per_90' in df.columns:
        df['per90_assist_diff'] = df['assists_per_90'] - df['xag_per_90']
        df['assist_conversion_per90'] = df['assists_per_90'] / df['xag_per_90'].replace(0, np.nan)
    if {'goals_+_assists_per_90', 'xg_+_xag_per_90'}.issubset(df.columns):
        df['per90_overall_diff'] = df['goals_+_assists_per_90'] - df['xg_+_xag_per_90']

    # Non-Penalty Metrics
    if {'goals_+_assists_-_penalties_per_90', 'non-penalty_xg_+_xag_per_90'}.issubset(df.columns):
        df['non_penalty_overperformance'] = df['goals_+_assists_-_penalties_per_90'] - df['non-penalty_xg_+_xag_per_90']
        if 'goals_+_assists_per_90' in df.columns:
            df['penalty_impact'] = df['goals_+_assists_per_90'] - df['goals_+_assists_-_penalties_per_90']

    # Progressive Metrics
    if {'progressive_carries', 'progressive_passes', 'progressive_receives', 'matches_played'}.issubset(df.columns):
        df['progressive_total'] = (df['progressive_carries'] +
                                   df['progressive_passes'] +
                                   df['progressive_receives'])
        df['progressive_actions_per_match'] = df['progressive_total'] / df['matches_played'].replace(0, np.nan)
        df['progressive_carries_ratio'] = df['progressive_carries'] / df['progressive_total'].replace(0, np.nan)
        df['progressive_passes_ratio'] = df['progressive_passes'] / df['progressive_total'].replace(0, np.nan)
        df['progressive_receives_ratio'] = df['progressive_receives'] / df['progressive_total'].replace(0, np.nan)

    # Age & Born Metrics
    if 'age' in df.columns:
        df['age_squared'] = df['age'] ** 2
    if 'born' in df.columns:
        try:
            df['birth_year'] = pd.to_datetime(df['born'], errors='coerce').dt.year
        except Exception as e:
            logger.warning("Could not parse 'born' column to extract birth_year: " + str(e))

    return df


def finalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    After all processing and feature engineering, fill any remaining numeric NaN values
    with 0.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def clean_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the country code column by extracting the three-letter code.
    For example, a value like 'ng NGA' will become 'NGA'. Also, if the value is
    a header-like value ('Nation'), it returns NaN.
    This version splits the string by whitespace and takes the last token.
    """
    if 'country_code' in df.columns:
        def extract_code(x):
            s = str(x).strip()
            if s.lower() == 'nation':
                return np.nan
            parts = s.split()
            return parts[-1].upper() if parts else s.upper()

        df['country_code'] = df['country_code'].apply(extract_code)
    return df

def data_integrity_checks(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Verify that essential columns exist, log any unmapped country codes,
    and warn about duplicate records.
    """
    if 'country_code' in df.columns:
        unmapped = df[~df['country_code'].isin(COUNTRY_CODE_MAPPING.keys())]
        if not unmapped.empty:
            logger.warning(f"Unmapped country codes in {file_path}: {unmapped['country_code'].unique()}")
    missing_ess = ESSENTIAL_COLUMNS - set(df.columns)
    if missing_ess:
        logger.error(f"Missing essential columns in {file_path}: {missing_ess}")
        raise ValueError("Essential columns missing, stopping processing.")
    if {'player', 'squad', 'season'}.issubset(df.columns):
        duplicates = df[df.duplicated(subset=['player', 'squad', 'season'], keep=False)]
        if not duplicates.empty:
            logger.warning(f"Duplicate records in {file_path}: "
                           f"{duplicates[['player', 'squad', 'season']].drop_duplicates().to_dict(orient='records')}")
    return df


def standardize_country_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the raw file has a 'nation' column (with three-letter codes), rename it to 'country_code'.
    """
    if 'nation' in df.columns and 'country_code' not in df.columns:
        df.rename(columns={'nation': 'country_code'}, inplace=True)
    return df


def remove_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are repeated header rows.
    For example, drop any row where the 'player' column (after stripping and lowercasing)
    is exactly 'player'.
    """
    if 'player' in df.columns:
        df = df[df['player'].astype(str).str.strip().str.lower() != 'player']
    return df

def preprocess_file(file_path: str, league: str, season: str) -> None:
    try:
        logger.info(f"Processing file: {file_path}")
        try:
            # Attempt multi-index header read:
            df = pd.read_csv(file_path, header=[0, 1])
            df = flatten_columns(df)
            df = normalize_column_names(df)
            df = standardize_country_column(df)
            df = remove_header_rows(df)
            # In the multi-index branch, rename the columns first...
            if df.shape[1] == len(EXPECTED_COLUMNS_ORDER) + 2:
                df = rename_columns(df)
                # Clean the 'country_code' column after renaming
                df = clean_country_codes(df)
            else:
                raise ValueError("Multi-index read did not produce the expected number of columns.")
        except Exception as e:
            logger.warning(f"Multi-level header read failed ({e}). Trying single header.")
            df = pd.read_csv(file_path, header=0)
            df = normalize_column_names(df)
            df = standardize_country_column(df)
            df = remove_header_rows(df)
            if df.shape[1] >= len(EXPECTED_COLUMNS_ORDER) + 2:
                df = df.iloc[:, :len(EXPECTED_COLUMNS_ORDER) + 2]
                df.columns = EXPECTED_COLUMNS_ORDER + ['league', 'season']
                # Clean the country codes after renaming
                df = clean_country_codes(df)
            else:
                raise ValueError("Single header read did not produce enough columns.")

        # Ensure no header rows remain and drop duplicates.
        df = remove_header_rows(df)
        df = df.drop_duplicates(subset=['player', 'squad', 'season'])

        # Process DataFrame: convert types, handle missing data, and create features.
        df = ensure_data_types(df)
        df = handle_missing_data(df)
        df = standardize_player_names(df)
        df = feature_engineering(df)
        df = additional_enhancements(df)
        df = advanced_feature_engineering(df)
        df = finalize_data(df)

        # Map full nation name if not already provided.
        if 'country_code' in df.columns and 'country' not in df.columns:
            # Convert to string so that fillna does not try to set a new category.
            mapped = df['country_code'].astype(str).map(COUNTRY_CODE_MAPPING)
            df['country'] = mapped.astype(object).fillna('Not Available')

        # Overwrite metadata columns with provided values.
        df['league'] = league
        df['season'] = season

        # Run integrity checks.
        df = data_integrity_checks(df, file_path)

        # Save to CSV (gzip) and Parquet.
        cleaned_csv_path = os.path.join(CLEANED_DATA_FOLDER, f"cleaned_{league}_{season}.csv.gz")
        df.to_csv(cleaned_csv_path, index=False, float_format="%.4f", compression='gzip')
        logger.info(f"Cleaned CSV saved to: {cleaned_csv_path}")

        cleaned_parquet_path = os.path.join(CLEANED_DATA_FOLDER, f"cleaned_{league}_{season}.parquet")
        df.to_parquet(cleaned_parquet_path, index=False)
        logger.info(f"Cleaned Parquet saved to: {cleaned_parquet_path}")

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

def process_single_file(file_name: str) -> None:
    """Process a single CSV file, expecting a filename format like <league>_<season>_..."""
    if file_name.endswith('.csv'):
        try:
            parts = Path(file_name).stem.split('_')
            if len(parts) >= 2:
                league, season = parts[0], parts[1]
            else:
                league, season = "unknown", "unknown"
            file_path = os.path.join(RAW_DATA_FOLDER, file_name)
            preprocess_file(file_path, league, season)
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")

def process_all_files(input_folder: str) -> None:
    """Process all CSV files in the given folder using multiprocessing."""
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    with Pool(processes=4) as pool:
        pool.map(process_single_file, files)

# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_all_files(RAW_DATA_FOLDER)