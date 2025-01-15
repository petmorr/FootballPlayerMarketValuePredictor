import os
from pathlib import Path

import pandas as pd

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logging/preprocessing.log", mode="a"),
        logging.StreamHandler()
    ]
)

# Define paths
RAW_DATA_FOLDER = './data/scraped'
CLEANED_DATA_FOLDER = './data/cleaned'
os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)

# Country code to full name mapping
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
    'N/A': 'Not Available',  # Explicit mapping for missing codes
}

# Expected columns for validation
EXPECTED_COLUMNS = [
    'Rank', 'Player', 'Country Code', 'Country', 'Position', 'Squad',
    'Age', 'Born', 'Matches Played', 'Starts', 'Minutes Played', '90s Played',
    'Goals', 'Assists', 'Goals + Assists', 'Goals - Penalties',
    'Penalty Kicks Made', 'Penalty Kicks Attempted', 'Yellow Cards',
    'Red Cards', 'Expected Goals', 'Non-Penalty xG', 'Expected Assists',
    'Non-Penalty xG + xAG', 'Progressive Carries', 'Progressive Passes',
    'Progressive Receives', 'Goals per 90', 'Assists per 90',
    'Goals + Assists per 90', 'Goals - Penalties per 90',
    'Goals + Assists - Penalties per 90', 'xG per 90', 'xAG per 90',
    'xG + xAG per 90', 'Non-Penalty xG per 90', 'Non-Penalty xG + xAG per 90'
]


def preprocess_file(file_path: str, league: str, season: str) -> None:
    """
    Preprocess a raw CSV file and save the cleaned version.

    Args:
        file_path (str): Path to the raw CSV file.
        league (str): League name (e.g., 'Bundesliga').
        season (str): Season year (e.g., '2019-2020').

    Returns:
        None
    """
    try:
        logging.info(f"Processing file: {file_path}")

        # Read and flatten multi-level headers
        df = pd.read_csv(file_path, header=[0, 1])
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(how='all', inplace=True)
        df = df[df[df.columns[0]] != 'Rk']
        df.reset_index(drop=True, inplace=True)

        # Rename columns
        column_mapping = {
            'Unnamed: 0_level_0_Rk': 'Rank',
            'Unnamed: 1_level_0_Player': 'Player',
            'Unnamed: 2_level_0_Nation': 'Nation',
            'Unnamed: 3_level_0_Pos': 'Position',
            'Unnamed: 4_level_0_Squad': 'Squad',
            'Unnamed: 5_level_0_Age': 'Age',
            'Unnamed: 6_level_0_Born': 'Born',
            'Playing Time_MP': 'Matches Played',
            'Playing Time_Starts': 'Starts',
            'Playing Time_Min': 'Minutes Played',
            'Playing Time_90s': '90s Played',
            'Performance_Gls': 'Goals',
            'Performance_Ast': 'Assists',
            'Performance_G+A': 'Goals + Assists',
            'Performance_G-PK': 'Goals - Penalties',
            'Performance_PK': 'Penalty Kicks Made',
            'Performance_PKatt': 'Penalty Kicks Attempted',
            'Performance_CrdY': 'Yellow Cards',
            'Performance_CrdR': 'Red Cards',
            'Expected_xG': 'Expected Goals',
            'Expected_npxG': 'Non-Penalty xG',
            'Expected_xAG': 'Expected Assists',
            'Expected_npxG+xAG': 'Non-Penalty xG + xAG',
            'Progression_PrgC': 'Progressive Carries',
            'Progression_PrgP': 'Progressive Passes',
            'Progression_PrgR': 'Progressive Receives',
            'Per 90 Minutes_Gls': 'Goals per 90',
            'Per 90 Minutes_Ast': 'Assists per 90',
            'Per 90 Minutes_G+A': 'Goals + Assists per 90',
            'Per 90 Minutes_G-PK': 'Goals - Penalties per 90',
            'Per 90 Minutes_G+A-PK': 'Goals + Assists - Penalties per 90',
            'Per 90 Minutes_xG': 'xG per 90',
            'Per 90 Minutes_xAG': 'xAG per 90',
            'Per 90 Minutes_xG+xAG': 'xG + xAG per 90',
            'Per 90 Minutes_npxG': 'Non-Penalty xG per 90',
            'Per 90 Minutes_npxG+xAG': 'Non-Penalty xG + xAG per 90',
            'Unnamed: 36_level_0_Matches': None  # Remove this column
        }
        df.rename(columns=column_mapping, inplace=True)

        # Handle Nation column
        if 'Nation' in df.columns:
            df['Country Code'] = df['Nation'].str.split(' ', n=1).str[1].fillna('N/A')
            df['Country'] = df['Country Code'].map(COUNTRY_CODE_MAPPING).fillna('Not Available')
            df.drop(columns=['Nation'], inplace=True)

        # Add league and season columns
        df['League'] = league
        df['Season'] = season

        # Validate column schema
        missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns:
            logging.error(f"Missing columns in {file_path}: {missing_columns}")
            return

        # Reorder columns
        df = df[EXPECTED_COLUMNS + ['League', 'Season']]

        # Save cleaned data
        cleaned_file_base = os.path.join(CLEANED_DATA_FOLDER, f"cleaned_{league}_{season}")
        df.to_csv(f"{cleaned_file_base}.csv", index=False)
        logging.info(f"Cleaned file saved to: {cleaned_file_base}.csv")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")


def process_all_files(input_folder: str, output_folder: str) -> None:
    """
    Process all CSV files in the input folder and save cleaned versions.

    Args:
        input_folder (str): Path to the raw data folder.
        output_folder (str): Path to save cleaned data.

    Returns:
        None
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            try:
                league, season = Path(file_name).stem.split('_')[:2]
                file_path = os.path.join(input_folder, file_name)
                logging.info(f"Starting processing for {file_name}")
                preprocess_file(file_path, league, season)
            except ValueError:
                logging.warning(f"Invalid file name format: {file_name}")
                continue

if __name__ == "__main__":
    process_all_files(RAW_DATA_FOLDER, CLEANED_DATA_FOLDER)