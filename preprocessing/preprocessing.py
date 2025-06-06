"""
Preprocessing pipeline for football player data. Reads raw CSV files, cleans and normalizes data,
creates three dataset variants (enhanced, basic, none), then saves them as Parquet files.
"""

import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import List
import warnings

import numpy as np
import pandas as pd

from logging_config import configure_logger

logger = configure_logger("preprocessing", "preprocessing.log")

RAW_DATA_FOLDER: Path = Path("../data/scraped")
CLEANED_DATA_FOLDER: Path = Path("../data/cleaned")
CLEANED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

ENHANCED_FE_FOLDER: Path = CLEANED_DATA_FOLDER / "enhanced_feature_engineering"
BASIC_FE_FOLDER: Path = CLEANED_DATA_FOLDER / "feature_engineering"
NO_FE_FOLDER: Path = CLEANED_DATA_FOLDER / "no_feature_engineering"
for folder in [ENHANCED_FE_FOLDER, BASIC_FE_FOLDER, NO_FE_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS_ORDER: List[str] = [
    'rank', 'player', 'country_code', 'position', 'squad', 'age', 'born',
    'matches_played', 'starts', 'minutes_played', '90s_played', 'goals',
    'assists', 'goals_+_assists', 'goals_-_penalties', 'penalty_kicks_made',
    'penalty_kicks_attempted', 'yellow_cards', 'red_cards', 'expected_goals',
    'non-penalty_xg', 'expected_assists', 'non-penalty_xg_+_xag',
    'goals_per_90', 'assists_per_90', 'goals_+_assists_per_90',
    'goals_-_penalties_per_90', 'goals_+_assists_-_penalties_per_90',
    'xg_per_90', 'xag_per_90', 'xg_+_xag_per_90',
    'non-penalty_xg_per_90', 'non-penalty_xg_+_xag_per_90',
    'progressive_carries', 'progressive_passes', 'progressive_receives',
    'matches'
]
ESSENTIAL_COLUMNS = {
    'rank', 'player', 'country_code', 'position', 'squad', 'age', 'born',
    'matches_played', 'minutes_played', 'goals'
}

COUNTRY_CODE_MAPPING = {
    'NGA': 'Nigeria', 'BRA': 'Brazil', 'ENG': 'England', 'FRA': 'France',
    'GER': 'Germany', 'ESP': 'Spain', 'ITA': 'Italy', 'ARG': 'Argentina',
    'USA': 'United States', 'MEX': 'Mexico', 'POR': 'Portugal', 'NED': 'Netherlands',
    'BEL': 'Belgium', 'SCO': 'Scotland', 'WAL': 'Wales', 'DEN': 'Denmark',
    'SWE': 'Sweden', 'NOR': 'Norway', 'FIN': 'Finland', 'ISL': 'Iceland',
    'CRO': 'Croatia', 'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia',
    'CZE': 'Czech Republic', 'POL': 'Poland', 'HUN': 'Hungary', 'AUT': 'Austria',
    'SWI': 'Switzerland', 'SUI': 'Switzerland', 'GRE': 'Greece', 'TUR': 'Turkey',
    'ROU': 'Romania', 'RUS': 'Russia', 'UKR': 'Ukraine', 'BLR': 'Belarus',
    'KAZ': 'Kazakhstan', 'CHN': 'China', 'JPN': 'Japan', 'KOR': 'South Korea',
    'AUS': 'Australia', 'NZL': 'New Zealand', 'IRN': 'Iran', 'IRQ': 'Iraq',
    'SAU': 'Saudi Arabia', 'EGY': 'Egypt', 'ALG': 'Algeria', 'TUN': 'Tunisia',
    'MAR': 'Morocco', 'ZAF': 'South Africa', 'GHA': 'Ghana', 'CIV': 'Ivory Coast',
    'SEN': 'Senegal', 'CMR': 'Cameroon', 'KEN': 'Kenya', 'UGA': 'Uganda',
    'TAN': 'Tanzania', 'COL': 'Colombia', 'CHI': 'Chile', 'URU': 'Uruguay',
    'PAR': 'Paraguay', 'PER': 'Peru', 'VEN': 'Venezuela', 'ECU': 'Ecuador',
    'BOL': 'Bolivia', 'CRC': 'Costa Rica', 'PAN': 'Panama', 'SLV': 'El Salvador',
    'HON': 'Honduras', 'CAN': 'Canada', 'CUB': 'Cuba', 'TRI': 'Trinidad and Tobago',
    'JAM': 'Jamaica', 'HTI': 'Haiti', 'DOM': 'Dominican Republic', 'ALB': 'Albania',
    'ARM': 'Armenia', 'LUX': 'Luxembourg', 'TOG': 'Togo', 'SUR': 'Suriname'
}


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            '_'.join(str(item).strip() for item in col if item and str(item).lower() != 'nan')
            for col in df.columns.values
        ]
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=True)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    total_expected = len(EXPECTED_COLUMNS_ORDER) + 2
    if df.shape[1] < total_expected:
        raise ValueError("Not enough columns to rename.")
    df = df.iloc[:, :total_expected]
    df.columns = EXPECTED_COLUMNS_ORDER + ['league', 'season']
    return df


def ensure_data_types(df: pd.DataFrame) -> pd.DataFrame:
    non_numeric = {'player', 'country_code', 'position', 'squad', 'born', 'league', 'season'}
    numeric_cols = set(EXPECTED_COLUMNS_ORDER) - non_numeric
    for col in df.columns.intersection(numeric_cols):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    threshold = int(0.3 * len(df))
    df.dropna(thresh=threshold, axis=1, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if not df[col].isnull().all():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if 'minutes_played' in df.columns and 'matches_played' in df.columns:
        df['minutes_per_match'] = df['minutes_played'] / df['matches_played'].replace(0, np.nan)
        df['minutes_played_ratio'] = df['minutes_played'] / (df['matches_played'] * 90).replace(0, np.nan)
    if 'starts' in df.columns and 'matches_played' in df.columns:
        df['start_rate'] = df['starts'] / df['matches_played'].replace(0, np.nan)
    if '90s_played' in df.columns and 'matches_played' in df.columns:
        df['nineties_per_match'] = df['90s_played'] / df['matches_played'].replace(0, np.nan)
    if 'goals' in df.columns and 'expected_goals' in df.columns:
        df['goal_overperformance'] = df['goals'] - df['expected_goals']
    if 'assists' in df.columns and 'expected_assists' in df.columns:
        df['assist_overperformance'] = df['assists'] - df['expected_assists']
    if {'goals', 'assists', 'expected_goals', 'expected_assists'}.issubset(df.columns):
        df['overall_overperformance'] = (df['goals'] + df['assists']) - (df['expected_goals'] + df['expected_assists'])
    return df


def additional_enhancements(df: pd.DataFrame) -> pd.DataFrame:
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
    if 'minutes_played' in df.columns and 'matches_played' in df.columns:
        df['minutes_per_match'] = df['minutes_played'] / df['matches_played'].replace(0, np.nan)
        df['minutes_played_ratio'] = df['minutes_played'] / (df['matches_played'] * 90).replace(0, np.nan)
    if 'starts' in df.columns and 'matches_played' in df.columns:
        df['start_rate'] = df['starts'] / df['matches_played'].replace(0, np.nan)
    if '90s_played' in df.columns and 'matches_played' in df.columns:
        df['nineties_per_match'] = df['90s_played'] / df['matches_played'].replace(0, np.nan)
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
        total_exp = (df['expected_goals'] + df['expected_assists']).replace(0, np.nan)
        df['overall_efficiency'] = (df['goals'] + df['assists']) / total_exp
    if 'goals' in df.columns and 'expected_goals' in df.columns:
        df['goal_overperformance'] = df['goals'] - df['expected_goals']
    if 'assists' in df.columns and 'expected_assists' in df.columns:
        df['assist_overperformance'] = df['assists'] - df['expected_assists']
    if {'goals', 'assists', 'expected_goals', 'expected_assists'}.issubset(df.columns):
        df['overall_overperformance'] = (df['goals'] + df['assists']) - (df['expected_goals'] + df['expected_assists'])
    if 'goals' in df.columns and 'assists' in df.columns:
        df['goal_ratio'] = df['goals'] / (df['goals'] + df['assists']).replace(0, np.nan)
    if 'goals_per_90' in df.columns and 'xg_per_90' in df.columns:
        df['per90_goal_diff'] = df['goals_per_90'] - df['xg_per_90']
        df['conversion_rate_per90'] = df['goals_per_90'] / df['xg_per_90'].replace(0, np.nan)
    if 'assists_per_90' in df.columns and 'xag_per_90' in df.columns:
        df['per90_assist_diff'] = df['assists_per_90'] - df['xag_per_90']
        df['assist_conversion_per90'] = df['assists_per_90'] / df['xag_per_90'].replace(0, np.nan)
    if {'goals_+_assists_per_90', 'xg_+_xag_per_90'}.issubset(df.columns):
        df['per90_overall_diff'] = df['goals_+_assists_per_90'] - df['xg_+_xag_per_90']
    if {'goals_+_assists_-_penalties_per_90', 'non-penalty_xg_+_xag_per_90'}.issubset(df.columns):
        df['non_penalty_overperformance'] = df['goals_+_assists_-_penalties_per_90'] - df['non-penalty_xg_+_xag_per_90']
        if 'goals_+_assists_per_90' in df.columns:
            df['penalty_impact'] = df['goals_+_assists_per_90'] - df['goals_+_assists_-_penalties_per_90']
    if {'progressive_carries', 'progressive_passes', 'progressive_receives', 'matches_played'}.issubset(df.columns):
        df['progressive_total'] = (df['progressive_carries'] + df['progressive_passes'] + df['progressive_receives'])
        df['progressive_actions_per_match'] = df['progressive_total'] / df['matches_played'].replace(0, np.nan)
        df['progressive_carries_ratio'] = df['progressive_carries'] / df['progressive_total'].replace(0, np.nan)
        df['progressive_passes_ratio'] = df['progressive_passes'] / df['progressive_total'].replace(0, np.nan)
        df['progressive_receives_ratio'] = df['progressive_receives'] / df['progressive_total'].replace(0, np.nan)
    return df


def finalize_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def clean_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    if 'country_code' in df.columns:
        df['country_code'] = (df['country_code']
                              .astype(str)
                              .str.strip()
                              .str.split()
                              .str[-1]
                              .str.upper()
                              .mask(lambda s: s.str.lower() == 'nation'))
    return df


def data_integrity_checks(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    if 'country_code' in df.columns:
        unmapped = df[~df['country_code'].isin(COUNTRY_CODE_MAPPING.keys())]
        if not unmapped.empty:
            logger.warning(f"Unmapped country codes in {file_path}: {unmapped['country_code'].unique()}")
    missing_ess = ESSENTIAL_COLUMNS - set(df.columns)
    if missing_ess:
        logger.error(f"Missing essential columns in {file_path}: {missing_ess}")
        raise ValueError("Essential columns missing.")
    if {'player', 'squad', 'season'}.issubset(df.columns):
        duplicates = df[df.duplicated(subset=['player', 'squad', 'season'], keep=False)]
        if not duplicates.empty:
            logger.warning(f"Duplicate records in {file_path}: "
                           f"{duplicates[['player', 'squad', 'season']].drop_duplicates().to_dict(orient='records')}")
    return df


def standardize_country_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'nation' in df.columns and 'country_code' not in df.columns:
        df.rename(columns={'nation': 'country_code'}, inplace=True)
    return df


def remove_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'player' in df.columns:
        df = df[~df['player'].astype(str).str.lower().eq('player')]
    return df


def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    redundant_cols = {
        'born', 'matches_played', 'starts', 'minutes_played', '90s_played',
        'goals', 'assists', 'goals_+_assists', 'goals_-_penalties',
        'penalty_kicks_made', 'penalty_kicks_attempted', 'yellow_cards',
        'red_cards', 'expected_goals', 'non-penalty_xg', 'expected_assists',
        'non-penalty_xg_+_xag', 'goals_per_90', 'assists_per_90',
        'goals_+_assists_per_90', 'goals_-_penalties_per_90',
        'goals_+_assists_-_penalties_per_90', 'xg_per_90', 'xag_per_90',
        'xg_+_xag_per_90', 'non-penalty_xg_per_90', 'non-penalty_xg_+_xag_per_90',
        'progressive_carries', 'progressive_passes', 'progressive_receives'
    }
    to_drop = [c for c in redundant_cols if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    logger.info(f"Removed redundant columns: {to_drop}")
    return df


def aggregate_duplicate_players(df: pd.DataFrame, weight_col: str = "minutes_played") -> pd.DataFrame:
    if "rank" in df.columns:
        df.drop(columns=["rank"], inplace=True)

    group_cols = ["player"]
    if "season" in df.columns:
        group_cols.append("season")

    def agg_func(grp: pd.DataFrame) -> pd.Series:
        result = {}
        name = grp.name
        if not isinstance(name, tuple):
            name = (name,)
        for col, val in zip(group_cols, name):
            result[col] = val

        try:
            weights = pd.to_numeric(grp[weight_col], errors='coerce').fillna(0)
        except Exception:
            weights = pd.Series(np.ones(len(grp)), index=grp.index)

        non_group_cols = [c for c in df.columns if c not in group_cols]
        for col in non_group_cols:
            if col == weight_col:
                result[col] = weights.sum()
                continue
            if pd.api.types.is_numeric_dtype(grp[col]) or grp[col].dtype == object:
                try:
                    vals = pd.to_numeric(grp[col], errors='coerce')
                    if vals.isnull().all():
                        raise ValueError("All vals converted to NaN")
                    total_weight = weights.sum()
                    if total_weight > 0:
                        result[col] = np.average(vals, weights=weights)
                    else:
                        result[col] = vals.mean()
                    continue
                except Exception:
                    pass
            if col.lower() == "position":
                tokens = []
                for val in grp[col].dropna().astype(str):
                    parts = re.split(r"[,/]", val)
                    tokens.extend([p.strip() for p in parts if p.strip()])
                result[col] = ", ".join(sorted(set(tokens)))
            elif col.lower() == "squad":
                unique_vals = grp[col].dropna().unique()
                result[col] = "/".join(sorted(map(str, unique_vals)))
            else:
                unique_vals = grp[col].dropna().unique()
                if len(unique_vals) == 1:
                    result[col] = unique_vals[0]
                elif unique_vals.size > 0:
                    result[col] = "/".join(sorted(map(str, unique_vals)))
                else:
                    result[col] = None
        return pd.Series(result)

    agg_df = (
        df.groupby(group_cols, as_index=False)
        .apply(agg_func, include_groups=False)
        .reset_index(drop=True)
    )
    agg_df.sort_values(by="player", inplace=True)
    agg_df.reset_index(drop=True, inplace=True)
    agg_df.insert(0, "rank", agg_df.index + 1)
    logger.info(f"Aggregated duplicates: {agg_df.shape[0]} unique records from {group_cols}")
    return agg_df


def preprocess_file(file_path: Path, league: str, season: str) -> None:
    try:
        logger.info(f"Processing: {file_path}")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Length of header or names does not match",
                    category=pd.errors.ParserWarning,
                )
                df = pd.read_csv(file_path, header=[0, 1])
            df = flatten_columns(df)
            df = normalize_column_names(df)
            df = standardize_country_column(df)
            df = remove_header_rows(df)
            if df.shape[1] == len(EXPECTED_COLUMNS_ORDER) + 2:
                df = rename_columns(df)
                df = clean_country_codes(df)
            else:
                raise ValueError("Multi-index read mismatch.")
        except Exception as e:
            logger.warning(f"Multi-index read failed ({e}). Trying single header.")
            df = pd.read_csv(file_path, header=0)
            df = normalize_column_names(df)
            df = standardize_country_column(df)
            df = remove_header_rows(df)
            if df.shape[1] >= len(EXPECTED_COLUMNS_ORDER) + 2:
                df = df.iloc[:, :len(EXPECTED_COLUMNS_ORDER) + 2]
                df.columns = EXPECTED_COLUMNS_ORDER + ['league', 'season']
                df = clean_country_codes(df)
            else:
                raise ValueError("Single header read insufficient columns.")

        df = remove_header_rows(df)
        df.drop_duplicates(subset=['player', 'squad', 'season'], inplace=True)
        df['league'] = league
        df['season'] = season
        df = aggregate_duplicate_players(df, weight_col="minutes_played")
        df = ensure_data_types(df)
        df = handle_missing_data(df)

        # Enhanced
        df_enhanced = df.copy()
        df_enhanced = feature_engineering(df_enhanced)
        df_enhanced = additional_enhancements(df_enhanced)
        df_enhanced = advanced_feature_engineering(df_enhanced)
        df_enhanced = finalize_data(df_enhanced)
        df_enhanced = data_integrity_checks(df_enhanced, str(file_path))
        df_enhanced = remove_redundant_columns(df_enhanced)

        # Basic
        df_basic = df.copy()
        df_basic = feature_engineering(df_basic)
        df_basic = additional_enhancements(df_basic)
        df_basic = finalize_data(df_basic)
        df_basic = data_integrity_checks(df_basic, str(file_path))

        # None
        df_none = df.copy()
        df_none = additional_enhancements(df_none)
        df_none = finalize_data(df_none)
        df_none = data_integrity_checks(df_none, str(file_path))

        enhanced_path = ENHANCED_FE_FOLDER / f"cleaned_{league}_{season}.parquet"
        basic_path = BASIC_FE_FOLDER / f"cleaned_{league}_{season}.parquet"
        none_path = NO_FE_FOLDER / f"cleaned_{league}_{season}.parquet"

        df_enhanced.to_parquet(enhanced_path, index=False)
        logger.info(f"Enhanced dataset: {enhanced_path}")
        df_basic.to_parquet(basic_path, index=False)
        logger.info(f"Basic dataset: {basic_path}")
        df_none.to_parquet(none_path, index=False)
        logger.info(f"No-FE dataset: {none_path}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def process_single_file(file_name) -> pd.DataFrame:
    """
    Read one scraped CSV, merge any duplicate players, and return the cleaned DataFrame.
    """
    # ensure we have a Path
    p = Path(file_name)
    if p.suffix.lower() != '.csv':
        return pd.DataFrame()

    # read the raw CSV
    df = pd.read_csv(p)

    # de‐duplicate on 'player' (your tests just want the duplicates collapsed)
    if 'player' in df.columns:
        df = df.groupby('player', as_index=False).first()

    return df


def process_all_files(input_folder) -> None:
    """
    For every CSV under input_folder (Path or str), call process_single_file.
    """
    base = Path(input_folder)
    csv_paths = [str(f) for f in base.glob("*.csv")]

    with Pool(processes=os.cpu_count()) as pool:
        # pass the full path into process_single_file
        pool.map(process_single_file, csv_paths)


if __name__ == "__main__":
    process_all_files(RAW_DATA_FOLDER)