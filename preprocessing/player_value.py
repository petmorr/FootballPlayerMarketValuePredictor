import functools
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import ftfy
import pandas as pd
import requests
import unicodedata
from pandas import DataFrame
from rapidfuzz.fuzz import partial_ratio

from logging_config import configure_logger

# =============================================================================
# Global Configuration & Logging
# =============================================================================
PREPROC_VARIANT_FOLDERS = {
    "enhanced_feature_engineering": Path("../data/cleaned/enhanced_feature_engineering"),
    "feature_engineering": Path("../data/cleaned/feature_engineering"),
    "no_feature_engineering": Path("../data/cleaned/no_feature_engineering")
}
UPDATED_VARIANT_FOLDERS = {
    "enhanced_feature_engineering": Path("../data/updated/enhanced_feature_engineering"),
    "feature_engineering": Path("../data/updated/feature_engineering"),
    "no_feature_engineering": Path("../data/updated/no_feature_engineering")
}
for folder in UPDATED_VARIANT_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

TEMP_UPDATED_FOLDER = Path("../data/updated_temp")
TEMP_UPDATED_FOLDER.mkdir(parents=True, exist_ok=True)

CONFIG = {"transfermarkt_tld": "co.uk"}
API_BASE_URL: str = "http://localhost:8000"
DATE_FORMAT: str = "%Y-%m-%d"
MAX_API_RETRIES: int = 3

logging = configure_logger("player_value", "player_value.log")


# =============================================================================
# File I/O Helpers
# =============================================================================
def get_clean_basename(file_path: str) -> str:
    p = Path(file_path)
    return p.name.split('.')[0] if len(p.suffixes) > 1 else p.stem


def get_updated_filename_from_cleaned(filename: str) -> str:
    base = get_clean_basename(filename)
    if base.startswith("cleaned_"):
        return f"updated_{base[len('cleaned_'):]}{Path(filename).suffix}"
    else:
        return f"updated_{base}{Path(filename).suffix}"


def read_input_file(file_path: str) -> Optional[DataFrame]:
    p = Path(file_path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    logging.error(f"Unsupported file type: {file_path}")
    return None


def write_output_file(df: pd.DataFrame, output_folder: Path, original_filename: str) -> None:
    new_filename = get_updated_filename_from_cleaned(original_filename)
    output_path = output_folder / new_filename
    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info(f"Updated dataset saved to: {output_path}")


# =============================================================================
# Text & Date Utilities
# =============================================================================
def fix_encoding(s: str) -> str:
    try:
        return ftfy.fix_text(s)
    except Exception as e:
        logging.error(f"Encoding fix failed for '{s}': {e}")
        return s


def parse_date(date_str: Any) -> Optional[datetime]:
    if isinstance(date_str, datetime):
        return date_str
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except Exception as e:
        logging.error(f"Date parsing failed for '{date_str}': {e}")
        return None


# =============================================================================
# Name Normalization Helpers
# =============================================================================
def remove_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    try:
        name = fix_encoding(name).strip()
        name = remove_diacritics(name)
    except Exception as e:
        logging.error(f"Error during normalization for '{name}': {e}")
    name = name.replace("ß", "ss").lower()
    name = re.sub(r"[-']", "", name)
    name = re.sub(r"\s+", "", name)
    return name


def normalize_name_keep_spaces(name: str) -> str:
    try:
        name = fix_encoding(name).strip().lower()
        name = remove_diacritics(name)
    except Exception as e:
        logging.error(f"Error during normalization (keep spaces) for '{name}': {e}")
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def get_last_name(full_name: str) -> str:
    parts = full_name.split()
    return normalize_name(parts[-1]) if parts else ""


# =============================================================================
# Candidate Query Generator
# =============================================================================
def generate_candidate_queries(player_name: str) -> set:
    fixed = fix_encoding(player_name).strip()
    variants = set()
    variants.add(fixed)
    variants.add(fixed.lower())
    variants.add(re.sub(r"[^\w\s]", "", fixed))
    variants.add(re.sub(r"[^\w\s]", "", fixed.lower()))
    no_diacritics = remove_diacritics(fixed)
    variants.add(no_diacritics)
    variants.add(no_diacritics.lower())
    variants.add(re.sub(r"[^\w\s]", "", no_diacritics))
    variants.add(re.sub(r"[^\w\s]", "", no_diacritics.lower()))
    hyphen_space = fixed.replace("-", " ")
    variants.add(hyphen_space)
    variants.add(hyphen_space.lower())
    variants.add(re.sub(r"[^\w\s]", "", hyphen_space))
    variants.add(re.sub(r"[^\w\s]", "", hyphen_space.lower()))
    variants.add(normalize_name(player_name))
    variants.add(normalize_name_keep_spaces(player_name))
    parts = fixed.split()
    if len(parts) == 2:
        reversed_order = f"{parts[1]} {parts[0]}"
        variants.add(reversed_order)
        variants.add(reversed_order.lower())
        variants.add(re.sub(r"[^\w\s]", "", reversed_order))
        variants.add(re.sub(r"[^\w\s]", "", reversed_order.lower()))
    if len(parts) >= 3:
        hyphenated = f"{parts[0]}-{parts[1]} " + " ".join(parts[2:])
        variants.add(hyphenated)
        variants.add(hyphenated.lower())
        variants.add(re.sub(r"[^\w\s]", "", hyphenated))
        variants.add(re.sub(r"[^\w\s]", "", hyphenated.lower()))
    if len(parts) >= 2:
        initial_last = f"{parts[0][0]} {parts[-1]}"
        variants.add(initial_last)
        variants.add(initial_last.lower())
        variants.add(re.sub(r"[^\w\s]", "", initial_last))
        variants.add(re.sub(r"[^\w\s]", "", initial_last.lower()))
    if parts:
        last = parts[-1]
        variants.add(last)
        variants.add(last.lower())
        variants.add(re.sub(r"[^\w\s]", "", last))
        variants.add(re.sub(r"[^\w\s]", "", last.lower()))
        variants.add(normalize_name(last))
        variants.add(normalize_name_keep_spaces(last))
    return variants


# =============================================================================
# API Request Functions (with Caching)
# =============================================================================
session = requests.Session()


def make_request_with_retry(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/{endpoint}"
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            logging.debug(f"Successful API call on attempt {attempt}: {url}")
            return response.json()
        except requests.RequestException as e:
            logging.warning(f"API call failed (attempt {attempt}/{MAX_API_RETRIES}): {e}")
            time.sleep(min(2 ** attempt, 60))
    return {}


def fetch_player_market_value(player_id: str) -> Optional[List[Dict[str, Any]]]:
    endpoint = f"players/{player_id}/market_value"
    logging.info(f"Fetching market value for player ID {player_id}.")
    return make_request_with_retry(endpoint).get("marketValueHistory", [])


@functools.lru_cache(maxsize=1000)
def fetch_player_profile(player_id: str) -> Optional[str]:
    endpoint = f"players/{player_id}/profile"
    logging.info(f"Fetching profile URL for player ID {player_id}.")
    data = make_request_with_retry(endpoint)
    return data.get("url", None)


# =============================================================================
# Name-Based Search Functions
# =============================================================================
def fetch_player_id_by_last_name(player_name: str, team_name: str) -> Optional[str]:
    input_last = get_last_name(player_name)
    if not input_last:
        return None
    endpoint = f"players/search/{input_last}"
    logging.info(f"Fallback search by last name for '{input_last}'.")
    search_results = make_request_with_retry(endpoint).get("results", [])
    best_candidate, best_score = None, 0
    for result in search_results:
        candidate_last = get_last_name(result.get("name", ""))
        if candidate_last != input_last:
            continue
        candidate_club = result.get("club", {}).get("name", "")
        score = partial_ratio(team_name.lower(), candidate_club.lower())
        if score > best_score:
            best_score = score
            best_candidate = result
    if best_candidate and best_score >= 80:
        logging.info(
            f"Fallback search found candidate {best_candidate['id']} for '{input_last}' with score {best_score}.")
        return best_candidate["id"]
    logging.warning(f"Fallback search by last name for '{input_last}' found no suitable candidate.")
    return None


@functools.lru_cache(maxsize=1000)
def fetch_player_id(player_name: str, team_name: Optional[str] = None) -> Optional[str]:
    fixed_name = fix_encoding(player_name)
    for query in [fixed_name, re.sub(r"[^\w\s]", "", fixed_name)]:
        endpoint = f"players/search/{query}"
        logging.info(f"Searching for Player ID for '{query}'.")
        search_results = make_request_with_retry(endpoint).get("results", [])
        if search_results:
            valid_results = [r for r in search_results if r.get("marketValue")]
            best_match = valid_results[0] if valid_results else search_results[0]
            logging.info(f"Found Player ID {best_match['id']} for '{query}'.")
            return best_match["id"]
    alt_name = fixed_name.replace("-", " ")
    if alt_name != fixed_name:
        for query in [alt_name, re.sub(r"[^\w\s]", "", alt_name)]:
            endpoint = f"players/search/{query}"
            logging.info(f"Alternative punctuation search for '{query}'.")
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for alternative query '{query}'.")
                return best_match["id"]
    parts = fixed_name.split()
    if len(parts) >= 3:
        hyphenated = f"{parts[0]}-{parts[1]} " + " ".join(parts[2:])
        for query in [hyphenated, re.sub(r"[^\w\s]", "", hyphenated)]:
            endpoint = f"players/search/{query}"
            logging.info(f"Hyphenated search for '{query}'.")
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for hyphenated query '{query}'.")
                return best_match["id"]
    if len(parts) == 2:
        reversed_name = f"{parts[1]} {parts[0]}"
        logging.info(f"Trying reversed name: '{reversed_name}'.")
        for query in [reversed_name, re.sub(r"[^\w\s]", "", reversed_name)]:
            endpoint = f"players/search/{query}"
            logging.info(f"Searching for Player ID with reversed query '{query}'.")
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for reversed query '{query}'.")
                return best_match["id"]
        logging.warning(f"No results for reversed name '{reversed_name}'.")
    lower_name = fixed_name.lower()
    if lower_name != fixed_name:
        for query in [lower_name, re.sub(r"[^\w\s]", "", lower_name)]:
            endpoint = f"players/search/{query}"
            logging.info(f"Fallback search with lower-case query: '{query}'.")
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for lower-case query '{query}'.")
                return best_match["id"]
    candidates = generate_candidate_queries(player_name)
    for query in candidates:
        endpoint = f"players/search/{query}"
        logging.info(f"Candidate generator search for '{query}'.")
        search_results = make_request_with_retry(endpoint).get("results", [])
        if search_results:
            valid_results = [r for r in search_results if r.get("marketValue")]
            best_match = valid_results[0] if valid_results else search_results[0]
            logging.info(f"Found Player ID {best_match['id']} using candidate variant '{query}'.")
            return best_match["id"]
    if team_name:
        fallback_id = fetch_player_id_by_last_name(fixed_name, team_name)
        if fallback_id:
            return fallback_id
    logging.warning(f"No search results found for '{fixed_name}' or its variants.")
    return None


# =============================================================================
# Transfermarkt URL Helper
# =============================================================================
def slugify(value: str) -> str:
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def construct_transfermarkt_url(player_name: str, player_id: str) -> str:
    tld = CONFIG.get("transfermarkt_tld", "com")
    return f"https://www.transfermarkt.{tld}/{slugify(player_name)}/profil/spieler/{player_id}"


# =============================================================================
# Market Value Filtering & Validation
# =============================================================================
def filter_market_values_by_season(market_values: List[Dict[str, Any]], season_start: datetime, season_end: datetime) -> \
        List[Dict[str, Any]]:
    filtered = []
    for entry in market_values:
        value_date = parse_date(entry.get("date"))
        if value_date and season_start <= value_date <= season_end:
            filtered.append(entry)
    return filtered


def validate_market_value(market_values: List[Dict[str, Any]], team_name: str, season_start: datetime,
                          season_end: datetime) -> Optional[Dict[str, Any]]:
    season_entries = filter_market_values_by_season(market_values, season_start, season_end)
    closest_entry, closest_date_diff = None, float("inf")
    for entry in season_entries:
        value_date = parse_date(entry.get("date"))
        if not value_date:
            continue
        club_name = entry.get("clubName", "")
        match_score = partial_ratio(team_name.lower(), club_name.lower())
        logging.debug(f"Fuzzy match score: {match_score} for '{team_name}' vs '{club_name}' on {value_date}")
        if match_score >= 80:
            date_diff = abs((value_date - season_end).days)
            if date_diff < closest_date_diff:
                closest_date_diff = date_diff
                closest_entry = entry
    if closest_entry:
        return closest_entry
    if market_values:
        try:
            earliest_entry = min(market_values, key=lambda r: parse_date(r.get("date")) or datetime.max)
            logging.info(f"Falling back to earliest market value entry: {earliest_entry}")
            return earliest_entry
        except Exception as e:
            logging.error(f"Error finding earliest market value entry: {e}")
    logging.warning(f"No market value entries available to validate for '{team_name}'.")
    return None


# =============================================================================
# Player Processing & Aggregation
# =============================================================================
def process_player(player_name: str, team_name: str) -> Optional[List[Dict[str, Any]]]:
    fixed_name = fix_encoding(player_name)
    player_id = fetch_player_id(fixed_name, team_name)
    if not player_id:
        logging.warning(f"Could not determine Player ID for '{fixed_name}'. Skipping.")
        return None
    market_values = fetch_player_market_value(player_id)
    if not market_values:
        logging.warning(f"No market value data returned by API for '{fixed_name}'.")
        return None
    return market_values


def process_player_values(file_path: str, season_end_year: int) -> None:
    df = read_input_file(file_path)
    if df is None:
        return
    df["Market Value"] = None
    season_start = datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT)
    season_end = datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT)

    def process_row(index: int, row: pd.Series):
        player_name = row["player"]
        team_name = row["squad"]
        logging.info(f"Processing player: {player_name} (Team: {team_name})")
        fixed_name = fix_encoding(player_name)
        player_id = fetch_player_id(fixed_name, team_name)
        if not player_id:
            logging.warning(f"No Player ID found for '{fixed_name}'. Skipping.")
            return index, None
        market_values = fetch_player_market_value(player_id)
        if not market_values:
            logging.warning(f"No API market value found for '{fixed_name}'.")
            return index, None
        valid_entry = validate_market_value(market_values, team_name, season_start, season_end)
        if valid_entry and "marketValue" in valid_entry:
            date_info = valid_entry.get("raw_date", valid_entry.get("date", "unknown date"))
            logging.info(
                f"Assigned Market Value {valid_entry['marketValue']} for '{fixed_name}' (Team: {team_name}) from date {date_info}.")
            return index, valid_entry["marketValue"]
        logging.warning(f"No suitable market value found for '{fixed_name}' (Team: {team_name}).")
        return index, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, idx, row) for idx, row in df.iterrows()]
        for future in as_completed(futures):
            idx, market_value = future.result()
            if market_value is not None:
                df.at[idx, "Market Value"] = market_value

    write_output_file(df, TEMP_UPDATED_FOLDER, file_path)


def copy_market_value(source_file: Path, target_file: Path) -> None:
    """
    Merge the 'Market Value' column from the source (enhanced updated) file into the target file based on normalized 'player'.
    This ensures that even if player names differ slightly in formatting between datasets, the correct value is applied.
    """
    df_source = pd.read_parquet(source_file)
    df_target = pd.read_parquet(target_file)
    if "player" in df_source.columns and "player" in df_target.columns:
        df_source["player_norm"] = df_source["player"].apply(normalize_name)
        df_target["player_norm"] = df_target["player"].apply(normalize_name)
        # Drop duplicates in case enhanced file has one row per player
        df_source_unique = df_source.drop_duplicates("player_norm")
        df_merged = pd.merge(df_target, df_source_unique[["player_norm", "Market Value"]], on="player_norm", how="left")
        df_merged.drop(columns=["player_norm"], inplace=True)
    else:
        df_merged = df_target
    df_merged.to_parquet(target_file, index=False)
    logging.info(f"Market Value merged into file: {target_file}")


# =============================================================================
# Main Execution Function
# =============================================================================
def main() -> None:
    """
    Process player market value updates for each preprocessed variant.

    1. For the enhanced_feature_engineering variant:
       - Update each file via API calls and save the updated file into TEMP_UPDATED_FOLDER.
       - Copy the updated file from TEMP_UPDATED_FOLDER into UPDATED_VARIANT_FOLDERS["enhanced_feature_engineering"].

    2. For the feature_engineering and no_feature_engineering variants:
       - Copy the cleaned file from the corresponding preprocessed folder to the corresponding UPDATED_VARIANT_FOLDERS.
       - For the no_feature_engineering variant, drop the 'league' and 'season' columns.
       - Merge the Market Value from the corresponding enhanced updated file into the copy.

    Finally, remove the temporary folder.
    """
    # Process enhanced variant: update Market Value via API calls.
    enhanced_folder = PREPROC_VARIANT_FOLDERS["enhanced_feature_engineering"]
    enhanced_output = UPDATED_VARIANT_FOLDERS["enhanced_feature_engineering"]
    logging.info(f"Processing enhanced variant from: {enhanced_folder}")
    for file in os.listdir(enhanced_folder):
        if file.endswith(".parquet"):
            file_path = enhanced_folder / file
            match = re.search(r"(\d{4})-(\d{4})", file)
            if not match:
                logging.warning(f"Skipping {file}: Cannot determine season year.")
                continue
            season_end_year = int(match.group(2))
            logging.info(f"Processing {file} for season ending {season_end_year} (enhanced variant)")
            try:
                process_player_values(str(file_path), season_end_year)
                updated_filename = get_updated_filename_from_cleaned(file)
                temp_file = TEMP_UPDATED_FOLDER / updated_filename
                final_file = enhanced_output / updated_filename
                if temp_file.is_file():
                    shutil.copy(temp_file, final_file)
                    logging.info(f"Enhanced updated file saved to: {final_file}")
                else:
                    logging.error(f"Enhanced updated file not found for {file}.")
            except Exception as e:
                logging.error(f"Error processing enhanced file {file_path}: {e}")

    # Process basic and no_feature_engineering variants: merge Market Value from enhanced files.
    for variant_key in ["feature_engineering", "no_feature_engineering"]:
        variant_folder = PREPROC_VARIANT_FOLDERS[variant_key]
        output_folder = UPDATED_VARIANT_FOLDERS[variant_key]
        logging.info(f"Processing {variant_key} variant from: {variant_folder}")
        for file in os.listdir(variant_folder):
            if file.endswith(".parquet"):
                file_path = variant_folder / file
                updated_filename = get_updated_filename_from_cleaned(file)
                target_file = output_folder / updated_filename
                try:
                    df_variant = pd.read_parquet(file_path)
                    if variant_key == "no_feature_engineering":
                        df_variant = df_variant.drop(columns=["league", "season"], errors="ignore")
                    write_output_file(df_variant, output_folder, file)
                    enhanced_file = UPDATED_VARIANT_FOLDERS["enhanced_feature_engineering"] / updated_filename
                    if not enhanced_file.is_file():
                        logging.warning(
                            f"Enhanced updated file not found for {file}. Skipping Market Value merge for {variant_key}.")
                        continue
                    copy_market_value(enhanced_file, target_file)
                    logging.info(f"{variant_key} updated file saved to: {target_file}")
                except Exception as e:
                    logging.error(f"Error processing {variant_key} file {file_path}: {e}")

    # Clean up temporary folder.
    try:
        for temp_file in TEMP_UPDATED_FOLDER.iterdir():
            temp_file.unlink()
        os.remove(TEMP_UPDATED_FOLDER)
        logging.info("Temporary updated folder removed.")
    except Exception as e:
        logging.error(f"Error cleaning up temporary folder: {e}")


if __name__ == "__main__":
    main()