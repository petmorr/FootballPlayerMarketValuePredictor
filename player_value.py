import functools
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import ftfy
import pandas as pd
import requests
import unicodedata
from rapidfuzz.fuzz import partial_ratio

from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------------
CONFIG = {
    # Transfermarkt top-level domain to use (e.g., "co.uk" or "com")
    "transfermarkt_tld": "co.uk"
}

logging = configure_logger("player_value", "player_value.log")
API_BASE_URL: str = "http://localhost:8000"
DATE_FORMAT: str = "%Y-%m-%d"
MAX_API_RETRIES: int = 3

# ------------------------------------------------------------------------------
# File I/O Helpers
# ------------------------------------------------------------------------------
def get_clean_basename(file_path: str) -> str:
    """
    Returns a clean base name without all extensions.
    For example, "players.parquet" becomes "players".
    """
    p = Path(file_path)
    if len(p.suffixes) > 1:
        # Remove all suffixes (split on period and take the first part)
        return p.name.split('.')[0]
    else:
        return p.stem

def read_input_file(file_path: str) -> pd.DataFrame:
    """
    Reads an input file that should be a Parquet file.
    """
    p = Path(file_path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)

def write_output_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Writes the DataFrame to the ./data/updated folder in parquet format.
    The output files will be named as <basename>.parquet.
    """
    output_dir = Path("./data/updated")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = get_clean_basename(file_path)
    parquet_path = output_dir / f"{base_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    logging.info(f"Updated dataset saved to Parquet: {parquet_path}")

# ------------------------------------------------------------------------------
# Text Encoding Fix
# ------------------------------------------------------------------------------
def fix_encoding(s: str) -> str:
    """
    Uses ftfy to repair mis-encoded text (mojibake).
    For example, converts "Thiago AlcÃ¡ntara" to "Thiago Alcântara".
    """
    try:
        return ftfy.fix_text(s)
    except Exception as e:
        logging.error(f"Encoding fix failed for {s}: {e}")
        return s

# ------------------------------------------------------------------------------
# API Request Functions (with caching)
# ------------------------------------------------------------------------------
session = requests.Session()

def make_request_with_retry(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Makes a GET request to the API endpoint with up to MAX_API_RETRIES.
    """
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

@functools.lru_cache(maxsize=1000)
def fetch_player_id(player_name: str) -> Optional[str]:
    """
    Searches for a player by name via the API and returns the best-matched Player ID.
    It tries both the fixed name and a punctuation‑stripped alternative.
    If multiple results are returned, it selects the first result that has a nonzero market value.
    If the name consists of exactly two parts and no match is found, it also tries the reversed name.
    """
    fixed_name = fix_encoding(player_name)
    alt_name = re.sub(r"[^\w\s]", "", fixed_name)
    for query in [fixed_name, alt_name]:
        endpoint = f"players/search/{query}"
        logging.info(f"Searching for Player ID for '{query}'.")
        search_results = make_request_with_retry(endpoint).get("results", [])
        if search_results:
            # Prefer entries with a market value.
            valid_results = [r for r in search_results if r.get("marketValue")]
            best_match = valid_results[0] if valid_results else search_results[0]
            logging.info(f"Found Player ID {best_match['id']} for '{query}'.")
            return best_match["id"]
    # Try reversing the name if it has exactly two parts.
    name_parts = fixed_name.split()
    if len(name_parts) == 2:
        reversed_name = f"{name_parts[1]} {name_parts[0]}"
        logging.info(f"No results found for '{fixed_name}'. Trying reversed name: '{reversed_name}'.")
        for query in [reversed_name, re.sub(r"[^\w\s]", "", reversed_name)]:
            endpoint = f"players/search/{query}"
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for reversed name '{query}'.")
                return best_match["id"]
        logging.warning(f"No search results found for reversed name '{reversed_name}'.")
    logging.warning(f"No search results found for '{fixed_name}' or its alternatives.")
    return None

@functools.lru_cache(maxsize=1000)
def fetch_player_profile(player_id: str) -> Optional[str]:
    """
    Retrieves the player's profile URL via the API.
    """
    endpoint = f"players/{player_id}/profile"
    logging.info(f"Fetching profile URL for player ID {player_id}.")
    data = make_request_with_retry(endpoint)
    return data.get("url", None)

def fetch_player_market_value(player_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves the player's full market value history from the API.
    """
    endpoint = f"players/{player_id}/market_value"
    logging.info(f"Fetching market value for player ID {player_id}.")
    return make_request_with_retry(endpoint).get("marketValueHistory", [])

# ------------------------------------------------------------------------------
# Helper to construct Transfermarkt fallback URL
# ------------------------------------------------------------------------------
def slugify(value: str) -> str:
    """
    Converts a string to a URL-friendly slug.
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def construct_transfermarkt_url(player_name: str, player_id: str) -> str:
    """
    Constructs a Transfermarkt profile URL for the given player using the configured TLD.
    Example: https://www.transfermarkt.co.uk/milos-veljkovic/profil/spieler/202228
    """
    tld = CONFIG.get("transfermarkt_tld", "com")
    name_slug = slugify(player_name)
    return f"https://www.transfermarkt.{tld}/{name_slug}/profil/spieler/{player_id}"

# ------------------------------------------------------------------------------
# Season Filtering Helper
# ------------------------------------------------------------------------------
def filter_market_values_by_season(market_values: List[Dict[str, Any]],
                                   season_start: datetime,
                                   season_end: datetime) -> List[Dict[str, Any]]:
    """
    Filters the market value entries to only those whose dates fall within the given season.
    """
    filtered = []
    for entry in market_values:
        try:
            value_date = entry["date"] if isinstance(entry["date"], datetime) else datetime.strptime(entry["date"],
                                                                                                     DATE_FORMAT)
            if season_start <= value_date <= season_end:
                filtered.append(entry)
        except Exception as e:
            logging.error(f"Error filtering market value entry: {e}")
    return filtered

# ------------------------------------------------------------------------------
# Player Processing and Market Value Validation
# ------------------------------------------------------------------------------
def process_player(player_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Processes a player by trying to get the market value history via the API.
    (All webscraping functionality has been removed.)
    If no market value history is found, this function returns None so that manual input can be requested later.
    """
    fixed_name = fix_encoding(player_name)
    player_id = fetch_player_id(fixed_name)
    if not player_id:
        logging.warning(f"Could not determine Player ID for {fixed_name}. Skipping.")
        return None

    market_values = fetch_player_market_value(player_id)
    if not market_values:
        logging.warning(f"No market value data returned by the API for {fixed_name}.")
        return None
    return market_values

def validate_market_value(
        market_values: List[Dict[str, Any]],
        team_name: str,
        season_start: datetime,
        season_end: datetime
) -> Optional[Dict[str, Any]]:
    """
    Selects the market value entry for the given team within the season.
    First, it filters the full history to entries within the season.
    If none are found (e.g. because the first market value is dated after the season),
    it falls back to returning the earliest available market value entry.
    Fuzzy matching is applied to prefer entries matching the team name.
    """
    season_entries = filter_market_values_by_season(market_values, season_start, season_end)
    if season_entries:
        closest_entry = None
        closest_date_diff = float("inf")
        for entry in season_entries:
            try:
                value_date = entry["date"] if isinstance(entry["date"], datetime) else datetime.strptime(entry["date"],
                                                                                                         DATE_FORMAT)
                club_name = entry.get("clubName", "") or ""
                match_score = partial_ratio(team_name.lower(), club_name.lower())
                logging.debug(f"Fuzzy match score: {match_score} for '{team_name}' vs '{club_name}' on {value_date}")
                if match_score >= 80:
                    date_diff = abs((value_date - season_end).days)
                    if date_diff < closest_date_diff:
                        closest_date_diff = date_diff
                        closest_entry = entry
            except Exception as e:
                logging.error(f"Error processing market value entry: {e}")
        if closest_entry:
            return closest_entry
        else:
            logging.warning(
                f"No valid market value match for '{team_name}' in season range; falling back to earliest entry.")
    # Fallback: return the earliest market value in the full history.
    if market_values:
        try:
            earliest_entry = min(
                market_values,
                key=lambda r: r["date"] if isinstance(r["date"], datetime)
                else datetime.strptime(r["date"], DATE_FORMAT)
            )
            logging.info(f"Falling back to earliest market value entry: {earliest_entry}")
            return earliest_entry
        except Exception as e:
            logging.error(f"Error finding earliest market value entry: {e}")
    logging.warning(f"No market value entries available to validate for '{team_name}'.")
    return None

# ------------------------------------------------------------------------------
# Data Processing Functions
# ------------------------------------------------------------------------------
def process_player_values(file_path: str, season_end_year: int) -> None:
    """
    Processes a file (CSV, gzipped CSV, or Parquet) to update each player's market value.
    Only market value entries from the season of interest (e.g., for 19/20, from July 2019 to June 2020)
    are considered. Each row is processed concurrently.
    If no valid market value is found for a player, a message is logged so that manual input can later be performed.
    """
    df = read_input_file(file_path)
    df["Market Value"] = None

    def process_row(index: int, row: pd.Series):
        player_name = row["player"]
        team_name = row["squad"]
        logging.info(f"Processing player: {player_name} (Team: {team_name})")
        fixed_name = fix_encoding(player_name)
        player_id = fetch_player_id(fixed_name)
        if not player_id:
            logging.warning(f"No Player ID found for {fixed_name}. Skipping.")
            return index, None
        market_values = fetch_player_market_value(player_id)
        if not market_values:
            logging.warning(f"No API market value found for {fixed_name}. Please input manually later.")
            return index, None
        season_start = datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT)
        season_end = datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT)
        valid_entry = validate_market_value(market_values, team_name, season_start, season_end)
        if valid_entry and "marketValue" in valid_entry:
            date_info = valid_entry.get("raw_date", valid_entry.get("date", "unknown date"))
            logging.info(
                f"Assigned Market Value {valid_entry['marketValue']} for {fixed_name} (Team: {team_name}) from date {date_info}.")
            return index, valid_entry["marketValue"]
        else:
            logging.warning(
                f"No suitable market value found for {fixed_name} (Team: {team_name}). Please input manually later.")
            return index, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, idx, row) for idx, row in df.iterrows()]
        for future in as_completed(futures):
            idx, market_value = future.result()
            if market_value is not None:
                df.at[idx, "Market Value"] = market_value

    write_output_file(df, file_path)

def process_all_files(input_folder: str, season_end_years: Dict[str, int]) -> None:
    """
    Processes all files in the input folder using the provided season mapping.
    """
    input_dir = Path(input_folder)
    for file_name, season_end_year in season_end_years.items():
        file_path = input_dir / file_name
        if file_path.is_file():
            logging.info(f"Processing file: {file_path} for season ending {season_end_year}.")
            try:
                process_player_values(str(file_path), season_end_year)
                logging.info(f"Successfully processed {file_path}.")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

def extract_season_year(file_name: str) -> Optional[int]:
    """
    Extracts the season end year from the file name using a regex.
    Expected format: contains a season in the format "YYYY-YYYY".
    """
    match = re.search(r"(\d{4})-(\d{4})", file_name)
    if match:
        return int(match.group(2))
    return None

# ------------------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main function to process player market value updates for all cleaned datasets.
    It iterates over files, extracts season info, and processes each file.
    If the API does not return a valid market value for a player,
    the script logs a warning so that manual input can later be provided via the GUI.
    """
    input_folder = Path("./data/cleaned")
    if not input_folder.is_dir():
        logging.error(f"Input folder '{input_folder}' does not exist.")
        return
    logging.info("Starting processing of player values...")
    for file in os.listdir(input_folder):
        season_end_year = extract_season_year(file)
        if season_end_year:
            logging.info(f"Processing {file} for season ending {season_end_year}")
            process_player_values(str(input_folder / file), season_end_year)
        else:
            logging.warning(f"Skipping {file}: Could not determine season year.")
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()