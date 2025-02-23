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
from pandas import DataFrame
from rapidfuzz.fuzz import partial_ratio

from logging_config import configure_logger

# =============================================================================
# Global Configuration & Logging
# =============================================================================
CONFIG = {
    "transfermarkt_tld": "co.uk"  # Transfermarkt top-level domain to use (e.g., "co.uk" or "com")
}
API_BASE_URL: str = "http://localhost:8000"
DATE_FORMAT: str = "%Y-%m-%d"
MAX_API_RETRIES: int = 3

logging = configure_logger("player_value", "player_value.log")


# =============================================================================
# File I/O Helpers
# =============================================================================
def get_clean_basename(file_path: str) -> str:
    """Return the base name without extensions (e.g. players.parquet -> players)."""
    p = Path(file_path)
    return p.name.split('.')[0] if len(p.suffixes) > 1 else p.stem


def read_input_file(file_path: str) -> Optional[DataFrame]:
    """Read a Parquet input file."""
    p = Path(file_path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    logging.error(f"Unsupported file type: {file_path}")
    return None


def write_output_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Writes the DataFrame to the ../data/updated folder in Parquet format.
    Output file is named as <basename>.parquet.
    """
    output_dir = Path("../data/updated")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = get_clean_basename(file_path)
    parquet_path = output_dir / f"{base_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    logging.info(f"Updated dataset saved to Parquet: {parquet_path}")


# =============================================================================
# Text & Date Utilities
# =============================================================================
def fix_encoding(s: str) -> str:
    """
    Repair mis-encoded text using ftfy.
    Example: "Thiago AlcÃ¡ntara" -> "Thiago Alcântara".
    """
    try:
        return ftfy.fix_text(s)
    except Exception as e:
        logging.error(f"Encoding fix failed for '{s}': {e}")
        return s


def parse_date(date_str: Any) -> Optional[datetime]:
    """Parse a date string or return the datetime if already parsed."""
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
    """Remove diacritic marks from a string while preserving the base characters."""
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    """
    Fully normalize a name by:
      1. Fixing encoding,
      2. Removing diacritics,
      3. Converting to lower-case, and
      4. Removing hyphens, apostrophes, and all whitespace.
    For example, both "Jan Luca Rumpf" and "Jan-Luca Rumpf" become "janlucarumpf".
    """
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
    """
    Normalize a name while preserving spaces.
    Removes diacritics and punctuation (but keeps spaces) and lowercases.
    """
    try:
        name = fix_encoding(name).strip().lower()
        name = remove_diacritics(name)
    except Exception as e:
        logging.error(f"Error during normalization (keep spaces) for '{name}': {e}")
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def get_last_name(full_name: str) -> str:
    """Extract and fully normalize the last name from a full name."""
    parts = full_name.split()
    return normalize_name(parts[-1]) if parts else ""


# =============================================================================
# Candidate Query Generator
# =============================================================================
def generate_candidate_queries(player_name: str) -> set:
    """
    Generate a comprehensive set of candidate query strings for a given player name.

    This function produces variants including:
      - The fixed name (as returned by fix_encoding)
      - Lower-case variants
      - Variants with punctuation removed
      - Variants with diacritics removed (using remove_diacritics)
      - Variants where hyphens are replaced with spaces
      - Reversed order (for two-part names)
      - If the name has three or more parts, a variant that hyphenates the first two parts
      - An "initial plus last name" variant (if possible)
      - Last name only variants
    """
    fixed = fix_encoding(player_name).strip()
    variants = set()

    # Original variants
    variants.add(fixed)
    variants.add(fixed.lower())
    variants.add(re.sub(r"[^\w\s]", "", fixed))
    variants.add(re.sub(r"[^\w\s]", "", fixed.lower()))

    # Diacritics removed variants
    no_diacritics = remove_diacritics(fixed)
    variants.add(no_diacritics)
    variants.add(no_diacritics.lower())
    variants.add(re.sub(r"[^\w\s]", "", no_diacritics))
    variants.add(re.sub(r"[^\w\s]", "", no_diacritics.lower()))

    # Hyphen replaced with space
    hyphen_space = fixed.replace("-", " ")
    variants.add(hyphen_space)
    variants.add(hyphen_space.lower())
    variants.add(re.sub(r"[^\w\s]", "", hyphen_space))
    variants.add(re.sub(r"[^\w\s]", "", hyphen_space.lower()))

    # Fully normalized variant (no spaces)
    normalized = normalize_name(player_name)
    variants.add(normalized)

    # Normalized variant keeping spaces
    norm_spaces = normalize_name_keep_spaces(player_name)
    variants.add(norm_spaces)

    # For names with two parts, try reversed order
    parts = fixed.split()
    if len(parts) == 2:
        reversed_order = f"{parts[1]} {parts[0]}"
        variants.add(reversed_order)
        variants.add(reversed_order.lower())
        variants.add(re.sub(r"[^\w\s]", "", reversed_order))
        variants.add(re.sub(r"[^\w\s]", "", reversed_order.lower()))
    # For names with three or more parts, try hyphenating first two parts
    if len(parts) >= 3:
        hyphenated = f"{parts[0]}-{parts[1]} " + " ".join(parts[2:])
        variants.add(hyphenated)
        variants.add(hyphenated.lower())
        variants.add(re.sub(r"[^\w\s]", "", hyphenated))
        variants.add(re.sub(r"[^\w\s]", "", hyphenated.lower()))

    # Try an "initial plus last name" variant for multi-part names
    if len(parts) >= 2:
        initial_last = f"{parts[0][0]} {parts[-1]}"
        variants.add(initial_last)
        variants.add(initial_last.lower())
        variants.add(re.sub(r"[^\w\s]", "", initial_last))
        variants.add(re.sub(r"[^\w\s]", "", initial_last.lower()))

    # Last name only
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
    """Make a GET request to the API endpoint with up to MAX_API_RETRIES."""
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
    """Retrieve the player's full market value history from the API."""
    endpoint = f"players/{player_id}/market_value"
    logging.info(f"Fetching market value for player ID {player_id}.")
    return make_request_with_retry(endpoint).get("marketValueHistory", [])


@functools.lru_cache(maxsize=1000)
def fetch_player_profile(player_id: str) -> Optional[str]:
    """Retrieve the player's profile URL via the API."""
    endpoint = f"players/{player_id}/profile"
    logging.info(f"Fetching profile URL for player ID {player_id}.")
    data = make_request_with_retry(endpoint)
    return data.get("url", None)


# =============================================================================
# Name-Based Search Functions
# =============================================================================
def fetch_player_id_by_last_name(player_name: str, team_name: str) -> Optional[str]:
    """
    Fallback search using only the player's last name.
    Returns the first candidate whose normalized last name matches and whose club name
    fuzzy-matches the provided team name.
    """
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
            f"Fallback search found candidate {best_candidate['id']} for last name '{input_last}' with team match score {best_score}.")
        return best_candidate["id"]
    logging.warning(f"Fallback search by last name for '{input_last}' found no suitable candidate.")
    return None


@functools.lru_cache(maxsize=1000)
def fetch_player_id(player_name: str, team_name: Optional[str] = None) -> Optional[str]:
    """
    Searches for a player by name via the API and returns the best-matched Player ID.

    This function uses a robust, iterative fallback strategy:
      1. Try the fixed (encoding-corrected) name and its punctuation-stripped version.
      2. Try alternative punctuation handling (e.g. replacing hyphens with spaces).
      3. For names with three or more parts, try a hyphenated variant of the first two parts.
      4. For two-part names, try the reversed order.
      5. Try lower-case variants (with and without punctuation).
      6. Try additional variants generated by the candidate generator.
      7. If a team name is provided, fall back to a last-name search.
    """
    fixed_name = fix_encoding(player_name)

    # 1. Primary attempts: fixed name and punctuation-stripped version.
    for query in [fixed_name, re.sub(r"[^\w\s]", "", fixed_name)]:
        endpoint = f"players/search/{query}"
        logging.info(f"Searching for Player ID for '{query}'.")
        search_results = make_request_with_retry(endpoint).get("results", [])
        if search_results:
            valid_results = [r for r in search_results if r.get("marketValue")]
            if valid_results:
                best_match = valid_results[0]
            else:
                best_match = search_results[0]
            logging.info(f"Found Player ID {best_match['id']} for '{query}'.")
            return best_match["id"]

    # 2. Alternative punctuation handling: replace hyphens with spaces.
    alt_name = fixed_name.replace("-", " ")
    if alt_name != fixed_name:
        for query in [alt_name, re.sub(r"[^\w\s]", "", alt_name)]:
            endpoint = f"players/search/{query}"
            logging.info(f"Alternative punctuation search for '{query}'.")
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for alternative punctuation query '{query}'.")
                return best_match["id"]

    # 3. For names with three or more parts, try a hyphenated variant of the first two parts.
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

    # 4. For two-part names, try reversed order.
    if len(parts) == 2:
        reversed_name = f"{parts[1]} {parts[0]}"
        logging.info(f"No results for '{fixed_name}'. Trying reversed name: '{reversed_name}'.")
        for query in [reversed_name, re.sub(r"[^\w\s]", "", reversed_name)]:
            endpoint = f"players/search/{query}"
            logging.info(f"Searching for Player ID with reversed query '{query}'.")
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                valid_results = [r for r in search_results if r.get("marketValue")]
                best_match = valid_results[0] if valid_results else search_results[0]
                logging.info(f"Found Player ID {best_match['id']} for reversed query '{query}'.")
                return best_match["id"]
        logging.warning(f"No search results found for reversed name '{reversed_name}'.")

    # 5. Try lower-case variants explicitly.
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

    # 6. Use the candidate generator to try many robust variants.
    candidates = generate_candidate_queries(player_name)
    for query in candidates:
        endpoint = f"players/search/{query}"
        logging.info(f"Candidate generator search for '{query}'.")
        search_results = make_request_with_retry(endpoint).get("results", [])
        if search_results:
            valid_results = [r for r in search_results if r.get("marketValue")]
            best_match = valid_results[0] if valid_results else search_results[0]
            # Optional: you could combine fuzzy matching scores here.
            logging.info(f"Found Player ID {best_match['id']} using candidate variant '{query}'.")
            return best_match["id"]

    # 7. If a team name is provided, fallback to a last-name search.
    if team_name:
        fallback_id = fetch_player_id_by_last_name(fixed_name, team_name)
        if fallback_id:
            return fallback_id

    logging.warning(f"No search results found for '{fixed_name}' or any of its alternatives.")
    return None


# =============================================================================
# Transfermarkt URL Helper
# =============================================================================
def slugify(value: str) -> str:
    """Convert a string to a URL-friendly slug."""
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def construct_transfermarkt_url(player_name: str, player_id: str) -> str:
    """Construct Transfermarkt profile URL using the configured TLD."""
    tld = CONFIG.get("transfermarkt_tld", "com")
    return f"https://www.transfermarkt.{tld}/{slugify(player_name)}/profil/spieler/{player_id}"


# =============================================================================
# Market Value Filtering & Validation
# =============================================================================
def filter_market_values_by_season(market_values: List[Dict[str, Any]],
                                   season_start: datetime,
                                   season_end: datetime) -> List[Dict[str, Any]]:
    """
    Filter market value entries to those with dates within the given season.
    """
    filtered = []
    for entry in market_values:
        value_date = parse_date(entry.get("date"))
        if value_date and season_start <= value_date <= season_end:
            filtered.append(entry)
    return filtered


def validate_market_value(market_values: List[Dict[str, Any]],
                          team_name: str,
                          season_start: datetime,
                          season_end: datetime) -> Optional[Dict[str, Any]]:
    """
    Select the market value entry for the team within the season.
    Uses fuzzy matching on the team name. Falls back to the earliest entry if needed.
    """
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
# Player Processing & Data Aggregation
# =============================================================================
def process_player(player_name: str, team_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Process a player to obtain their market value history.
    Returns the market value history or None if unavailable.
    """
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
    """
    Process a file (expected to be Parquet) to update each player's market value.
    Only market value entries from the season of interest are considered.
    Logs warnings if no valid market value is found.
    """
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
            logging.warning(f"No API market value found for '{fixed_name}'. Manual input may be required.")
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

    write_output_file(df, file_path)


def process_all_files(input_folder: str, season_end_years: Dict[str, int]) -> None:
    """
    Process all files in the input folder using the provided season mapping.
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
    Extract the season end year from the file name using a regex.
    Expected format: a season in the form 'YYYY-YYYY'.
    """
    match = re.search(r"(\d{4})-(\d{4})", file_name)
    return int(match.group(2)) if match else None


# =============================================================================
# Main Execution Function
# =============================================================================
def main() -> None:
    """
    Main function to process player market value updates for all cleaned datasets.
    Iterates over files in the input folder, extracts season info, and processes each file.
    """
    input_folder = Path("../data/cleaned")
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