import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from retrying import retry
from rapidfuzz import fuzz
import unicodedata

API_BASE_URL = "http://localhost:8000"  # Runs locally, change if using a server
CACHE = {}  # Dictionary-based cache for player searches and market values
DEBUG_LOG_FILE = "debug_log.txt"

def log_debug(message):
    """Log debug messages to a file and print to console."""
    print(message)
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()} - {message}\n")

def retry_if_connection_error(exception):
    """Retry logic for connection-related errors."""
    return isinstance(exception, requests.RequestException)

@retry(
    retry_on_exception=retry_if_connection_error,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=5,
)
def make_request(url, params=None):
    """Make a GET request with retry logic."""
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def normalize_string(input_string):
    """Normalize strings for better matching."""
    if not isinstance(input_string, str):
        return ""
    return unicodedata.normalize("NFKD", input_string).encode("ascii", "ignore").decode("utf-8").lower().replace(" ", "").replace("-", "")

def reverse_name(player_name):
    """Reverse the name for cases where the surname comes first (e.g., 'Kwon Chang-hoon')."""
    parts = player_name.split(" ")
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return player_name

def search_player(player_name):
    """Search for a player using the Transfermarkt API with caching."""
    if player_name in CACHE:
        return CACHE[player_name]

    try:
        # First attempt with the provided name
        response = make_request(f"{API_BASE_URL}/players/search/{player_name}")
        results = response.get("results", [])
        if results:
            CACHE[player_name] = results
            log_debug(f"API Response for player search '{player_name}': {results}")
            return results

        # Attempt with the reversed name
        reversed_name = reverse_name(player_name)
        response = make_request(f"{API_BASE_URL}/players/search/{reversed_name}")
        results = response.get("results", [])
        CACHE[reversed_name] = results
        log_debug(f"API Response for player search '{reversed_name}': {results}")
        return results
    except requests.RequestException as e:
        log_debug(f"Error searching for player '{player_name}': {e}")
        return []

def validate_player_with_history(player_id, team_name, target_date):
    """Validate player using historical market value and team data."""
    try:
        response = make_request(f"{API_BASE_URL}/players/{player_id}/market_value")
        market_values = response.get("marketValueHistory", [])
        if not market_values:
            log_debug(f"No historical data for player ID '{player_id}'.")
            return None

        closest_value = None
        for value_entry in market_values:
            if "value" not in value_entry or "date" not in value_entry:
                log_debug(f"Incomplete market value entry: {value_entry}")
                continue

            value_date = datetime.strptime(value_entry["date"], "%b %d, %Y")
            club_name = normalize_string(value_entry.get("clubName", ""))
            dataset_team = normalize_string(team_name)
            similarity = fuzz.partial_ratio(club_name, dataset_team)

            log_debug(f"Validating historical data: API team '{club_name}' vs Dataset team '{dataset_team}' - Similarity: {similarity}")

            if similarity > 80 and abs(value_date - target_date) <= timedelta(days=60):
                if closest_value is None or abs(value_date - target_date) < abs(closest_value["date"] - target_date):
                    closest_value = {"date": value_date, "value": value_entry["value"]}

        if closest_value:
            log_debug(f"Validated historical match: {closest_value['value']}")
            return closest_value["value"]

        log_debug(f"No valid historical match for player ID '{player_id}'.")
        return None
    except requests.RequestException as e:
        log_debug(f"Error fetching market value for player ID '{player_id}': {e}")
        return None

def add_market_values_to_dataset(file_path, season_end_year):
    """Add market values to the dataset."""
    df = pd.read_csv(file_path)
    df["Market Value"] = "N/A"
    processed_players = {}

    target_date = datetime.strptime(f"{season_end_year}-05-31", "%Y-%m-%d")

    for index, row in df.iterrows():
        player_name = row["Player"]
        team_name = row["Squad"]

        if player_name in processed_players:
            market_value = processed_players[player_name]
            df.at[index, "Market Value"] = market_value
            log_debug(f"Reused {player_name} with market value: {market_value}")
            continue

        log_debug(f"Searching for player: {player_name} (Team: {team_name})")
        search_results = search_player(player_name)

        if not search_results:
            log_debug(f"No results found for player: {player_name}")
            continue

        best_match = search_results[0]
        if len(search_results) > 1:
            best_match = search_results[0]

        player_id = best_match.get("id")
        if not player_id:
            log_debug(f"No valid player ID found for player: {player_name}")
            continue

        market_value = validate_player_with_history(player_id, team_name, target_date)
        if not market_value:
            log_debug(f"Fallback failed for player ID '{player_id}'.")

        df.at[index, "Market Value"] = market_value if market_value else "N/A"
        processed_players[player_name] = market_value
        log_debug(f"Updated {player_name} (Team: {team_name}) with market value: {market_value}")

    updated_folder = Path("./data/updated")
    updated_folder.mkdir(parents=True, exist_ok=True)
    output_path = updated_folder / f"updated_{Path(file_path).name}"
    df.to_csv(output_path, index=False)
    log_debug(f"Updated dataset saved to {output_path}")

def process_all_files(input_folder, season_end_years):
    """Process all player data files in the input folder and add market values."""
    for file_name, season_end_year in season_end_years.items():
        file_path = Path(input_folder) / file_name
        if file_path.is_file():
            log_debug(f"Processing file: {file_name}")
            add_market_values_to_dataset(file_path, season_end_year)
        else:
            log_debug(f"Skipping non-file path: {file_path}")

if __name__ == "__main__":
    INPUT_FOLDER = "./data/cleaned"
    SEASON_END_YEARS = {
        # Bundesliga
        "cleaned_Bundesliga_2019-2020_player_data.csv": 2020,
        "cleaned_Bundesliga_2020-2021_player_data.csv": 2021,
        "cleaned_Bundesliga_2021-2022_player_data.csv": 2022,
        "cleaned_Bundesliga_2022-2023_player_data.csv": 2023,
        "cleaned_Bundesliga_2023-2024_player_data.csv": 2024,
        # La Liga
        "cleaned_La-Liga_2019-2020_player_data.csv": 2020,
        "cleaned_La-Liga_2020-2021_player_data.csv": 2021,
        "cleaned_La-Liga_2021-2022_player_data.csv": 2022,
        "cleaned_La-Liga_2022-2023_player_data.csv": 2023,
        "cleaned_La-Liga_2023-2024_player_data.csv": 2024,
        # Ligue 1
        "cleaned_Ligue-1_2019-2020_player_data.csv": 2020,
        "cleaned_Ligue-1_2020-2021_player_data.csv": 2021,
        "cleaned_Ligue-1_2021-2022_player_data.csv": 2022,
        "cleaned_Ligue-1_2022-2023_player_data.csv": 2023,
        "cleaned_Ligue-1_2023-2024_player_data.csv": 2024,
        # Premier League
        "cleaned_Premier-League_2019-2020_player_data.csv": 2020,
        "cleaned_Premier-League_2020-2021_player_data.csv": 2021,
        "cleaned_Premier-League_2021-2022_player_data.csv": 2022,
        "cleaned_Premier-League_2022-2023_player_data.csv": 2023,
        "cleaned_Premier-League_2023-2024_player_data.csv": 2024,
        # Serie A
        "cleaned_Serie-A_2019-2020_player_data.csv": 2020,
        "cleaned_Serie-A_2020-2021_player_data.csv": 2021,
        "cleaned_Serie-A_2021-2022_player_data.csv": 2022,
        "cleaned_Serie-A_2022-2023_player_data.csv": 2023,
        "cleaned_Serie-A_2023-2024_player_data.csv": 2024,
    }
    process_all_files(INPUT_FOLDER, SEASON_END_YEARS)