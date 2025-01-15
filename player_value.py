import os
import time
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import requests
from rapidfuzz.fuzz import partial_ratio

from logging_config import configure_logger

# Configure logger for player_value
logging = configure_logger("player_value", "player_value.log")

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Constants
DATE_FORMAT = "%Y-%m-%d"


def make_request_with_infinite_retry(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """
    Make a GET request to the API with infinite retry logic.

    Args:
        endpoint (str): The API endpoint to call.
        params (Optional[Dict]): Query parameters for the request.

    Returns:
        Dict: JSON response from the API.
    """
    url = f"{API_BASE_URL}/{endpoint}"
    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            logging.debug(f"Successful API call on attempt {attempt}: {url}")
            return response.json()
        except requests.RequestException as e:
            logging.warning(f"Retrying API call to {url} (attempt {attempt}): {e}")
            time.sleep(min(2 ** attempt, 60))


def fetch_player_market_value_with_retry(player_id: str) -> Optional[List[Dict]]:
    """
    Fetch market value history for a player with infinite retry.

    Args:
        player_id (str): The player's unique ID.

    Returns:
        List[Dict]: Market value history for the player.
    """
    endpoint = f"players/{player_id}/market_value"
    logging.info(f"Fetching market value for player ID {player_id} with retry logic.")
    return make_request_with_infinite_retry(endpoint).get("marketValueHistory", [])

def validate_market_value(
        market_values: List[Dict], team_name: str, season_start: datetime, season_end: datetime
) -> Optional[Dict]:
    """
    Validate market value based on team and season using fuzzy matching.

    Args:
        market_values (List[Dict]): Market value history for the player.
        team_name (str): The name of the team to validate against.
        season_start (datetime): Start date of the season.
        season_end (datetime): End date of the season.

    Returns:
        Optional[Dict]: The closest market value entry to the end of the season if a valid match is found, else None.
    """
    closest_entry = None
    closest_date_diff = float("inf")

    for entry in market_values:
        try:
            value_date = datetime.strptime(entry["date"], DATE_FORMAT)
            club_name = entry.get("clubName", "")
            match_score = partial_ratio(team_name.lower(), club_name.lower())

            if season_start <= value_date <= season_end and match_score >= 80:
                date_diff = abs((value_date - season_end).days)
                if date_diff < closest_date_diff:
                    closest_date_diff = date_diff
                    closest_entry = entry
                logging.debug(
                    f"Fuzzy match score: {match_score} for '{team_name}' vs '{club_name}' on {value_date}"
                )
        except Exception as e:
            logging.error(f"Error processing market value entry: {e}")

    if not closest_entry:
        logging.warning(f"No valid market value match for '{team_name}' in the season.")
    return closest_entry

def process_player_values(file_path: str, season_end_year: int) -> None:
    """
    Process player values and update the dataset.

    Args:
        file_path (str): Path to the player dataset file.
        season_end_year (int): The season's end year.
    """
    season_start = datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT)
    season_end = datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT)

    df = pd.read_csv(file_path)
    df["Market Value"] = None

    for index, row in df.iterrows():
        player_name = row["Player"]
        team_name = row["Squad"]

        logging.info(f"Processing player: {player_name} (Team: {team_name})")

        search_results = make_request_with_infinite_retry(f"players/search/{player_name}").get("results", [])
        if not search_results:
            logging.warning(f"No search results found for {player_name}.")
            continue

        valid_player = None
        for result in search_results:
            if partial_ratio(player_name.lower(), result["name"].lower()) >= 75:
                market_values = fetch_player_market_value_with_retry(result["id"])
                if market_values:
                    closest_entry = validate_market_value(market_values, team_name, season_start, season_end)
                    if closest_entry:
                        valid_player = result
                        df.at[index, "Market Value"] = closest_entry["marketValue"]
                        logging.info(
                            f"Assigned market value {closest_entry['marketValue']} "
                            f"for {player_name} (Team: {team_name}) from date {closest_entry['date']}."
                        )
                        break

        if not valid_player:
            logging.warning(f"No suitable match found for {player_name} (Team: {team_name}).")

    output_dir = "./data/updated"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"updated_{os.path.basename(file_path)}")
    df.to_csv(output_path, index=False)
    logging.info(f"Updated dataset saved to {output_path}")

def process_all_files(input_folder: str, season_end_years: Dict[str, int]) -> None:
    """
    Process all files in the input folder and add market values to player datasets.

    Args:
        input_folder (str): The folder containing input player data files.
        season_end_years (Dict[str, int]): A dictionary mapping file names to season end years.

    Returns:
        None
    """
    for file_name, season_end_year in season_end_years.items():
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            logging.info(f"Processing file: {file_path} for season ending {season_end_year}.")
            try:
                process_player_values(file_path, season_end_year)
                logging.info(f"Successfully processed {file_path}.")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
        else:
            logging.warning(f"File {file_path} does not exist. Skipping.")

if __name__ == "__main__":
    INPUT_FOLDER = "./data/cleaned"
    SEASON_END_YEARS = {
        "cleaned_Bundesliga_2019-2020.csv": 2020,
        "cleaned_Bundesliga_2020-2021.csv": 2021,
        "cleaned_Bundesliga_2021-2022.csv": 2022,
        "cleaned_Bundesliga_2022-2023.csv": 2023,
        "cleaned_Bundesliga_2023-2024.csv": 2024,
        "cleaned_La-Liga_2019-2020.csv": 2020,
        "cleaned_La-Liga_2020-2021.csv": 2021,
        "cleaned_La-Liga_2021-2022.csv": 2022,
        "cleaned_La-Liga_2022-2023.csv": 2023,
        "cleaned_La-Liga_2023-2024.csv": 2024,
        "cleaned_Ligue-1_2019-2020.csv": 2020,
        "cleaned_Ligue-1_2020-2021.csv": 2021,
        "cleaned_Ligue-1_2021-2022.csv": 2022,
        "cleaned_Ligue-1_2022-2023.csv": 2023,
        "cleaned_Ligue-1_2023-2024.csv": 2024,
        "cleaned_Premier-League_2019-2020.csv": 2020,
        "cleaned_Premier-League_2020-2021.csv": 2021,
        "cleaned_Premier-League_2021-2022.csv": 2022,
        "cleaned_Premier-League_2022-2023.csv": 2023,
        "cleaned_Premier-League_2023-2024.csv": 2024,
        "cleaned_Serie-A_2019-2020.csv": 2020,
        "cleaned_Serie-A_2020-2021.csv": 2021,
        "cleaned_Serie-A_2021-2022.csv": 2022,
        "cleaned_Serie-A_2022-2023.csv": 2023,
        "cleaned_Serie-A_2023-2024.csv": 2024,
    }

    if not os.path.isdir(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist. Please verify the path.")
    else:
        logging.info("Starting processing of player values...")
        process_all_files(INPUT_FOLDER, SEASON_END_YEARS)
        logging.info("Processing complete.")