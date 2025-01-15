import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import requests
from retrying import retry

# Configure logging
LOG_FILE = "player_value.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Constants
DATE_FORMAT = "%Y-%m-%d"


# Retry logic for API calls
@retry(
    retry_on_exception=lambda e: isinstance(e, requests.RequestException),
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=5,
)
def make_request(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """
    Make a GET request to the API with retry logic.

    Args:
        endpoint (str): The API endpoint to call.
        params (Optional[Dict]): Query parameters for the request.

    Returns:
        Dict: JSON response from the API.
    """
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        logging.debug(f"Successful API call: {url}")
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error making API call to {url}: {e}")
        raise


def search_player(player_name: str) -> List[Dict]:
    """
    Search for a player using the Transfermarkt API.

    Args:
        player_name (str): The player's name.

    Returns:
        List[Dict]: A list of player search results.
    """
    try:
        data = make_request(f"players/search/{player_name}")
        logging.info(f"API response for player search '{player_name}': {data}")
        return data.get("results", [])
    except Exception as e:
        logging.error(f"Error searching for player '{player_name}': {e}")
        return []


def fetch_player_market_value(player_id: str) -> List[Dict]:
    """
    Fetch market value history for a given player.

    Args:
        player_id (str): The player's unique ID.

    Returns:
        List[Dict]: Market value history for the player.
    """
    try:
        data = make_request(f"players/{player_id}/market_value")
        logging.info(f"Market value history fetched for player ID {player_id}.")
        return data.get("marketValueHistory", [])
    except Exception as e:
        logging.error(f"Error fetching market value for player ID {player_id}: {e}")
        return []


def validate_market_value(
        market_values: List[Dict],
        team_name: str,
        target_date: datetime,
        date_buffer: int = 60,
) -> Optional[int]:
    """
    Validate market value based on team and target date.

    Args:
        market_values (List[Dict]): Market value history for the player.
        team_name (str): The name of the team to validate against.
        target_date (datetime): The target date for market value.
        date_buffer (int): The date range buffer in days (default: 60).

    Returns:
        Optional[int]: The market value if a valid match is found, else None.
    """
    closest_value = None
    for entry in market_values:
        value_date = datetime.strptime(entry["date"], DATE_FORMAT)
        if abs(value_date - target_date) <= timedelta(days=date_buffer):
            if entry["clubName"].lower() == team_name.lower():
                logging.info(
                    f"Match found: {entry['clubName']} on {value_date} with value {entry['marketValue']}"
                )
                closest_value = entry["marketValue"]
    if closest_value is None:
        logging.warning(
            f"No valid market value match for team '{team_name}' within {date_buffer} days of {target_date}."
        )
    return closest_value


def process_player_values(file_path: str, season_end_year: int) -> None:
    """
    Process player values and update the dataset.

    Args:
        file_path (str): Path to the player dataset file.
        season_end_year (int): The season's end year.
    """
    import pandas as pd

    target_date = datetime.strptime(f"{season_end_year}-05-31", DATE_FORMAT)
    df = pd.read_csv(file_path)
    df["Market Value"] = None

    for index, row in df.iterrows():
        player_name = row["Player"]
        team_name = row["Squad"]

        logging.info(f"Processing player: {player_name} (Team: {team_name})")

        search_results = search_player(player_name)
        if not search_results:
            logging.warning(f"No search results found for {player_name}.")
            continue

        best_match = search_results[0]
        player_id = best_match.get("id")

        market_values = fetch_player_market_value(player_id)
        if not market_values:
            logging.warning(f"No market value history for player ID {player_id}.")
            continue

        value = validate_market_value(market_values, team_name, target_date)
        df.at[index, "Market Value"] = value or "N/A"

    output_path = os.path.join(
        os.path.dirname(file_path), f"updated_{os.path.basename(file_path)}"
    )
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
    import os

    for file_name, season_end_year in season_end_years.items():
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            logging.info(f"Processing file: {file_name} for season ending {season_end_year}.")
            try:
                process_player_values(file_path, season_end_year)
                logging.info(f"Successfully processed {file_name}.")
            except Exception as e:
                logging.error(f"Error processing file {file_name}: {e}")
        else:
            logging.warning(f"File {file_path} does not exist. Skipping.")

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

    # Ensure the input folder exists
    if not os.path.isdir(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist. Please verify the path.")
    else:
        logging.info(f"Starting processing of files in {INPUT_FOLDER}...")
        process_all_files(INPUT_FOLDER, SEASON_END_YEARS)
        logging.info("Processing complete.")
