import os
import re
import time
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import requests
from rapidfuzz.fuzz import partial_ratio
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from logging_config import configure_logger

# Configure logger
logging = configure_logger("player_value", "player_value.log")

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Constants
DATE_FORMAT = "%Y-%m-%d"
MAX_API_RETRIES = 3


def make_request_with_retry(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """ Makes a GET request to the API with a max of 3 retries. """
    url = f"{API_BASE_URL}/{endpoint}"
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            logging.debug(f"Successful API call on attempt {attempt}: {url}")
            return response.json()
        except requests.RequestException as e:
            logging.warning(f"API call failed (attempt {attempt}/{MAX_API_RETRIES}): {e}")
            time.sleep(min(2 ** attempt, 60))
    return {}


def fetch_player_id(player_name: str) -> Optional[str]:
    """ Searches for a player by name and retrieves their Player ID. """
    endpoint = f"players/search/{player_name}"
    logging.info(f"Searching for Player ID for {player_name}.")
    search_results = make_request_with_retry(endpoint).get("results", [])

    if not search_results:
        logging.warning(f"No search results found for {player_name}.")
        return None

    best_match = max(search_results, key=lambda r: partial_ratio(player_name.lower(), r["name"].lower()), default=None)

    if best_match:
        logging.info(f"Found Player ID {best_match['id']} for {player_name}.")
        return best_match["id"]

    logging.warning(f"No suitable Player ID match found for {player_name}.")
    return None


def fetch_player_profile(player_id: str) -> Optional[str]:
    """ Fetches the player's Transfermarkt profile URL using their ID. """
    endpoint = f"players/{player_id}/profile"
    logging.info(f"Fetching profile URL for player ID {player_id}.")
    data = make_request_with_retry(endpoint)
    return data.get("url", None)


def fetch_player_market_value(player_id: str) -> Optional[List[Dict]]:
    """ Fetches market value history for a player using the API. """
    endpoint = f"players/{player_id}/market_value"
    logging.info(f"Fetching market value for player ID {player_id}.")
    return make_request_with_retry(endpoint).get("marketValueHistory", [])


def setup_driver() -> webdriver.Chrome:
    """ Sets up Selenium Chrome WebDriver in headless mode. """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,800")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def extract_market_value_from_graph(driver: webdriver.Chrome) -> List[Dict]:
    """ Extracts market value data from the Transfermarkt graph using Selenium. """
    market_data = []
    try:
        logging.info("Waiting for the market value graph to load...")
        time.sleep(10)

        graph = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tm-market-value-development-graph-integrated"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", graph)
        time.sleep(5)

        voronoi_cells = driver.find_elements(By.CSS_SELECTOR, "path.voronoi-cell.svelte-1plgtdf")
        logging.info(f"Found {len(voronoi_cells)} voronoi cell elements.")

        if not voronoi_cells:
            logging.error("No voronoi cells found! Page structure may have changed.")
            return market_data

        action = ActionChains(driver)
        for cell in voronoi_cells:
            action.move_to_element(cell).perform()
            time.sleep(1)
            try:
                tooltip = WebDriverWait(driver, 4).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, "div.chart-tooltip"))
                )
                tooltip_text = tooltip.text.strip()
                if tooltip_text:
                    date_str, value = tooltip_text.split("\n")[:2]
                    market_data.append({"date": date_str, "market_value": value})
            except TimeoutException:
                logging.warning("Tooltip did not appear for one of the voronoi cells.")
    except Exception as e:
        logging.error(f"Error extracting market value from graph: {e}")
    return market_data


def process_player(player_name: str) -> Optional[List[Dict]]:
    """ Retrieves player ID, then fetches market value from API or Selenium. """
    player_id = fetch_player_id(player_name)
    if not player_id:
        logging.warning(f"Could not determine Player ID for {player_name}. Skipping.")
        return None

    market_values = fetch_player_market_value(player_id)

    if not market_values:
        logging.warning(f"No market values from API for {player_name}. Using Selenium fallback.")
        profile_url = fetch_player_profile(player_id)
        if not profile_url:
            logging.error(f"Could not retrieve profile URL for {player_name}.")
            return None

        driver = setup_driver()
        try:
            driver.get(profile_url)
            time.sleep(10)
            market_values = extract_market_value_from_graph(driver)
        except Exception as e:
            logging.error(f"Error processing {player_name} with Selenium: {e}")
        finally:
            driver.quit()

    return market_values


def validate_market_value(
        market_values: List[Dict], team_name: str, season_start: datetime, season_end: datetime
) -> Optional[Dict]:
    """
    Matches a market value entry based on team and season using fuzzy logic.

    Args:
        market_values (List[Dict]): Market value history for the player.
        team_name (str): The name of the team to validate against.
        season_start (datetime): Start date of the season.
        season_end (datetime): End date of the season.

    Returns:
        Optional[Dict]: The closest market value entry to the season end date, or None if not found.
    """
    closest_entry = None
    closest_date_diff = float("inf")

    for entry in market_values:
        try:
            value_date = datetime.strptime(entry["date"], DATE_FORMAT)
            club_name = entry.get("clubName", "")
            match_score = partial_ratio(team_name.lower(), club_name.lower())

            # Ensure the entry falls within the season range and has a strong fuzzy match
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
    """ Processes a dataset and updates each player's market value. """
    df = pd.read_csv(file_path)
    df["Market Value"] = None

    for index, row in df.iterrows():
        player_name = row["Player"]
        team_name = row["Squad"]

        logging.info(f"Processing player: {player_name} (Team: {team_name})")

        # Fetch Player ID
        player_id = fetch_player_id(player_name)
        if not player_id:
            logging.warning(f"No Player ID found for {player_name}. Skipping.")
            continue

        # Fetch Market Value
        market_values = fetch_player_market_value(player_id)
        if not market_values:
            logging.warning(f"No API market value found for {player_name}, using Selenium fallback.")
            market_values = process_player(player_name)

        # Validate Market Value
        if market_values:
            closest_entry = validate_market_value(market_values, team_name,
                                                  datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT),
                                                  datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT))
            if closest_entry and "marketValue" in closest_entry:
                df.at[index, "Market Value"] = closest_entry["marketValue"]
                logging.info(f"Assigned Market Value {closest_entry['marketValue']} "
                             f"for {player_name} (Team: {team_name}) from date {closest_entry['date']}.")
            else:
                logging.warning(f"No suitable market value found for {player_name} (Team: {team_name}).")
        else:
            logging.warning(f"No market value data available for {player_name} (Team: {team_name}).")

    # Save updated dataset
    output_dir = "./data/updated"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"updated_{os.path.basename(file_path)}")
    df.to_csv(output_path, index=False)
    logging.info(f"Updated dataset saved to {output_path}")


def process_all_files(input_folder: str, season_end_years: Dict[str, int]) -> None:
    """ Processes all CSV files in the input folder. """
    for file_name, season_end_year in season_end_years.items():
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            logging.info(f"Processing file: {file_path} for season ending {season_end_year}.")
            try:
                process_player_values(file_path, season_end_year)
                logging.info(f"Successfully processed {file_path}.")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")


def extract_season_year(file_name: str) -> Optional[int]:
    """
    Extracts the season end year from the file name using regex.

    Args:
        file_name (str): The name of the file.

    Returns:
        Optional[int]: The extracted season end year, or None if not found.
    """
    match = re.search(r"(\d{4})-(\d{4})", file_name)
    if match:
        return int(match.group(2))  # Extract the second year as the season end year
    return None


def main():
    INPUT_FOLDER = "./data/cleaned"

    if not os.path.isdir(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist.")
    else:
        logging.info("Starting processing of player values...")
        for file in os.listdir(INPUT_FOLDER):
            season_end_year = extract_season_year(file)
            if season_end_year:
                logging.info(f"Processing {file} for season ending {season_end_year}")
                process_player_values(os.path.join(INPUT_FOLDER, file), season_end_year)
            else:
                logging.warning(f"Skipping {file}: Could not determine season year.")
        logging.info("Processing complete.")


if __name__ == "__main__":
    main()