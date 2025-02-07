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
    # Set a fixed window size to help ensure consistent rendering.
    options.add_argument("--window-size=1280,800")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def handle_accept_and_continue(driver):
    """Handles the 'Accept & Continue' modal if it appears, checking both iframe and direct modal cases."""
    try:
        logging.info("Checking for 'Accept & Continue' modal...")

        # === First, try handling the iframe version ===
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframes:
            iframe_id = iframe.get_attribute("id")
            if "sp_message_iframe" in iframe_id:  # Matches any iframe containing this pattern
                logging.info(f"Switching to iframe: {iframe_id}")
                driver.switch_to.frame(iframe)

                try:
                    btn = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(@title, 'Accept & continue')]"))
                    )
                    btn.click()
                    logging.info("'Accept & Continue' button clicked inside iframe.")
                    driver.switch_to.default_content()
                    return  # Exit function after handling modal
                except Exception as e:
                    logging.warning(f"Error clicking button inside iframe: {e}")

                driver.switch_to.default_content()  # Ensure we switch back

        # === If no iframe found, try handling modal directly in the document ===
        logging.info("Checking for modal directly in the document...")
        try:
            modal = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "notice"))  # The modal <div id="notice">
            )
            accept_button = modal.find_element(By.XPATH, ".//button[contains(@title, 'Accept & continue')]")
            accept_button.click()
            logging.info("'Accept & Continue' button clicked inside modal.")
        except Exception as e:
            logging.warning(f"Error clicking button inside modal: {e}")

    except Exception as e:
        logging.warning(f"Error handling cookie modal: {e}")

    finally:
        driver.switch_to.default_content()  # Ensure we always switch back to main document


def get_graph_container(driver):
    """
    Waits up to 60 seconds for the market–value graph container element
    (<tm-market-value-development-graph-integrated>) to appear and be loaded.
    If not immediately found, scrolls down and retries.
    """
    try:
        graph = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tm-market-value-development-graph-integrated"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", graph)
        # Wait until the innerHTML is non‑empty (i.e. the Svelte component has rendered)
        WebDriverWait(driver, 60).until(lambda d: len(graph.get_attribute("innerHTML").strip()) > 0)
        logging.info("Graph container is present and loaded.")
        return graph
    except Exception as e:
        logging.error(f"Error getting graph container: {e}")
        logging.info("Scrolling to bottom and waiting 10 seconds...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
        try:
            graph = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tm-market-value-development-graph-integrated"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", graph)
            WebDriverWait(driver, 30).until(lambda d: len(graph.get_attribute("innerHTML").strip()) > 0)
            logging.info("Graph container found after scrolling.")
            return graph
        except Exception as e2:
            logging.error(f"Error getting graph container after scrolling: {e2}")
            return None


def parse_tooltip(tooltip_text):
    """
    Parses tooltip text expected in the format:

      "Dec 15, 2014
       Market value: €250k
       Club: FC Winterthur
       Age: 19"

    Returns a dictionary with:
      - date: datetime object (from the first line)
      - raw_date: original date string
      - market_value: market value string
      - full_text: full tooltip text
    """
    lines = tooltip_text.split("\n")
    if len(lines) < 2:
        return None
    date_str = lines[0].strip()
    try:
        date_obj = datetime.strptime(date_str, "%b %d, %Y")
    except Exception as e:
        logging.error(f"Error parsing date from '{date_str}': {e}")
        return None
    market_value = None
    if len(lines) >= 2:
        parts = lines[1].split("Market value:")
        if len(parts) >= 2:
            market_value = parts[1].strip()
        else:
            market_value = lines[1].strip()
    return {
        "date": date_obj,
        "raw_date": date_str,
        "market_value": market_value,
        "full_text": tooltip_text
    }


def extract_market_value_from_graph(driver):
    """
    Extracts market value data by hovering over the voronoi-cell path elements.
    Uses multiple techniques to ensure all tooltips are captured.
    """
    market_data = []
    try:
        logging.info("Waiting for the market value graph to load...")
        graph = get_graph_container(driver)
        if not graph:
            logging.error("Graph container not found.")
            return market_data

        voronoi_cells = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "path.voronoi-cell.svelte-1plgtdf"))
        )
        num_cells = len(voronoi_cells)
        logging.info(f"Found {num_cells} voronoi cell elements in the graph.")

        seen_dates = set()

        for index, cell in enumerate(voronoi_cells):
            tooltip_text = None

            for attempt in range(4):  # Try multiple times
                try:
                    # Ensure the element is visible
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", cell)
                    time.sleep(0.5)

                    # **Use JavaScript to trigger hover (more reliable than ActionChains)**
                    driver.execute_script(
                        "var evt = document.createEvent('MouseEvents');"
                        "evt.initMouseEvent('mouseover', true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);"
                        "arguments[0].dispatchEvent(evt);", cell
                    )

                    logging.info(f"Hovered over voronoi cell {index + 1}/{num_cells} (Attempt {attempt + 1}).")
                    time.sleep(1.5)  # Give tooltip enough time to appear

                    tooltip = WebDriverWait(driver, 3).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.chart-tooltip"))
                    )
                    tooltip_text = tooltip.text.strip()
                    if tooltip_text:
                        break  # Stop trying once tooltip is found

                except TimeoutException:
                    logging.warning(f"Tooltip not found for cell {index + 1} (Attempt {attempt + 1}).")
                except Exception as e:
                    logging.error(f"Error hovering over cell {index + 1}: {e}")

            if tooltip_text:
                rec = parse_tooltip(tooltip_text)
                if rec and rec["raw_date"] not in seen_dates:
                    seen_dates.add(rec["raw_date"])
                    market_data.append(tooltip_text)
                    logging.info(f"Extracted tooltip for cell {index + 1}: {tooltip_text}")
                else:
                    logging.info(f"Duplicate or invalid tooltip for cell {index + 1} skipped.")
            else:
                logging.warning(f"No tooltip extracted for cell {index + 1}.")

        logging.info(f"Final extracted market data: {market_data}")
    except TimeoutException:
        logging.error("Timeout while waiting for market value graph or tooltips.")
    except Exception as e:
        logging.error(f"Error extracting market data: {e}")

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
            time.sleep(5)  # Allow time for the page to load

            # ✅ Handle cookie modal before proceeding
            handle_accept_and_continue(driver)

            # ✅ Extract market values using Selenium
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