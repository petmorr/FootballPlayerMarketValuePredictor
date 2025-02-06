import os
# Reconfigure stdout to use UTF-8 to avoid UnicodeDecode errors
import sys
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

import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ----------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("player_value")

# ----------------------------------------------------------------
# API Functions (Single Attempt)
# ----------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"  # Adjust as needed
DATE_FORMAT = "%Y-%m-%d"


def make_request_once(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """Makes a single GET request to the API and returns its JSON response."""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        logger.debug(f"Successful API call: {url}")
        return response.json()
    except requests.RequestException as e:
        logger.warning(f"API call failed for {url}: {e}")
        return {}


def fetch_player_market_value(player_id: str) -> Optional[List[Dict]]:
    """Fetches market value history for a player using a single API attempt."""
    endpoint = f"players/{player_id}/market_value"
    logger.info(f"Fetching market value for player ID {player_id} via API.")
    data = make_request_once(endpoint)
    return data.get("marketValueHistory", [])


def fetch_player_profile(player_id: str) -> Optional[str]:
    """Fetches the player's profile URL from the API."""
    endpoint = f"players/{player_id}/profile"
    logger.info(f"Fetching profile for player ID {player_id} via API.")
    data = make_request_once(endpoint)
    return data.get("url", None)


def validate_market_value(market_values: List[Dict], team_name: str,
                          season_start: datetime, season_end: datetime) -> Optional[Dict]:
    """
    Validates market value records using fuzzy matching against the team name
    and checking that the record date is within the season.
    Returns the record closest to season_end.
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
                logger.debug(f"Fuzzy match: score={match_score} for '{team_name}' vs '{club_name}' on {value_date}")
        except Exception as e:
            logger.error(f"Error processing market value entry: {e}")
    if not closest_entry:
        logger.warning(f"No valid market value match for '{team_name}' in the season.")
    return closest_entry


# ----------------------------------------------------------------
# Selenium Fallback Extraction Functions
# ----------------------------------------------------------------
def setup_driver() -> webdriver.Chrome:
    """Sets up Selenium Chrome WebDriver in headless mode with a fixed window size."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,800")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def handle_accept_and_continue(driver: webdriver.Chrome) -> None:
    """Handles the 'Accept & Continue' cookie modal if present."""
    try:
        logger.info("Checking for 'Accept & Continue' modal...")
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframes:
            if iframe.get_attribute("id") == "sp_message_iframe_953778":
                driver.switch_to.frame(iframe)
                btn = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@title='Accept & continue']"))
                )
                btn.click()
                logger.info("'Accept & Continue' button clicked.")
                break
    except Exception as e:
        logger.warning(f"Error handling cookie modal: {e}")
    finally:
        driver.switch_to.default_content()


def get_graph_container(driver: webdriver.Chrome) -> Optional[webdriver.remote.webelement.WebElement]:
    """
    Waits for the market value graph container (<tm-market-value-development-graph-integrated>)
    to appear and load. If not found immediately, scrolls and retries.
    """
    try:
        graph = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tm-market-value-development-graph-integrated"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", graph)
        WebDriverWait(driver, 60).until(lambda d: len(graph.get_attribute("innerHTML").strip()) > 0)
        logger.info("Graph container is present and loaded.")
        return graph
    except Exception as e:
        logger.error(f"Error getting graph container: {e}")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
        try:
            graph = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tm-market-value-development-graph-integrated"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", graph)
            WebDriverWait(driver, 30).until(lambda d: len(graph.get_attribute("innerHTML").strip()) > 0)
            logger.info("Graph container found after scrolling.")
            return graph
        except Exception as e2:
            logger.error(f"Error getting graph container after scrolling: {e2}")
            return None


def parse_tooltip(tooltip_text: str) -> Optional[Dict]:
    """
    Parses tooltip text expected in the following format:

        "Dec 15, 2014
         Market value: €250k
         Club: FC Winterthur
         Age: 19"

    Returns a dictionary with keys:
      - date (datetime object)
      - raw_date (string)
      - market_value (string)
      - full_text (the complete tooltip text)
    """
    lines = tooltip_text.split("\n")
    if len(lines) < 2:
        return None
    date_str = lines[0].strip()
    try:
        date_obj = datetime.strptime(date_str, "%b %d, %Y")
    except Exception as e:
        logger.error(f"Error parsing date from '{date_str}': {e}")
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


def extract_market_value_from_graph(driver: webdriver.Chrome) -> List[str]:
    """
    Uses the invisible voronoi-cell <path> elements (with class "voronoi-cell svelte-1plgtdf")
    as the interactive areas of the market value graph. For each such element, the script
    simulates a hover, waits up to 6 seconds for the tooltip (selector "div.chart-tooltip")
    to appear, and extracts its text. Duplicate tooltips (based on the date) are skipped.
    Returns a list of tooltip texts.
    """
    market_data = []
    try:
        logger.info("Waiting for the market value graph to load (Selenium fallback)...")
        graph = get_graph_container(driver)
        if not graph:
            logger.error("Graph container not found in fallback extraction.")
            return market_data

        # Wait for the voronoi cell elements.
        voronoi_cells = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "path.voronoi-cell.svelte-1plgtdf"))
        )
        num_cells = len(voronoi_cells)
        logger.info(f"Found {num_cells} voronoi cell elements in the graph (fallback).")
        if num_cells == 0:
            logger.error("No voronoi cell elements found; the page structure may have changed.")
            return market_data

        action = ActionChains(driver)
        seen_dates = set()
        # Increase sleep duration and tooltip wait timeout to allow AJAX to load
        for index, cell in enumerate(voronoi_cells):
            tooltip_text = None
            for attempt in range(3):
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", cell)
                    action.move_to_element(cell).perform()
                    logger.info(f"Hovered over voronoi cell {index + 1}/{num_cells} (Attempt {attempt + 1}).")
                    time.sleep(1.0)  # Increased sleep to allow tooltip to appear
                    tooltip = WebDriverWait(driver, 6).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.chart-tooltip"))
                    )
                    tooltip_text = tooltip.text.strip()
                    if tooltip_text:
                        break
                except TimeoutException:
                    logger.warning(f"Tooltip not found for cell {index + 1} (Attempt {attempt + 1}).")
                except Exception as e:
                    logger.error(f"Error hovering over cell {index + 1}: {e}")
            if tooltip_text:
                rec = parse_tooltip(tooltip_text)
                if rec and rec["raw_date"] not in seen_dates:
                    seen_dates.add(rec["raw_date"])
                    market_data.append(tooltip_text)
                    logger.info(f"Extracted tooltip for cell {index + 1}: {tooltip_text}")
                else:
                    logger.info(f"Duplicate or invalid tooltip for cell {index + 1} skipped.")
            else:
                logger.warning(f"No tooltip extracted for cell {index + 1}.")
        logger.info(f"Final extracted market data from graph (fallback): {market_data}")
    except TimeoutException:
        logger.error("Timeout while waiting for the market value graph or tooltip elements during fallback extraction.")
    except Exception as e:
        logger.error(f"Error during fallback extraction from graph: {e}")
    return market_data


def find_record_closest_to(target_date: datetime, records: List[Dict]) -> Optional[Dict]:
    """Returns the record whose date is closest to the target_date."""
    if not records:
        return None
    return min(records, key=lambda rec: abs(rec["date"] - target_date))


# ----------------------------------------------------------------
# Processing Player Values with Fallback
# ----------------------------------------------------------------
def process_player_values(file_path: str, season_end_year: int) -> None:
    """
    Processes a player dataset CSV file and updates each player's market value.
    First, it attempts to fetch market value history via the API (single attempt).
    If no valid market value is found, it falls back to Selenium extraction from the player's profile.
    If a CSV row lacks a PlayerID, the script attempts to retrieve one via the search endpoint.
    """
    season_start = datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT)
    season_end = datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT)

    df = pd.read_csv(file_path)
    df["Market Value"] = None

    for index, row in df.iterrows():
        player_name = row["Player"]
        team_name = row["Squad"]
        player_id = str(row.get("PlayerID", "")).strip()
        if not player_id:
            logger.warning(f"No PlayerID for {player_name}; attempting search to retrieve ID.")
            search_endpoint = f"players/search/{player_name}"
            search_results = make_request_once(search_endpoint).get("results", [])
            if search_results:
                best_match = max(search_results, key=lambda r: partial_ratio(player_name.lower(), r["name"].lower()))
                player_id = best_match.get("id", "").strip()
                if player_id:
                    logger.info(f"Found PlayerID {player_id} for {player_name} via search.")
                else:
                    logger.warning(f"Could not determine PlayerID for {player_name} from search results.")
                    continue
            else:
                logger.warning(f"No search results found for {player_name}.")
                continue

        logger.info(f"Processing player: {player_name} (Team: {team_name}, ID: {player_id})")

        # Try API approach first.
        search_endpoint = f"players/search/{player_name}"
        search_results = make_request_once(search_endpoint).get("results", [])
        valid_player = None
        for result in search_results:
            if partial_ratio(player_name.lower(), result["name"].lower()) >= 75:
                market_values = fetch_player_market_value(result["id"])
                if market_values:
                    closest_entry = validate_market_value(market_values, team_name, season_start, season_end)
                    if closest_entry:
                        valid_player = result
                        df.at[index, "Market Value"] = closest_entry["marketValue"]
                        logger.info(f"Assigned API market value {closest_entry['marketValue']} for {player_name} "
                                    f"(Team: {team_name}) from date {closest_entry['date']}.")
                        break

        # Fallback to Selenium extraction if API approach did not yield valid data.
        if not valid_player:
            logger.warning(
                f"API did not yield a valid market value for {player_name} (Team: {team_name}). Using fallback extraction.")
            profile_url = fetch_player_profile(player_id)
            if not profile_url:
                logger.warning(f"No profile URL found for {player_name} (ID: {player_id}); skipping fallback.")
                continue
            selenium_driver = setup_driver()
            try:
                selenium_driver.get(profile_url)
                handle_accept_and_continue(selenium_driver)
                # Increase wait time to allow the graph to load
                time.sleep(15)
                fallback_data = extract_market_value_from_graph(selenium_driver)
                if fallback_data:
                    fallback_records = []
                    seen = set()
                    for tooltip in fallback_data:
                        rec = parse_tooltip(tooltip)
                        if rec and rec["raw_date"] not in seen:
                            seen.add(rec["raw_date"])
                            fallback_records.append(rec)
                    if fallback_records:
                        fallback_record = find_record_closest_to(season_end, fallback_records)
                        if fallback_record:
                            df.at[index, "Market Value"] = fallback_record["market_value"]
                            logger.info(
                                f"Fallback assigned market value {fallback_record['market_value']} for {player_name} "
                                f"(Team: {team_name}) from date {fallback_record['raw_date']}.")
                        else:
                            logger.warning(f"No fallback record close to target season for {player_name}.")
                    else:
                        logger.warning(f"No fallback records parsed for {player_name}.")
                else:
                    logger.warning(f"Selenium fallback extraction returned no data for {player_name}.")
            except Exception as e:
                logger.error(f"Error during fallback extraction for {player_name}: {e}")
            finally:
                selenium_driver.quit()

    output_dir = "./data/updated"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"updated_{os.path.basename(file_path)}")
    df.to_csv(output_path, index=False)
    logger.info(f"Updated dataset saved to {output_path}")


def process_all_files(input_folder: str, season_end_years: Dict[str, int]) -> None:
    """
    Processes all CSV files in the input folder and updates each with market value data.
    """
    for file_name, season_end_year in season_end_years.items():
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            logger.info(f"Processing file: {file_path} for season ending {season_end_year}.")
            try:
                process_player_values(file_path, season_end_year)
                logger.info(f"Successfully processed {file_path}.")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        else:
            logger.warning(f"File {file_path} does not exist. Skipping.")


def main():
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
        logger.error(f"Input folder '{INPUT_FOLDER}' does not exist. Please verify the path.")
    else:
        logger.info("Starting processing of player values...")
        process_all_files(INPUT_FOLDER, SEASON_END_YEARS)
        logger.info("Processing complete.")


if __name__ == "__main__":
    main()
