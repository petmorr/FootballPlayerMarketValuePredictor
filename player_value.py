import functools
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Dict, Any

import ftfy
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

# ------------------------------------------------------------------------------
# Logger and API Configuration
# ------------------------------------------------------------------------------
logging = configure_logger("player_value", "player_value.log")

API_BASE_URL: str = "http://localhost:8000"
DATE_FORMAT: str = "%Y-%m-%d"
MAX_API_RETRIES: int = 3


# ------------------------------------------------------------------------------
# File I/O Helpers
# ------------------------------------------------------------------------------
def read_input_file(file_path: str) -> pd.DataFrame:
    """
    Reads an input file that may be a gzipped CSV or a Parquet file.
    """
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.csv.gz'):
        return pd.read_csv(file_path, compression='gzip', encoding='utf-8')
    else:
        return pd.read_csv(file_path, encoding='utf-8')


def write_output_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Writes the DataFrame to the updated folder in the same format as the input.
    """
    output_dir = "./data/updated"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"updated_{os.path.basename(file_path)}")
    if file_path.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    elif file_path.endswith('.csv.gz'):
        df.to_csv(output_file, index=False, compression='gzip')
    else:
        df.to_csv(output_file, index=False)
    logging.info(f"Updated dataset saved to {output_file}")


# ------------------------------------------------------------------------------
# Text Encoding Fix (using ftfy)
# ------------------------------------------------------------------------------
def fix_encoding(s: str) -> str:
    """
    Uses ftfy to repair mojibake (mis-encoded text).
    This should convert mis-decoded names like "Thiago AlcÃ¡ntara" to "Thiago Alcântara".
    """
    try:
        return ftfy.fix_text(s)
    except Exception as e:
        logging.error(f"Encoding fix failed for {s}: {e}")
        return s


# ------------------------------------------------------------------------------
# API Request Functions (with caching and session reuse)
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
    The player's name is fixed for encoding issues before lookup.
    If no results are found, and the name consists of exactly two parts, then try a reversed name lookup.
    """
    fixed_name = fix_encoding(player_name)
    endpoint = f"players/search/{fixed_name}"
    logging.info(f"Searching for Player ID for {fixed_name}.")
    search_results = make_request_with_retry(endpoint).get("results", [])

    if not search_results:
        # Try reversing the name if it contains exactly two parts.
        name_parts = fixed_name.split()
        if len(name_parts) == 2:
            reversed_name = f"{name_parts[1]} {name_parts[0]}"
            logging.info(f"No search results found for {fixed_name}. Trying reversed name: {reversed_name}.")
            endpoint = f"players/search/{reversed_name}"
            search_results = make_request_with_retry(endpoint).get("results", [])
            if search_results:
                best_match = max(
                    search_results,
                    key=lambda r: partial_ratio(reversed_name.lower(), r["name"].lower()),
                    default=None
                )
                if best_match:
                    logging.info(f"Found Player ID {best_match['id']} for reversed name {reversed_name}.")
                    return best_match["id"]
            else:
                logging.warning(f"No search results found for reversed name {reversed_name}.")
        logging.warning(f"No search results found for {fixed_name}.")
        return None

    best_match = max(
        search_results,
        key=lambda r: partial_ratio(fixed_name.lower(), r["name"].lower()),
        default=None
    )
    if best_match:
        logging.info(f"Found Player ID {best_match['id']} for {fixed_name}.")
        return best_match["id"]
    logging.warning(f"No suitable Player ID match found for {fixed_name}.")
    return None


@functools.lru_cache(maxsize=1000)
def fetch_player_profile(player_id: str) -> Optional[str]:
    """
    Fetches the player's profile URL using their Player ID.
    """
    endpoint = f"players/{player_id}/profile"
    logging.info(f"Fetching profile URL for player ID {player_id}.")
    data = make_request_with_retry(endpoint)
    return data.get("url", None)


def fetch_player_market_value(player_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves the market value history for a player from the API.
    """
    endpoint = f"players/{player_id}/market_value"
    logging.info(f"Fetching market value for player ID {player_id}.")
    return make_request_with_retry(endpoint).get("marketValueHistory", [])


# ------------------------------------------------------------------------------
# Selenium Driver Setup and Helper Functions
# ------------------------------------------------------------------------------
def setup_driver() -> webdriver.Chrome:
    """
    Sets up and returns a headless Chrome WebDriver.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,800")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def handle_accept_and_continue(driver: webdriver.Chrome) -> None:
    """
    Handles the 'Accept & Continue' modal (e.g. cookie consent) if present.
    """
    try:
        logging.info("Checking for 'Accept & Continue' modal...")
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframes:
            iframe_id = iframe.get_attribute("id")
            if "sp_message_iframe" in iframe_id:
                logging.info(f"Switching to iframe: {iframe_id}")
                driver.switch_to.frame(iframe)
                try:
                    btn = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(@title, 'Accept & continue')]"))
                    )
                    btn.click()
                    logging.info("'Accept & Continue' button clicked inside iframe.")
                    driver.switch_to.default_content()
                    return
                except Exception as e:
                    logging.warning(f"Error clicking button inside iframe: {e}")
                driver.switch_to.default_content()
        logging.info("Checking for modal directly in the document...")
        try:
            modal = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "notice"))
            )
            accept_button = modal.find_element(By.XPATH, ".//button[contains(@title, 'Accept & continue')]")
            accept_button.click()
            logging.info("'Accept & Continue' button clicked inside modal.")
        except Exception as e:
            logging.warning(f"Error clicking button inside modal: {e}")
    except Exception as e:
        logging.warning(f"Error handling cookie modal: {e}")
    finally:
        driver.switch_to.default_content()


def get_graph_container(driver) -> Optional[webdriver.remote.webelement.WebElement]:
    """
    Attempts to locate the market value graph container.
    Tries several selectors:
      1. The custom element with shadow DOM ("tm-market-value-development-graph-integrated")
      2. A similar custom element ("tm-market-value-development-graph")
      3. A generic container ("div.chart-container")
    """
    selectors = [
        "tm-market-value-development-graph-integrated",
        "tm-market-value-development-graph",
        "div.chart-container"
    ]
    for sel in selectors:
        try:
            logging.info(f"Trying to locate graph container using selector: {sel}")
            container = WebDriverWait(driver, 60).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, sel))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", container)
            if sel.startswith("tm-"):
                try:
                    shadow_root = driver.execute_script("return arguments[0].shadowRoot", container)
                    if shadow_root:
                        graph = WebDriverWait(driver, 60).until(
                            lambda d: shadow_root.find_element(By.CSS_SELECTOR, "svg")
                        )
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", graph)
                        logging.info("Graph container loaded from shadow DOM.")
                        return graph
                except Exception as e:
                    logging.warning(f"Could not retrieve shadow DOM for selector {sel}: {e}")
            logging.info("Graph container found.")
            return container
        except TimeoutException:
            logging.warning(f"Graph container not found using selector {sel} on initial attempt.")
        except Exception as e:
            logging.error(f"Unexpected error using selector {sel}: {e}")
        logging.info("Scrolling to bottom and retrying after 10 seconds...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
    logging.error("Graph container not found after trying all selectors.")
    with open("debug_page_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    logging.info("Saved page source as debug_page_source.html.")
    return None


# ------------------------------------------------------------------------------
# Tooltip Parsing and Market Value Extraction
# ------------------------------------------------------------------------------
def parse_tooltip(tooltip_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses tooltip text containing market value information.
    Expected format (example):
      "Dec 15, 2014
       Market value: €250k
       Club: FC Winterthur
       Age: 19"
    Returns a dictionary with parsed data or None if parsing fails.
    The returned dictionary contains:
      - date: a datetime object
      - raw_date: the original date string
      - marketValue: the extracted market value string
      - clubName: the extracted club name (if present)
      - full_text: the full tooltip text
    """
    lines = tooltip_text.splitlines()
    if not lines:
        return None

    try:
        date_obj = datetime.strptime(lines[0].strip(), "%b %d, %Y")
    except Exception as e:
        logging.error(f"Error parsing date from '{lines[0]}': {e}")
        return None

    market_value = None
    club_name = None
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Market value:"):
            market_value = line.split("Market value:")[1].strip()
        elif line.startswith("Club:"):
            club_name = line.split("Club:")[1].strip()

    return {
        "date": date_obj,  # stored as a datetime object
        "raw_date": lines[0].strip(),
        "marketValue": market_value,  # key matches expected downstream
        "clubName": club_name,
        "full_text": tooltip_text
    }


def extract_market_value_from_graph(driver: webdriver.Chrome) -> List[str]:
    """
    Extracts market value tooltips by hovering over voronoi-cell elements in the graph.
    Returns a list of tooltip texts.
    """
    market_data: List[str] = []
    try:
        logging.info("Waiting for the market value graph to load...")
        graph = get_graph_container(driver)
        if not graph:
            logging.error("Graph container not found.")
            return market_data
        voronoi_cells = graph.find_elements(By.CSS_SELECTOR, "path.voronoi-cell.svelte-1plgtdf")
        num_cells = len(voronoi_cells)
        logging.info(f"Found {num_cells} voronoi cell elements in the graph.")
        seen_dates = set()
        for index, cell in enumerate(voronoi_cells):
            tooltip_text: Optional[str] = None
            for attempt in range(4):
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", cell)
                    time.sleep(0.5)
                    driver.execute_script(
                        "var evt = document.createEvent('MouseEvents');"
                        "evt.initMouseEvent('mouseover', true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);"
                        "arguments[0].dispatchEvent(evt);", cell
                    )
                    logging.info(f"Hovered over voronoi cell {index + 1}/{num_cells} (Attempt {attempt + 1}).")
                    time.sleep(1.5)
                    tooltip = WebDriverWait(driver, 3).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.chart-tooltip"))
                    )
                    tooltip_text = tooltip.text.strip()
                    if tooltip_text:
                        break
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


# ------------------------------------------------------------------------------
# Player Processing and Market Value Validation
# ------------------------------------------------------------------------------
def process_player(player_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Processes a player by retrieving their Player ID and then fetching their market value data.
    Uses Selenium fallback if API data is not available.
    """
    fixed_name = fix_encoding(player_name)
    player_id = fetch_player_id(fixed_name)
    if not player_id:
        logging.warning(f"Could not determine Player ID for {fixed_name}. Skipping.")
        return None
    market_values = fetch_player_market_value(player_id)
    if not market_values:
        logging.warning(f"No market values from API for {fixed_name}. Using Selenium fallback.")
        profile_url = fetch_player_profile(player_id)
        if not profile_url:
            logging.error(f"Could not retrieve profile URL for {fixed_name}.")
            return None
        driver = setup_driver()
        try:
            driver.get(profile_url)
            time.sleep(5)  # Allow the page to load
            handle_accept_and_continue(driver)
            tooltip_texts = extract_market_value_from_graph(driver)
            # Parse each tooltip only once and filter out failed parses.
            market_values = [parse_tooltip(text) for text in tooltip_texts if parse_tooltip(text)]
        except Exception as e:
            logging.error(f"Error processing {fixed_name} with Selenium: {e}")
        finally:
            driver.quit()
    return market_values


def validate_market_value(
        market_values: List[Dict[str, Any]],
        team_name: str,
        season_start: datetime,
        season_end: datetime
) -> Optional[Dict[str, Any]]:
    """
    Validates and selects a market value entry based on team and season info.
    Uses fuzzy matching on team names (and only considers entries with a clubName)
    and chooses the entry closest to season end.
    """
    closest_entry = None
    closest_date_diff = float("inf")
    for entry in market_values:
        try:
            # Handle date: if already a datetime, use it; otherwise, parse it.
            if isinstance(entry["date"], datetime):
                value_date = entry["date"]
            else:
                value_date = datetime.strptime(entry["date"], DATE_FORMAT)
            club_name = entry.get("clubName", "") or ""
            match_score = partial_ratio(team_name.lower(), club_name.lower())
            logging.debug(f"Fuzzy match score: {match_score} for '{team_name}' vs '{club_name}' on {value_date}")
            if season_start <= value_date <= season_end and match_score >= 80:
                date_diff = abs((value_date - season_end).days)
                if date_diff < closest_date_diff:
                    closest_date_diff = date_diff
                    closest_entry = entry
        except Exception as e:
            logging.error(f"Error processing market value entry: {e}")
    if not closest_entry:
        logging.warning(f"No valid market value match for '{team_name}' in the season.")
    return closest_entry


# ------------------------------------------------------------------------------
# Data Processing Functions
# ------------------------------------------------------------------------------
def process_player_values(file_path: str, season_end_year: int) -> None:
    """
    Processes a preprocessed file (CSV, CSV.gz, or Parquet) to update each player's market value.
    Each row is processed concurrently.
    """
    df = read_input_file(file_path)
    df["Market Value"] = None

    def process_row(index, row):
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
            logging.warning(f"No API market value found for {fixed_name}, using Selenium fallback.")
            market_values = process_player(fixed_name)
        if market_values:
            season_start = datetime.strptime(f"{season_end_year - 1}-07-01", DATE_FORMAT)
            season_end = datetime.strptime(f"{season_end_year}-06-30", DATE_FORMAT)
            closest_entry = validate_market_value(market_values, team_name, season_start, season_end)
            if closest_entry and "marketValue" in closest_entry:
                # Use 'raw_date' if available; otherwise, fallback to 'date' (or "unknown date")
                date_info = closest_entry.get("raw_date", closest_entry.get("date", "unknown date"))
                logging.info(
                    f"Assigned Market Value {closest_entry['marketValue']} for {fixed_name} (Team: {team_name}) from date {date_info}."
                )
                return index, closest_entry["marketValue"]
            else:
                logging.warning(f"No suitable market value found for {fixed_name} (Team: {team_name}).")
                return index, None
        else:
            logging.warning(f"No market value data available for {fixed_name} (Team: {team_name}).")
            return index, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, index, row) for index, row in df.iterrows()]
        for future in as_completed(futures):
            index, market_value = future.result()
            if market_value is not None:
                df.at[index, "Market Value"] = market_value

    write_output_file(df, file_path)


def process_all_files(input_folder: str, season_end_years: Dict[str, int]) -> None:
    """
    Processes all files in the input folder using the provided season mapping.
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
    Iterates over files, extracts season info, and processes each file.
    """
    INPUT_FOLDER = "./data/cleaned"
    if not os.path.isdir(INPUT_FOLDER):
        logging.error(f"Input folder '{INPUT_FOLDER}' does not exist.")
        return
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