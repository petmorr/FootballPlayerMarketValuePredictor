import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Logger Configuration
# ------------------------------------------------------------------------------
# Configure a logger for the web scraping process.
logging = configure_logger("web_scrape", "logging/web_scrape.log")

# ------------------------------------------------------------------------------
# Constants & Configuration
# ------------------------------------------------------------------------------
# Mapping of league names to their corresponding base URLs on FBref.
LEAGUES = {
    "Premier-League": "https://fbref.com/en/comps/9/",
    "La-Liga": "https://fbref.com/en/comps/12/",
    "Serie-A": "https://fbref.com/en/comps/11/",
    "Bundesliga": "https://fbref.com/en/comps/20/",
    "Ligue-1": "https://fbref.com/en/comps/13/",
}

# List of seasons to scrape data for.
SEASONS = ["2023-2024", "2022-2023", "2021-2022", "2020-2021", "2019-2020"]

# Output directory for storing scraped CSV files.
OUTPUT_DIR = "data/scraped"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def configure_driver() -> webdriver.Chrome:
    """
    Configure and initialize the Chrome WebDriver in headless mode.

    Returns:
        webdriver.Chrome: A configured Chrome WebDriver instance.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run browser in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        logging.info("WebDriver configured successfully.")
        return driver
    except Exception as e:
        logging.error(f"Failed to configure WebDriver: {e}")
        raise


def build_league_season_url(league_name: str, base_url: str, season: str) -> str:
    """
    Build the URL to scrape based on the league, base URL, and season.

    The URL format is constructed by appending the season and stats information.
    For example, for Premier-League and season 2023-2024 the URL might be:
    https://fbref.com/en/comps/9/2023-2024/stats/2023_2024-Premier_League-Stats

    Args:
        league_name (str): The name of the league (e.g., "Premier-League").
        base_url (str): The base URL for the league.
        season (str): The season string (e.g., "2023-2024").

    Returns:
        str: The complete URL to scrape.
    """
    # Replace '-' with '_' in season and league name for URL construction
    season_url = season.replace('-', '_')
    league_url = league_name.replace('-', '_')
    url = f"{base_url}{season}/stats/{season_url}-{league_url}-Stats"
    return url


def get_player_data_selenium(url: str, driver: webdriver.Chrome) -> pd.DataFrame:
    """
    Scrape player statistics data from a given URL using Selenium.

    This function navigates to the URL, waits until the player data table (with id "stats_standard")
    is loaded, and then uses pandas to parse the HTML table into a DataFrame.

    Args:
        url (str): The URL from which to scrape data.
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        pd.DataFrame: DataFrame containing the scraped player data.
                      Returns None if scraping fails.
    """
    try:
        logging.info(f"Accessing URL: {url}")
        driver.get(url)

        # Wait up to 10 seconds for the stats table to be present
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.ID, "stats_standard")))

        # Extract HTML of the table and convert it to a DataFrame
        table_html = table.get_attribute("outerHTML")
        df = pd.read_html(table_html)[0]
        logging.info(f"Successfully scraped data from: {url}")
        return df

    except Exception as e:
        logging.error(f"Failed to scrape data from {url}: {e}")
        return None


# ------------------------------------------------------------------------------
# Main Scraping Function
# ------------------------------------------------------------------------------

def scrape_league_data() -> None:
    """
    Scrape player data for all specified leagues and seasons.

    For each league and season, the function builds the URL, scrapes the player data,
    adds metadata columns (League and Season), and saves the data to CSV files.
    """
    # Initialize the WebDriver.
    driver = configure_driver()

    try:
        # Loop over each league and season combination.
        for league_name, base_url in LEAGUES.items():
            for season in SEASONS:
                url = build_league_season_url(league_name, base_url, season)
                logging.info(f"Scraping data for {league_name} - {season} from {url}")

                # Attempt to retrieve the player data
                df = get_player_data_selenium(url, driver)
                if df is not None:
                    # Append metadata columns for context
                    df["League"] = league_name
                    df["Season"] = season

                    # Construct a filename and save the DataFrame as CSV.
                    file_name = os.path.join(OUTPUT_DIR, f"{league_name}_{season}_player_data.csv")
                    df.to_csv(file_name, index=False)
                    logging.info(f"Data successfully saved to {file_name}")
                else:
                    logging.warning(f"No data found for {league_name} - {season}.")
    except Exception as e:
        logging.error(f"An error occurred during scraping: {e}")
    finally:
        # Ensure the WebDriver is closed to free up resources.
        driver.quit()
        logging.info("Web scraping process completed and WebDriver closed.")


# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Starting web scraping process...")
    scrape_league_data()
    logging.info("Web scraping script execution finished.")