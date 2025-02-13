import os
import time
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from logging_config import configure_logger

# ------------------------------------------------------------------------------
# Logger Configuration
# ------------------------------------------------------------------------------
logging = configure_logger("web_scrape", "web_scrape.log")

# ------------------------------------------------------------------------------
# Constants & Configuration
# ------------------------------------------------------------------------------
LEAGUES = {
    "Premier-League": "https://fbref.com/en/comps/9/",
    "La-Liga": "https://fbref.com/en/comps/12/",
    "Serie-A": "https://fbref.com/en/comps/11/",
    "Bundesliga": "https://fbref.com/en/comps/20/",
    "Ligue-1": "https://fbref.com/en/comps/13/",
}
SEASONS = ["2023-2024", "2022-2023", "2021-2022", "2020-2021", "2019-2020"]
OUTPUT_DIR = "../data/scraped"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def configure_driver() -> webdriver.Chrome:
    """
    Configure and initialize the Chrome WebDriver in headless mode with a proper user-agent.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )
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

    Example URL:
      https://fbref.com/en/comps/9/2023-2024/stats/2023_2024-Premier_League-Stats
    """
    season_url = season.replace('-', '_')
    league_url = league_name.replace('-', '_')
    url = f"{base_url}{season}/stats/{season_url}-{league_url}-Stats"
    return url


def get_player_data_selenium(url: str, driver: webdriver.Chrome) -> pd.DataFrame:
    """
    Scrape player statistics data from a given URL using Selenium.

    Steps:
      1. Load the URL and wait for the page to load.
      2. Obtain the page source and parse it with BeautifulSoup.
      3. Try to find the table with id "stats_standard" directly.
      4. If not found, look for a <div> with id "all_stats_standard" and search its HTML comments.
      5. If still not found, search all comments for the string "stats_standard".
      6. Wrap the obtained HTML in a StringIO object and pass it to pd.read_html.

    Returns:
        pd.DataFrame: DataFrame containing the scraped data, or None if not found.
    """
    try:
        logging.info(f"Accessing URL: {url}")
        driver.get(url)
        # Wait until the page reports it is complete.
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(2)  # Extra delay for dynamic content
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Attempt 1: Find the table directly.
        table = soup.find("table", id="stats_standard")
        table_html = str(table) if table else None

        # Attempt 2: Look inside the div that usually holds the commented-out table.
        if not table_html:
            div_all = soup.find("div", id="all_stats_standard")
            if div_all:
                comments = div_all.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    if "stats_standard" in comment:
                        table_html = comment
                        break

        # Attempt 3: Fallback to scanning all comments in the page.
        if not table_html:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if "stats_standard" in comment:
                    table_html = comment
                    break

        if not table_html:
            logging.error(f"Could not find table 'stats_standard' in page source from {url}")
            return None

        # Wrap the HTML string in StringIO to avoid FutureWarning.
        df_list = pd.read_html(StringIO(table_html))
        if df_list:
            df = df_list[0]
            logging.info(f"Successfully scraped data from: {url}")
            return df
        else:
            logging.error(f"pd.read_html did not return any tables for {url}")
            return None
    except Exception as e:
        logging.error(f"Failed to scrape data from {url}: {e}")
        return None


def scrape_league_data() -> None:
    """
    Scrape player data for all specified leagues and seasons.

    For each league and season, the function builds the URL, scrapes the player data,
    appends metadata columns (League and Season), and saves the data as a CSV file.
    """
    driver = configure_driver()
    try:
        for league_name, base_url in LEAGUES.items():
            for season in SEASONS:
                try:
                    url = build_league_season_url(league_name, base_url, season)
                    logging.info(f"Scraping data for {league_name} - {season} from {url}")
                    df = get_player_data_selenium(url, driver)
                    if df is not None:
                        df["League"] = league_name
                        df["Season"] = season
                        file_name = os.path.join(OUTPUT_DIR, f"{league_name}_{season}_player_data.csv")
                        df.to_csv(file_name, index=False)
                        logging.info(f"Data successfully saved to {file_name}")
                    else:
                        logging.warning(f"No data found for {league_name} - {season}.")
                except Exception as inner_e:
                    logging.error(f"Error processing {league_name} - {season}: {inner_e}")
                time.sleep(2)  # Pause between requests to be polite
    except Exception as e:
        logging.error(f"An error occurred during scraping: {e}")
    finally:
        driver.quit()
        logging.info("Web scraping process completed and WebDriver closed.")


# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Starting web scraping process...")
    scrape_league_data()
    logging.info("Web scraping script execution finished.")