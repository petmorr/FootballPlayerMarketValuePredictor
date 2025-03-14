"""
web_scrape.py

This module uses Selenium and BeautifulSoup to scrape player statistics from FBref.
It processes the scraped HTML to extract a statistics table and saves the data as CSV files.
"""

import os
import time
from io import StringIO
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from logging_config import configure_logger

# =============================================================================
# Logger Configuration
# =============================================================================
logger = configure_logger("web_scrape", "web_scrape.log")

# =============================================================================
# Constants & Configuration
# =============================================================================
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


# =============================================================================
# Helper Functions
# =============================================================================
def configure_driver() -> webdriver.Chrome:
    """
    Configure and initialize the Chrome WebDriver in headless mode with a custom user-agent.

    Returns:
        webdriver.Chrome: Configured Chrome WebDriver.

    Raises:
        Exception: If the WebDriver cannot be configured.
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
        logger.info("WebDriver configured successfully.")
        return driver
    except Exception as e:
        logger.error(f"Failed to configure WebDriver: {e}")
        raise


def build_league_season_url(league_name: str, base_url: str, season: str) -> str:
    """
    Build the URL for scraping based on league, base URL, and season.

    Args:
        league_name (str): The league name.
        base_url (str): The base URL for the league.
        season (str): The season (e.g., '2023-2024').

    Returns:
        str: The constructed URL.
    """
    season_url = season.replace('-', '_')
    league_url = league_name.replace('-', '_')
    url = f"{base_url}{season}/stats/{season_url}-{league_url}-Stats"
    return url


def get_player_data_selenium(url: str, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
    """
    Scrape player statistics from a given URL using Selenium.

    The function searches for a table with the ID 'stats_standard'. If not found,
    it attempts to extract the table from HTML comments.

    Args:
        url (str): The URL to scrape.
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the scraped data, or None if scraping fails.
    """
    try:
        logger.info(f"Accessing URL: {url}")
        driver.get(url)
        WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
        time.sleep(2)  # Additional delay for dynamic content
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Attempt to find the primary table
        table = soup.find("table", id="stats_standard")
        table_html = str(table) if table else None

        # Check within a specific div if not found
        if not table_html:
            div_all = soup.find("div", id="all_stats_standard")
            if div_all:
                comments = div_all.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    if "stats_standard" in comment:
                        table_html = comment
                        break

        # Fallback: search all comments in the page
        if not table_html:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if "stats_standard" in comment:
                    table_html = comment
                    break

        if not table_html:
            logger.error(f"Could not find table 'stats_standard' in page source from {url}")
            return None

        df_list = pd.read_html(StringIO(table_html))
        if df_list:
            df = df_list[0]
            logger.info(f"Successfully scraped data from: {url}")
            return df
        else:
            logger.error(f"pd.read_html did not return any tables for {url}")
            return None
    except Exception as e:
        logger.error(f"Failed to scrape data from {url}: {e}")
        return None


def scrape_league_data() -> None:
    """
    Scrape player data for all specified leagues and seasons.

    For each league and season, build the URL, scrape the data,
    append metadata (league and season), and save as a CSV file.
    """
    driver = configure_driver()
    try:
        for league_name, base_url in LEAGUES.items():
            for season in SEASONS:
                try:
                    url = build_league_season_url(league_name, base_url, season)
                    logger.info(f"Scraping data for {league_name} - {season} from {url}")
                    df = get_player_data_selenium(url, driver)
                    if df is not None:
                        df["League"] = league_name
                        df["Season"] = season
                        file_name = os.path.join(OUTPUT_DIR, f"{league_name}_{season}_player_data.csv")
                        df.to_csv(file_name, index=False)
                        logger.info(f"Data successfully saved to {file_name}")
                    else:
                        logger.warning(f"No data found for {league_name} - {season}.")
                except Exception as inner_e:
                    logger.error(f"Error processing {league_name} - {season}: {inner_e}")
                time.sleep(2)
    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")
    finally:
        driver.quit()
        logger.info("Web scraping process completed and WebDriver closed.")


if __name__ == "__main__":
    logger.info("Starting web scraping process...")
    scrape_league_data()
    logger.info("Web scraping script execution finished.")