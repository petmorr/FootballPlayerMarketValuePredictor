"""
Scrapes player statistics from FBref via Selenium + BeautifulSoup, extracting
the main stats table and saving to CSV.
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

logger = configure_logger("web_scrape", "web_scrape.log")

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


def configure_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    logger.info("WebDriver configured.")
    return driver


def build_league_season_url(league_name: str, base_url: str, season: str) -> str:
    season_url = season.replace('-', '_')
    league_url = league_name.replace('-', '_')
    return f"{base_url}{season}/stats/{season_url}-{league_url}-Stats"


def get_player_data_selenium(url: str, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Accessing URL: {url}")
        driver.get(url)
        WebDriverWait(driver, 10).until(lambda d: d.execute_script("return document.readyState") == "complete")
        time.sleep(2)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="stats_standard")
        table_html = str(table) if table else None

        if not table_html:
            div_all = soup.find("div", id="all_stats_standard")
            if div_all:
                comments = div_all.find_all(string=lambda text: isinstance(text, Comment))
                for c in comments:
                    if "stats_standard" in c:
                        table_html = c
                        break

        if not table_html:
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if "stats_standard" in c:
                    table_html = c
                    break

        if not table_html:
            logger.error(f"No table 'stats_standard' in {url}")
            return None

        df_list = pd.read_html(StringIO(table_html))
        if df_list:
            logger.info(f"Scraped data from {url}")
            return df_list[0]
        logger.error(f"pd.read_html returned no tables for {url}")
        return None
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return None


def scrape_league_data() -> None:
    driver = configure_driver()
    try:
        for league_name, base_url in LEAGUES.items():
            for season in SEASONS:
                try:
                    url = build_league_season_url(league_name, base_url, season)
                    logger.info(f"Scraping {league_name} - {season} from {url}")
                    df = get_player_data_selenium(url, driver)
                    if df is not None:
                        df["League"] = league_name
                        df["Season"] = season
                        file_name = os.path.join(OUTPUT_DIR, f"{league_name}_{season}_player_data.csv")
                        df.to_csv(file_name, index=False)
                        logger.info(f"Saved data to {file_name}")
                    else:
                        logger.warning(f"No data for {league_name} - {season}.")
                except Exception as inner_e:
                    logger.error(f"Error in {league_name} - {season}: {inner_e}")
                time.sleep(2)
    finally:
        driver.quit()
        logger.info("Web scraping completed. WebDriver closed.")


if __name__ == "__main__":
    logger.info("Starting web scraping process.")
    scrape_league_data()
    logger.info("Web scraping script finished.")
