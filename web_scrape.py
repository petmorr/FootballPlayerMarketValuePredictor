import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

import logging

# Constants
LEAGUES = {
    "Premier-League": "https://fbref.com/en/comps/9/",
    "La-Liga": "https://fbref.com/en/comps/12/",
    "Serie-A": "https://fbref.com/en/comps/11/",
    "Bundesliga": "https://fbref.com/en/comps/20/",
    "Ligue-1": "https://fbref.com/en/comps/13/",
}
SEASONS = ["2023-2024", "2022-2023", "2021-2022", "2020-2021", "2019-2020"]
OUTPUT_DIR = "data/scraped"
LOG_DIR = "./logging"
LOG_FILE = os.path.join(LOG_DIR, "web_scrape_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def configure_driver() -> webdriver.Chrome:
    """
    Configure and initialize the Chrome WebDriver with headless mode.

    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=chrome_options
        )
        logging.info("WebDriver configured successfully.")
        return driver
    except Exception as e:
        logging.error(f"Failed to configure WebDriver: {e}")
        raise

def get_player_data_selenium(url: str, driver: webdriver.Chrome) -> pd.DataFrame:
    """
    Scrape player data from the given URL using Selenium.

    Args:
        url (str): URL to scrape.
        driver (webdriver.Chrome): Selenium WebDriver instance.

    Returns:
        pd.DataFrame: DataFrame containing player data, or None if scraping fails.
    """
    try:
        logging.info(f"Accessing URL: {url}")
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.ID, "stats_standard")))
        df = pd.read_html(table.get_attribute("outerHTML"))[0]
        logging.info(f"Successfully scraped data from: {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to scrape data from {url}: {e}")
        return None

def scrape_league_data():
    """
    Scrape data for all leagues and seasons, saving results as CSV files.
    """
    driver = configure_driver()
    try:
        for league_name, base_url in LEAGUES.items():
            for season in SEASONS:
                url = f"{base_url}{season}/stats/{season.replace('-', '_')}-{league_name.replace('-', '_')}-Stats"
                logging.info(f"Scraping data for {league_name} - {season}")
                df = get_player_data_selenium(url, driver)
                if df is not None:
                    # Add league and season columns
                    df["League"] = league_name
                    df["Season"] = season

                    # Save scraped data to CSV
                    file_name = os.path.join(OUTPUT_DIR, f"{league_name}_{season}_player_data.csv")
                    df.to_csv(file_name, index=False)
                    logging.info(f"Data saved to {file_name}")
                else:
                    logging.warning(f"No data found for {league_name} - {season}.")
    except Exception as e:
        logging.error(f"An error occurred during scraping: {e}")
    finally:
        driver.quit()
        logging.info("Web scraping process completed and WebDriver closed.")

if __name__ == "__main__":
    logging.info("Starting web scraping process...")
    scrape_league_data()
    logging.info("Web scraping script execution finished.")
