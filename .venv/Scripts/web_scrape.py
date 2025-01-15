from io import StringIO
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Define leagues and seasons
leagues = {
    "Premier-League": "https://fbref.com/en/comps/9/",
    "La-Liga": "https://fbref.com/en/comps/12/",
    "Serie-A": "https://fbref.com/en/comps/11/",
    "Bundesliga": "https://fbref.com/en/comps/20/",
    "Ligue-1": "https://fbref.com/en/comps/13/",
}
seasons = ["2023-2024", "2022-2023", "2021-2022", "2020-2021", "2019-2020"]

# Create data folder
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Function to scrape player data using Selenium
def get_player_data_selenium(url, driver):
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.ID, "stats_standard")))
        html_content = StringIO(table.get_attribute('outerHTML'))
        df = pd.read_html(html_content)[0]
        return df
    except Exception as e:
        print(f"Failed to extract data from {url}: {e}")
        return None

# Main scraping function
def scrape_league_data():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        for league_name, base_url in leagues.items():
            for season in seasons:
                url = f"{base_url}{season}/stats/{season.replace('-', '_')}-{league_name.replace('-', '_')}-Stats"
                print(f"Scraping data from: {url}")
                df = get_player_data_selenium(url, driver)
                if df is not None:
                    # Add transfer values
                    df = add_transfermarkt_values(df, season)

                    # Save data to CSV
                    file_name = os.path.join(output_dir, f"{league_name}_{season}_player_data.csv")
                    df.to_csv(file_name, index=False)
                    print(f"Saved data to {file_name}")
                else:
                    print(f"Failed to scrape data for {league_name} {season}.")
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_league_data()