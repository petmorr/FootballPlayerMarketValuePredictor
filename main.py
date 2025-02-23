import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, flash

from logging_config import configure_logger

# Ensure BeautifulSoup is available; install it if necessary.
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
    from bs4 import BeautifulSoup, Comment

# Initialize logger and Flask application.
logger = configure_logger("web_portal", "web_portal.log")
app = Flask(__name__)
app.secret_key = "replace_with_your_secret_key"

# Directory containing the updated (preprocessed) datasets.
UPDATED_DATA_DIR = Path("data/updated")


# =============================================================================
# Helper Functions for Manual Transfer Value Input
# =============================================================================
def get_clean_basename(file_path: str) -> str:
    """
    Returns the base name of a file (without its extension).
    Example: "cleaned_Bundesliga_2019-2020.parquet" becomes "cleaned_Bundesliga_2019-2020".
    """
    p = Path(file_path)
    return p.name.split('.')[0] if len(p.suffixes) > 1 else p.stem


def load_missing_transfer_values():
    """
    Scans each updated dataset (Parquet or gzipped CSV) in UPDATED_DATA_DIR for rows
    where the "market value" column is missing.

    Returns:
        A list of dictionaries, each with details about one missing entry.
    """
    missing_entries = []
    for file in UPDATED_DATA_DIR.glob("updated_*"):
        base_name = get_clean_basename(file.name)  # e.g., "updated_Bundesliga_2019-2020"
        parts = base_name.split("_")
        league = parts[1] if len(parts) >= 3 else "Unknown"
        season = parts[2] if len(parts) >= 3 else "Unknown"
        try:
            if file.suffix == ".parquet":
                df = pd.read_parquet(file)
            elif file.name.endswith(".csv.gz"):
                df = pd.read_csv(file, compression="gzip", encoding="utf-8")
            else:
                continue
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

        # Normalize column names.
        df.columns = [col.lower() for col in df.columns]
        if "market value" not in df.columns:
            logger.warning(f"'market value' column not found in {file.name}; skipping.")
            continue

        missing_df = df[df["market value"].isna()]
        for _, row in missing_df.iterrows():
            entry = {
                "dataset": f"{league}_{season}",
                "season": season,
                "player": row.get("player", "Unknown"),
                "team": row.get("squad", "Unknown"),
                "closest_date": row.get("closest_date", "N/A"),
                "current_value": row.get("market value", "N/A")
            }
            missing_entries.append(entry)
    return missing_entries


def group_missing_entries(missing_entries):
    """
    Groups missing transfer value entries by dataset.

    Returns:
        A dictionary with dataset names as keys and lists of entries as values.
    """
    grouped = {}
    for entry in missing_entries:
        ds = entry["dataset"]
        grouped.setdefault(ds, []).append(entry)
    return grouped


def update_transfer_value_in_parquet(dataset, season, updates):
    """
    For the given dataset (e.g. "Bundesliga_2019-2020") and season, updates rows in the
    corresponding parquet file (in UPDATED_DATA_DIR) with manually entered transfer values.

    Returns:
        True if the update was successful; otherwise, False.
    """
    filename = f"updated_{dataset}.parquet"
    file_path = UPDATED_DATA_DIR / filename
    if not file_path.exists():
        logger.error(f"File {file_path} not found for updating transfer value.")
        return False
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False

    df.columns = [col.lower() for col in df.columns]
    for update in updates:
        player = update["player"]
        team = update["team"]
        new_value = update["manual_transfer_value"]
        mask = (df["player"].str.lower() == player.lower()) & (df["squad"].str.lower() == team.lower())
        if mask.sum() == 0:
            logger.warning(f"No matching row found for {player} in {team} in {file_path}.")
            continue
        df.loc[mask, "market value"] = new_value
        logger.info(f"Updated {player} ({team}) in {dataset} with value {new_value}.")
    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        logger.error(f"Error writing {file_path}: {e}")
        return False
    return True


# =============================================================================
# API Server Startup Functions
# =============================================================================
def check_api_running():
    """
    Checks if the Transfermarkt API server is running at http://localhost:8000.

    Returns:
        True if the server responds with status code 200; otherwise, False.
    """
    try:
        response = requests.get("http://localhost:8000", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def start_local_api():
    """
    Starts the Transfermarkt API server using Poetry (if not already running).
    This function assumes that the 'transfermarkt-api' repository exists as a submodule.

    Steps:
      1. If the repository does not exist, clone it.
      2. Install the API dependencies using "poetry install --no-root".
      3. Append the current directory to PYTHONPATH.
      4. Start the API server using "poetry run python app/main.py" in the background.
      5. Open the API URL in the default browser.

    Returns:
        True if the API server starts successfully; otherwise, False.
    """
    if check_api_running():
        logger.info("API server already running. Skipping startup.")
        return True

    api_repo_dir = Path("transfermarkt-api")
    if not api_repo_dir.exists():
        logger.info("Cloning transfermarkt-api repository...")
        try:
            subprocess.check_call(["git", "clone", "https://github.com/felipeall/transfermarkt-api.git"])
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    else:
        logger.info("transfermarkt-api repository already exists.")

    try:
        logger.info("Installing API dependencies via Poetry...")
        subprocess.check_call(["poetry", "install", "--no-root"], cwd=str(api_repo_dir))
    except Exception as e:
        logger.error(f"Error installing API dependencies: {e}")
        return False

    # Append the current directory to PYTHONPATH.
    current_dir = os.getcwd()
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + current_dir

    try:
        logger.info("Starting API server using Poetry...")
        subprocess.Popen(["poetry", "run", "python", "app/main.py"], cwd=str(api_repo_dir))
        time.sleep(5)  # Wait for the API server to start.
        if not check_api_running():
            logger.error("API server did not start successfully.")
            return False
        logger.info("API server started successfully.")
        webbrowser.open("http://localhost:8000/")
        return True
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return False


# =============================================================================
# Automatic Initialization via Flask
# =============================================================================
@app.before_request
def run_preprocessing_once():
    """
    Runs before the first Flask request to ensure the API server is running.
    """
    if not app.config.get("PREPROCESSING_DONE"):
        if not start_local_api():
            flash("Failed to start local API server. Some functionality may be unavailable.", "danger")
        else:
            flash("Local API server started and backend initialized.", "success")
        app.config["PREPROCESSING_DONE"] = True


# =============================================================================
# Flask Routes
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html", title="Home")


@app.route("/model_preprocessing", methods=["GET", "POST"])
def model_preprocessing():
    """
    Combines web scraping, preprocessing, and player value extraction.
    Checks that the API server is running before executing the scripts.
    """
    if request.method == "POST":
        if not check_api_running():
            flash("API server is not running. Please start it manually.", "danger")
            return redirect(url_for("model_preprocessing"))
        try:
            subprocess.check_call(["python", "./preprocessing/web_scrape.py"])
            subprocess.check_call(["python", "./preprocessing/preprocessing.py"])
            subprocess.check_call(["python", "./preprocessing/player_value.py"])
            flash("Preprocessing completed successfully.", "success")
        except Exception as e:
            flash("Error during preprocessing.", "danger")
            logger.error(f"Error during preprocessing: {e}")
            return redirect(url_for("model_preprocessing"))
        missing_entries = load_missing_transfer_values()
        if missing_entries:
            flash("Some players are missing transfer values. Please provide manual inputs.", "warning")
            logger.info("Missing transfer values detected; redirecting to manual input.")
            return redirect(url_for("manual_input"))
        else:
            flash("Model preprocessing completed successfully. No missing transfer values found.", "success")
            return redirect(url_for("index"))
    return render_template("model_preprocessing.html", title="Model Preprocessing")


@app.route("/manual_input", methods=["GET", "POST"])
def manual_input():
    """
    Renders a form for manual input of missing transfer values.
    """
    if request.method == "POST":
        players = request.form.getlist("player")
        teams = request.form.getlist("team")
        datasets = request.form.getlist("dataset")
        seasons = request.form.getlist("season")
        closest_dates = request.form.getlist("closest_date")
        manual_values = request.form.getlist("manual_value")
        updates = []
        for ds, ss, p, t, cd, mv in zip(datasets, seasons, players, teams, closest_dates, manual_values):
            if mv.strip() == "":
                continue
            updates.append({
                "dataset": ds,
                "season": ss,
                "player": p,
                "team": t,
                "closest_date": cd,
                "manual_transfer_value": float(mv)
            })
        grouped_updates = {}
        for update in updates:
            ds = update["dataset"]
            grouped_updates.setdefault(ds, []).append(update)
        for ds, upd_list in grouped_updates.items():
            if not update_transfer_value_in_parquet(ds, upd_list[0]["season"], upd_list):
                flash(f"Failed to update dataset {ds}.", "danger")
            else:
                flash(f"Successfully updated dataset {ds}.", "success")
        return redirect(url_for("manual_input"))
    else:
        missing_entries = load_missing_transfer_values()
        grouped_missing = group_missing_entries(missing_entries)
        return render_template("manual_input.html", grouped_missing=grouped_missing,
                               title="Manual Transfer Value Input")


@app.route("/model_creation", methods=["GET", "POST"])
def model_creation():
    """
    Placeholder route for model creation.
    """
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        flash(f"Model creation triggered for {model_choice} (functionality not yet implemented).", "info")
        logger.info(f"Model creation triggered: {model_choice} (placeholder).")
        return redirect(url_for("model_creation"))
    return render_template("model_creation.html", title="Model Creation")


@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation():
    """
    Placeholder route for model evaluation.
    """
    if request.method == "POST":
        flash("Model evaluation functionality is not yet implemented.", "info")
        logger.info("Model evaluation triggered (placeholder).")
        return redirect(url_for("model_evaluation"))
    return render_template("model_evaluation.html", title="Model Evaluation")


@app.route("/run_all", methods=["GET", "POST"])
def run_all():
    """
    Placeholder route for running the full pipeline.
    """
    if request.method == "POST":
        flash("Full pipeline run functionality is not yet implemented.", "info")
        logger.info("Full pipeline run triggered (placeholder).")
        return redirect(url_for("index"))
    return render_template("run_all.html", title="Run Full Pipeline")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Attempt to start the local API server.
    if not start_local_api():
        logger.error("Failed to start local API server. Some functionality may be unavailable.")
    # Start the Flask web server.
    app.run(debug=True)