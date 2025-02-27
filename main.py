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
def ensure_module(module_name, package_name=None):
    try:
        __import__(module_name)
    except ImportError:
        pkg = package_name if package_name else module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(module_name)


ensure_module("bs4", "beautifulsoup4")

# Initialize logger and Flask application.
logger = configure_logger("web_portal", "web_portal.log")
app = Flask(__name__)
app.secret_key = "replace_with_your_secret_key"

# Directory containing the updated (preprocessed) datasets.
UPDATED_DATA_DIR = Path("data/updated")


# =============================================================================
# Helper function to run external commands using the same interpreter
# =============================================================================
def run_command(command):
    """
    Runs a command using the current Python interpreter.
    For example, instead of calling ["python", "script.py"],
    we call [sys.executable, "script.py"].
    Returns True if the command executes successfully.
    """
    full_command = [sys.executable] + command[1:] if command[0].lower() == "python" else command
    try:
        logger.info(f"Running command: {' '.join(full_command)}")
        subprocess.check_call(full_command)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command {' '.join(full_command)} failed: {e}")
        return False

# =============================================================================
# Helper Functions for Manual Transfer Value Input
# =============================================================================
def get_clean_basename(file_path: str) -> str:
    p = Path(file_path)
    return p.name.split('.')[0] if len(p.suffixes) > 1 else p.stem

def load_missing_transfer_values():
    missing_entries = []
    for file in UPDATED_DATA_DIR.glob("updated_*"):
        base_name = get_clean_basename(file.name)
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
    grouped = {}
    for entry in missing_entries:
        ds = entry["dataset"]
        grouped.setdefault(ds, []).append(entry)
    return grouped

def update_transfer_value_in_parquet(dataset, season, updates):
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
    try:
        response = requests.get("http://localhost:8000", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def start_local_api():
    if check_api_running():
        logger.info("API server already running. Skipping startup.")
        return True

    api_repo_dir = Path("transfermarkt-api")
    if not api_repo_dir.exists():
        logger.info("Cloning transfermarkt-api repository...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "git", "clone", "https://github.com/felipeall/transfermarkt-api.git"])
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    else:
        logger.info("transfermarkt-api repository already exists.")

    try:
        logger.info("Installing API dependencies via Poetry...")
        subprocess.check_call(["poetry", "install", "--no-root"], cwd=str(api_repo_dir))
        logger.info("Running poetry check to verify dependencies...")
        subprocess.check_call(["poetry", "check"], cwd=str(api_repo_dir))
    except Exception as e:
        logger.error(f"Error installing or verifying API dependencies: {e}")
        return False

    current_dir = os.getcwd()
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + current_dir

    try:
        logger.info("Starting API server using Poetry...")
        subprocess.Popen(["poetry", "run", "python", "app/main.py"], cwd=str(api_repo_dir))
        time.sleep(5)
        if not check_api_running():
            logger.error("API server did not start successfully.")
            return False
        logger.info("API server started successfully.")
        webbrowser.open("http://localhost:8000/")
        return True
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return False

@app.before_request
def run_preprocessing_once():
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
    if request.method == "POST":
        if not check_api_running():
            flash("API server is not running. Please start it manually.", "danger")
            return redirect(url_for("model_preprocessing"))
        try:
            if not run_command(["python", "./preprocessing/web_scrape.py"]):
                raise Exception("Web scraping failed.")
            if not run_command(["python", "./preprocessing/preprocessing.py"]):
                raise Exception("Preprocessing failed.")
            if not run_command(["python", "./preprocessing/player_value.py"]):
                raise Exception("Player value update failed.")
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
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        flash(f"Model creation triggered for {model_choice} (functionality not yet implemented).", "info")
        logger.info(f"Model creation triggered: {model_choice} (placeholder).")
        return redirect(url_for("model_creation"))
    return render_template("model_creation.html", title="Model Creation")

@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation():
    if request.method == "POST":
        flash("Model evaluation functionality is not yet implemented.", "info")
        logger.info("Model evaluation triggered (placeholder).")
        return redirect(url_for("model_evaluation"))
    return render_template("model_evaluation.html", title="Model Evaluation")

@app.route("/run_all", methods=["GET", "POST"])
def run_all():
    if request.method == "POST":
        steps = [
            (["python", "./preprocessing/web_scrape.py"], "Web scraping"),
            (["python", "./preprocessing/preprocessing.py"], "Preprocessing"),
            (["python", "./preprocessing/player_value.py"], "Player value update"),
            (["python", "./models/linear_regression_model.py"], "Linear Regression model training"),
            (["python", "./models/random_forest_model.py"], "Random Forest model training")
        ]
        for cmd, description in steps:
            if not run_command(cmd):
                flash(f"{description} failed.", "danger")
                logger.error(f"{description} failed.")
                return redirect(url_for("run_all"))
            else:
                flash(f"{description} completed successfully.", "success")
        flash("Full pipeline executed successfully.", "success")
        return redirect(url_for("index"))
    return render_template("run_all.html", title="Run Full Pipeline")


@app.route("/logs")
def view_logs():
    log_file = Path("web_portal.log")
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = f.read()
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            logs = f"Error reading log file: {e}"
    else:
        logs = "Log file not found."
    return render_template("logs.html", logs=logs)

if __name__ == "__main__":
    if not start_local_api():
        logger.error("Failed to start local API server. Some functionality may be unavailable.")
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)