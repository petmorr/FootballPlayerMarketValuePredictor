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


# ------------------------------------------------------------------------------
# Module Setup
# ------------------------------------------------------------------------------
def ensure_module(module_name: str, package_name: str = None) -> None:
    """
    Ensure that the specified module is installed. If missing, install it via pip.

    Args:
        module_name (str): Name of the module to check.
        package_name (str, optional): Name of the package to install if different.
                                    Defaults to None.
    """
    try:
        __import__(module_name)
    except ImportError:
        pkg = package_name if package_name else module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(module_name)


ensure_module("bs4", "beautifulsoup4")

# ------------------------------------------------------------------------------
# Application Setup
# ------------------------------------------------------------------------------
logger = configure_logger("web_portal", "web_portal.log")
app = Flask(__name__)
app.secret_key = "replace_with_your_secret_key"


# ------------------------------------------------------------------------------
# API Server Functions
# ------------------------------------------------------------------------------
def check_api_running() -> bool:
    """
    Check if the local API server is running.

    Returns:
        bool: True if API responds with status code 200, otherwise False.
    """
    try:
        response = requests.get("http://localhost:8000", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def start_local_api() -> bool:
    """
    Start the local API server if it is not already running.

    This function clones the repository if needed, installs dependencies using Poetry,
    and starts the API server. It then verifies that the server is running.

    Returns:
        bool: True if the API server is running, otherwise False.
    """
    if check_api_running():
        logger.info("API server already running. Skipping startup.")
        return True

    api_repo_dir = Path("transfermarkt-api")
    if not api_repo_dir.exists():
        logger.info("Cloning transfermarkt-api repository.")
        try:
            subprocess.check_call(["git", "clone", "https://github.com/felipeall/transfermarkt-api.git"])
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    else:
        logger.info("transfermarkt-api repository already exists.")

    try:
        logger.info("Installing API dependencies via Poetry.")
        subprocess.check_call(["poetry", "install", "--no-root"], cwd=str(api_repo_dir))
        logger.info("Verifying dependencies with Poetry.")
        subprocess.check_call(["poetry", "check"], cwd=str(api_repo_dir))
    except Exception as e:
        logger.error(f"Error installing or verifying API dependencies: {e}")
        return False

    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + str(os.getcwd())
    try:
        logger.info("Starting API server using Poetry.")
        subprocess.Popen(["poetry", "run", "python", "app/main.py"], cwd=str(api_repo_dir))
        time.sleep(5)
        if not check_api_running():
            logger.error("API server did not start successfully.")
            return False
        logger.info("API server started successfully.")
        return True
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return False


@app.before_request
def run_preprocessing_once() -> None:
    """
    Flask hook to ensure the local API server starts only once.
    """
    if not app.config.get("PREPROCESSING_DONE"):
        if not start_local_api():
            flash("Failed to start local API server. Some functionality may be unavailable.", "danger")
        else:
            flash("Local API server started and backend initialized.", "success")
        app.config["PREPROCESSING_DONE"] = True


def run_command(command: list) -> bool:
    """
    Execute a system command.

    If the command starts with 'python', it will run using the current interpreter
    with its working directory set to the script's parent folder.

    Args:
        command (list): Command and its arguments as a list.

    Returns:
        bool: True if the command executes successfully, otherwise False.
    """
    cwd = None
    if command and command[0].lower() == "python" and len(command) > 1:
        script_path = Path(command[1]).resolve()
        cwd = str(script_path.parent)
        new_script = script_path.name
        full_command = [sys.executable, new_script] + command[2:]
    else:
        full_command = command

    try:
        logger.info(f"Running command: {' '.join(map(str, full_command))} (cwd={cwd})")
        subprocess.check_call(full_command, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command {' '.join(map(str, full_command))} failed: {e}")
        return False


# ------------------------------------------------------------------------------
# Route Handlers
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    """
    Render the home page.
    """
    return render_template("index.html", title="Home")


@app.route("/logs")
def view_logs():
    """
    Render the logs page by reading various log files.

    Returns:
        Rendered HTML for the logs page.
    """
    log_files = {
        "Web Portal": Path("logging/web_portal.log"),
        "Web Scrape": Path("preprocessing/logging/web_scrape.log"),
        "Preprocessing": Path("preprocessing/logging/preprocessing.log"),
        "Player Value": Path("preprocessing/logging/player_value.log"),
        "Models": Path("models/logging/model_utils.log"),
    }
    logs_content = {}
    for log_name, file_path in log_files.items():
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    logs_content[log_name] = f.read()
            except UnicodeDecodeError:
                logger.error(f"UTF-8 decoding failed for {file_path}, trying fallback encoding.")
                try:
                    with open(file_path, "r", encoding="latin1") as f:
                        logs_content[log_name] = f.read()
                except Exception as e2:
                    logs_content[log_name] = f"Error reading log file: {e2}"
            except Exception as e:
                logs_content[log_name] = f"Error reading log file: {e}"
        else:
            logs_content[log_name] = "Log file not found."
    return render_template("logs.html", logs=logs_content)


@app.route("/model_preprocessing", methods=["GET", "POST"])
def model_preprocessing():
    """
    Handle model preprocessing.

    On POST: runs web scraping, data preprocessing, and player value update scripts.
    On GET: renders the preprocessing page.
    """
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


# ------------------------------------------------------------------------------
# Helper Functions for Data Processing
# ------------------------------------------------------------------------------
def get_clean_basename(file_path: str) -> str:
    """
    Extract a clean basename from a file path by removing extra extensions.

    Args:
        file_path (str): The file path.

    Returns:
        str: Clean basename.
    """
    p = Path(file_path)
    return p.name.split('.')[0] if len(p.suffixes) > 1 else p.stem


def load_missing_transfer_values() -> list:
    """
    Load entries with missing transfer values from all subdirectories in data/updated.

    Returns:
        list: List of dictionaries with details of missing entries.
    """
    missing_entries = []
    updated_dir = Path("data/updated")
    # Recursively search for all .parquet files in subdirectories
    for file in updated_dir.rglob("updated_*.parquet"):
        try:
            df = pd.read_parquet(file)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Determine the market value column
        if "market value" in df.columns:
            mv_col = "market value"
        elif "market_value" in df.columns:
            mv_col = "market_value"
        else:
            candidate_cols = [c for c in df.columns if "market" in c and "value" in c]
            if candidate_cols:
                mv_col = candidate_cols[0]
            else:
                logger.warning(f"No market value column found in {file.name}; skipping.")
                continue

        # Log unique values for debugging purposes
        unique_vals = df[mv_col].unique()
        logger.info(f"Unique values in '{mv_col}' for {file.name}: {unique_vals}")

        # Helper to determine if a value is missing
        def is_missing(val):
            if pd.isna(val):
                return True
            if isinstance(val, str):
                return val.strip().lower() in ["", "n/a", "na", "null", "none", "nan", "-"]
            return False

        missing_df = df[df[mv_col].apply(is_missing)]
        logger.info(f"Found {missing_df.shape[0]} missing market value entries in {file.name}")

        # Extract league and season from the filename; e.g. updated_Premier-League_2022-2023
        base_name = file.stem  # gets "updated_Premier-League_2022-2023"
        parts = base_name.split("_")
        league = parts[1] if len(parts) >= 3 else "Unknown"
        season = parts[2] if len(parts) >= 3 else "Unknown"

        # Optionally, include the feature engineering variant (i.e. subdirectory name)
        fe_variant = file.parent.name  # e.g. enhanced_feature_engineering

        for _, row in missing_df.iterrows():
            entry = {
                "dataset": f"{league}_{season}_{fe_variant}",
                "season": season,
                "player": row.get("player", "Unknown"),
                "team": row.get("squad", "Unknown"),
                "closest_date": row.get("closest_date", "N/A"),
                "current_value": row.get(mv_col, "N/A")
            }
            missing_entries.append(entry)
    return missing_entries


def group_missing_entries(missing_entries: list) -> dict:
    """
    Group missing entries by dataset.

    Args:
        missing_entries (list): List of missing transfer value entries.

    Returns:
        dict: Grouped entries keyed by dataset.
    """
    grouped = {}
    for entry in missing_entries:
        ds = entry["dataset"]
        grouped.setdefault(ds, []).append(entry)
    return grouped


def update_transfer_value_in_parquet(dataset: str, season: str, updates: list) -> bool:
    """
    Update transfer values in a parquet file for a given dataset.

    Args:
        dataset (str): Dataset identifier.
        season (str): Season identifier.
        updates (list): List of update dictionaries.

    Returns:
        bool: True if updates were successful, otherwise False.
    """
    updated_dir = Path("data/updated")
    filename = f"updated_{dataset}.parquet"
    file_path = updated_dir / filename
    if not file_path.exists():
        logger.error(f"File {file_path} not found for updating transfer value.")
        return False
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False
    df.columns = [col.lower() for col in df.columns]
    for upd in updates:
        player = upd["player"]
        team = upd["team"]
        new_value = upd["manual_transfer_value"]
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


# ------------------------------------------------------------------------------
# Additional Route Handlers
# ------------------------------------------------------------------------------
@app.route("/manual_input", methods=["GET", "POST"])
def manual_input():
    """
    Handle manual input for missing transfer values.

    On POST: process submitted manual updates.
    On GET: display missing entries for user input.
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
        for upd in updates:
            ds = upd["dataset"]
            grouped_updates.setdefault(ds, []).append(upd)
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
    Handle model creation requests.

    On POST: run the selected model training script.
    """
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        if model_choice == "LinearRegression":
            if run_command(["python", "./models/linear_regression_model.py"]):
                flash("Linear Regression model created successfully.", "success")
                logger.info("Linear Regression model creation succeeded.")
            else:
                flash("Linear Regression model creation failed.", "danger")
                logger.error("Linear Regression model creation failed.")
        elif model_choice == "RandomForest":
            if run_command(["python", "./models/random_forest_model.py"]):
                flash("Random Forest model created successfully.", "success")
                logger.info("Random Forest model creation succeeded.")
            else:
                flash("Random Forest model creation failed.", "danger")
                logger.error("Random Forest model creation failed.")
        elif model_choice == "XGBoost":
            if run_command(["python", "./models/xgboost_model.py"]):
                flash("XGBoost model created successfully.", "success")
                logger.info("XGBoost model creation succeeded.")
            else:
                flash("XGBoost model creation failed.", "danger")
                logger.error("XGBoost model creation failed.")
        else:
            flash("Unknown model choice.", "warning")
            logger.warning("Unknown model choice selected in model_creation.")
        return redirect(url_for("model_creation"))
    return render_template("model_creation.html", title="Model Creation")


def safe_relative_path(file_path: Path, base: Path) -> str:
    """
    Return the relative path of file_path with respect to base, if possible.

    Args:
        file_path (Path): The file path.
        base (Path): The base directory.

    Returns:
        str: Relative path or absolute path if relative conversion fails.
    """
    try:
        return str(file_path.relative_to(base))
    except ValueError:
        return str(file_path)


def parse_predicted_file_details(file_path: Path, base: Path) -> dict:
    """
    Parse details from a predicted file's path.

    Args:
        file_path (Path): The predicted file path.
        base (Path): The base directory for relative paths.

    Returns:
        dict: Dictionary containing model name, feature variant, league/season, and file link.
    """
    parts = file_path.parts
    model_name = "Unknown Model"
    if "linear_regression" in parts:
        model_name = "Linear Regression"
    elif "random_forest" in parts:
        model_name = "Random Forest"
    elif "xgboost" in parts:
        model_name = "XGBoost"
    fe_variant = "Unknown"
    # Include xgboost in the check so we capture the feature engineering variant if present
    for i, p in enumerate(parts):
        if p in ("linear_regression", "random_forest", "xgboost") and i + 1 < len(parts):
            fe_variant = parts[i + 1]
            break
    filename = file_path.name
    if filename.startswith("predicted_updated_"):
        filename = filename[len("predicted_updated_"):]
    if filename.endswith(".parquet"):
        filename = filename[:-8]
    league_season = filename
    link = safe_relative_path(file_path, base)
    return {"model_name": model_name, "fe_variant": fe_variant, "league_season": league_season, "link": link}


@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation():
    """
    Display performance metrics and predictions; support searching within prediction data.
    """
    lr_metrics_file = Path("models/results/performance_metrics_linear_regression.csv")
    rf_metrics_file = Path("models/results/performance_metrics_random_forest.csv")
    xgboost_metrics_file = Path("models/results/performance_metrics_xgboost.csv")
    metrics = {}
    if lr_metrics_file.exists():
        try:
            lr_df = pd.read_csv(lr_metrics_file)
            metrics["Linear Regression"] = lr_df.to_html(classes="table table-striped table-bordered", index=False)
        except Exception as e:
            logger.error(f"Error reading linear regression metrics: {e}")
            metrics["Linear Regression"] = f"Error reading metrics: {e}"
    else:
        metrics["Linear Regression"] = "Metrics file not found."
    if rf_metrics_file.exists():
        try:
            rf_df = pd.read_csv(rf_metrics_file)
            metrics["Random Forest"] = rf_df.to_html(classes="table table-striped table-bordered", index=False)
        except Exception as e:
            logger.error(f"Error reading random forest metrics: {e}")
            metrics["Random Forest"] = f"Error reading metrics: {e}"
    else:
        metrics["Random Forest"] = "Metrics file not found."
    if xgboost_metrics_file.exists():
        try:
            xgboost_df = pd.read_csv(xgboost_metrics_file)
            metrics["XGBoost"] = xgboost_df.to_html(classes="table table-striped table-bordered", index=False)
        except Exception as e:
            logger.error(f"Error reading XGBoost metrics: {e}")
            metrics["XGBoost"] = f"Error reading metrics: {e}"
    else:
        metrics["XGBoost"] = "Metrics file not found."

    predictions = {"Linear Regression": {}, "Random Forest": {}, "XGBoost": {}}

    def insert_into_tree(details: dict) -> None:
        """
        Insert predicted file details into the predictions dictionary.
        """
        m = details["model_name"]
        fe = details["fe_variant"]
        ls = details["league_season"]
        link = details["link"]
        if m not in predictions:
            predictions[m] = {}
        if fe not in predictions[m]:
            predictions[m][fe] = {}
        predictions[m][fe][ls] = link

    base = Path.cwd()
    lr_pred_dir = Path("data/predictions/linear_regression")
    rf_pred_dir = Path("data/predictions/random_forest")
    xgboost_pred_dir = Path("data/predictions/xgboost")
    if lr_pred_dir.exists():
        for f in lr_pred_dir.rglob("*"):
            if f.is_file():
                info = parse_predicted_file_details(f, base)
                insert_into_tree(info)
    if rf_pred_dir.exists():
        for f in rf_pred_dir.rglob("*"):
            if f.is_file():
                info = parse_predicted_file_details(f, base)
                insert_into_tree(info)
    if xgboost_pred_dir.exists():
        for f in xgboost_pred_dir.rglob("*"):
            if f.is_file():
                info = parse_predicted_file_details(f, base)
                insert_into_tree(info)

    search_results = None
    search_params = {}
    if request.method == "POST" and "search_term" in request.form:
        selected_model = request.form.get("model")
        selected_fe = request.form.get("fe_variant")
        selected_league = request.form.get("league")
        selected_season = request.form.get("season")
        view_type = request.form.get("view_type")
        search_term = request.form.get("search_term", "").strip().lower()
        search_params = {
            "model": selected_model,
            "fe_variant": selected_fe,
            "league": selected_league,
            "season": selected_season,
            "view_type": view_type,
            "search_term": search_term
        }
        if view_type == "Raw":
            file_path = Path("data/updated") / selected_fe / f"updated_{selected_league}_{selected_season}.parquet"
        else:
            if selected_model == "Linear Regression":
                file_path = Path(
                    "data/predictions/linear_regression") / selected_fe / f"predicted_updated_{selected_league}_{selected_season}.parquet"
            elif selected_model == "Random Forest":
                file_path = Path(
                    "data/predictions/random_forest") / selected_fe / f"predicted_updated_{selected_league}_{selected_season}.parquet"
            # Adjust condition to cover both naming formats for XGBoost
            elif selected_model in ("XGBoost", "XG Boost"):
                file_path = Path(
                    "data/predictions/xgboost") / selected_fe / f"predicted_updated_{selected_league}_{selected_season}.parquet"
                if not file_path.exists():
                    file_path = Path(
                        "data/predictions/xgboost") / f"predicted_updated_{selected_league}_{selected_season}.parquet"
            else:
                file_path = None
        if file_path is None or not file_path.exists():
            flash(f"Data file not found: {file_path}", "danger")
            search_results = f"Data file not found: {file_path}"
        else:
            try:
                df = pd.read_parquet(file_path)
                if search_term:
                    df = df[df["player"].str.lower().str.contains(search_term)]
                cols = ["player", "position", "squad", "Market Value", "predicted_market_value"]
                keep_cols = [c for c in cols if c in df.columns]
                df = df[keep_cols]
                if "predicted_market_value" in df.columns:
                    df.rename(columns={"predicted_market_value": "Predicted Price"}, inplace=True)
                search_results = df.to_html(classes="table table-striped table-bordered", index=False)
            except Exception as e:
                flash(f"Error loading data: {e}", "danger")
                search_results = f"Error loading data: {e}"
    return render_template(
        "model_evaluation.html",
        title="Model Evaluation",
        metrics=metrics,
        predictions=predictions,
        search_results=search_results,
        search_params=search_params
    )


@app.route("/run_all", methods=["GET", "POST"])
def run_all():
    """
    Execute the full pipeline: web scraping, preprocessing, player value update,
    and model training.
    """
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


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if not start_local_api():
        logger.error("Failed to start local API server. Some functionality may be unavailable.")
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000/", new=0)
    app.run(debug=True)