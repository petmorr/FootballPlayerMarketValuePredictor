import base64
import os
import subprocess
import sys
import time
import webbrowser
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, flash

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logging_config import configure_logger

def ensure_module(module_name: str, package_name: str = None) -> None:
    """Ensures a Python module is installed."""
    try:
        __import__(module_name)
    except ImportError:
        pkg = package_name if package_name else module_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        __import__(module_name)


# Ensure required modules
ensure_module("bs4", "beautifulsoup4")
# Weird fix, wouldn't run without specifically checking this

logger = configure_logger("web_portal", "web_portal.log")
app = Flask(__name__)
app.secret_key = "replace_with_secret_key"


# -----------------------------------------------------------------------------
# API Server Checks & Startup
# -----------------------------------------------------------------------------
def check_api_running() -> bool:
    """Checks if the local API server is running."""
    try:
        response = requests.get("http://localhost:8000", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def start_local_api() -> bool:
    """Starts the local API server if not already running."""
    if check_api_running():
        logger.info("API server already running.")
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
        logger.info("transfermarkt-api repository exists.")

    try:
        logger.info("Installing API dependencies via Poetry.")
        subprocess.check_call(["poetry", "install", "--no-root"], cwd=str(api_repo_dir))
        logger.info("Verifying dependencies with Poetry.")
        subprocess.check_call(["poetry", "check"], cwd=str(api_repo_dir))
    except Exception as e:
        logger.error(f"Error installing or verifying dependencies: {e}")
        return False

    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + str(os.getcwd())
    try:
        logger.info("Starting API server via Poetry.")
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
    """Ensures the API server is started once before handling requests."""
    if not app.config.get("PREPROCESSING_DONE"):
        if not start_local_api():
            flash("Failed to start local API server.", "danger")
        else:
            flash("Local API server started.", "success")
        app.config["PREPROCESSING_DONE"] = True


# -----------------------------------------------------------------------------
# Command Execution & Log Reading Helpers
# -----------------------------------------------------------------------------
def run_command(command: list) -> bool:
    """Runs a command in a subprocess."""
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
        logger.error(f"Command failed: {e}")
        return False

def read_log_file(file_path: Path) -> str:
    """Reads a log file with latin1 encoding."""
    if not file_path.exists():
        return "Log file not found."
    try:
        with open(file_path, "r", encoding="latin1") as f:
            return f.read()
    except Exception as e:
        return f"Error reading log file: {e}"


# -----------------------------------------------------------------------------
# Missing Transfer Values & Updating Parquet
# -----------------------------------------------------------------------------
def load_missing_transfer_values() -> list:
    """Loads missing market value entries from parquet files."""
    missing_entries = []
    updated_dir = Path("data/updated")

    # Search for feature engineering variant directories
    fe_variants = [d for d in updated_dir.iterdir() if d.is_dir()]

    for fe_dir in fe_variants:
        fe_variant = fe_dir.name
        for file in fe_dir.rglob("updated_*.parquet"):
            try:
                df = pd.read_parquet(file)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue
            df.columns = [c.lower() for c in df.columns]

            if "market value" in df.columns:
                mv_col = "market value"
            elif "market_value" in df.columns:
                mv_col = "market_value"
            else:
                candidate_cols = [c for c in df.columns if "market" in c and "value" in c]
                if candidate_cols:
                    mv_col = candidate_cols[0]
                else:
                    logger.warning(f"No market value column found in {file.name}.")
                    continue

            def is_missing(val):
                if pd.isna(val):
                    return True
                if isinstance(val, str):
                    return val.strip().lower() in ["", "n/a", "na", "null", "none", "nan", "-"]
                return False

            missing_df = df[df[mv_col].apply(is_missing)]

            # Extract league and season from the filename
            base_name = file.stem
            parts = base_name.split("_")
            if len(parts) >= 3:
                league = parts[1]
                season = parts[2]

                # Store dataset as league_season_fe_variant
                dataset = f"{league}_{season}_{fe_variant}"

                for _, row in missing_df.iterrows():
                    entry = {
                        "dataset": dataset,
                        "season": season,
                        "player": row.get("player", "Unknown"),
                        "team": row.get("squad", "Unknown"),
                        "closest_date": row.get("closest_date", "N/A"),
                        "current_value": row.get(mv_col, "N/A"),
                        "fe_variant": fe_variant
                    }
                    missing_entries.append(entry)
            else:
                logger.warning(f"Invalid filename format: {file.name}")

    return missing_entries

def group_missing_entries(missing_entries: list) -> dict:
    """Groups missing entries by dataset."""
    grouped = {}
    for entry in missing_entries:
        ds = entry["dataset"]
        grouped.setdefault(ds, []).append(entry)
    return grouped


def update_transfer_value_in_parquet(dataset: str, season: str, updates: list) -> bool:
    """Updates the parquet files with new market values."""
    updated_dir = Path("data/updated")

    # Split dataset to extract league, season, and feature engineering variant
    parts = dataset.split('_')
    if len(parts) >= 3:
        # Extract league, season, and feature engineering variant
        league = parts[0]
        season = parts[1]
        fe_variant = '_'.join(parts[2:])  # Properly handle multi-part feature engineering names

        # Construct the correct path with the feature engineering variant as a directory
        file_path = updated_dir / fe_variant / f"updated_{league}_{season}.parquet"
    else:
        # Legacy path handling (fallback)
        file_path = updated_dir / f"updated_{dataset}.parquet"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
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
            logger.warning(f"No matching row for {player} in {team}.")
            continue
        df.loc[mask, "market value"] = new_value
        logger.info(f"Updated {player} in {dataset} with {new_value}.")

    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        logger.error(f"Error writing {file_path}: {e}")
        return False
    return True


# -----------------------------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index() -> str:
    return render_template("index.html", title="Home")

@app.route("/logs")
def view_logs() -> str:
    base_dir = Path(__file__).parent
    log_files = {
        "Web Portal": base_dir / "logging" / "web_portal.log",
        "Web Scrape": base_dir / "preprocessing" / "logging" / "web_scrape.log",
        "Preprocessing": base_dir / "preprocessing" / "logging" / "preprocessing.log",
        "Player Value": base_dir / "preprocessing" / "logging" / "player_value.log",
        "Models": base_dir / "models" / "logging" / "model_utils.log"
    }
    logs_content = {name: read_log_file(path) for name, path in log_files.items()}
    return render_template("logs.html", logs=logs_content)

@app.route("/model_preprocessing", methods=["GET", "POST"])
def model_preprocessing() -> str:
    if request.method == "POST":
        if not check_api_running():
            flash("API server is not running.", "danger")
            return redirect(url_for("model_preprocessing"))
        try:
            if not run_command(["python", "./preprocessing/web_scrape.py"]):
                raise Exception("Web scraping failed.")
            if not run_command(["python", "./preprocessing/preprocessing.py"]):
                raise Exception("Preprocessing failed.")
            if not run_command(["python", "./preprocessing/player_value.py"]):
                raise Exception("Player value update failed.")
            flash("Preprocessing completed.", "success")
        except Exception as e:
            flash("Error during preprocessing.", "danger")
            logger.error(f"Error: {e}")
            return redirect(url_for("model_preprocessing"))
        missing_entries = load_missing_transfer_values()
        if missing_entries:
            flash("Some players are missing transfer values.", "warning")
            return redirect(url_for("manual_input"))
        else:
            flash("No missing transfer values.", "success")
            return redirect(url_for("index"))
    return render_template("model_preprocessing.html", title="Model Preprocessing")

@app.route("/manual_input", methods=["GET", "POST"])
def manual_input() -> str:
    if request.method == "POST":
        players = request.form.getlist("player")
        teams = request.form.getlist("team")
        datasets = request.form.getlist("dataset")
        seasons = request.form.getlist("season")
        closest_dates = request.form.getlist("closest_date")
        manual_values = request.form.getlist("manual_value")
        updates = []
        for ds, ss, p, t, cd, mv in zip(datasets, seasons, players, teams, closest_dates, manual_values):
            if mv.strip():
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
                flash(f"Failed to update {ds}.", "danger")
            else:
                flash(f"Updated {ds}.", "success")
        return redirect(url_for("manual_input"))
    else:
        missing_entries = load_missing_transfer_values()
        group_missing = group_missing_entries(missing_entries)
        return render_template("manual_input.html", grouped_missing=group_missing, title="Manual Transfer Value Input")

@app.route("/model_creation", methods=["GET", "POST"])
def model_creation() -> str:
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        if model_choice == "LinearRegression":
            if run_command(["python", "./models/linear_regression_model.py"]):
                flash("Linear Regression created.", "success")
            else:
                flash("Linear Regression failed.", "danger")
        elif model_choice == "RandomForest":
            if run_command(["python", "./models/random_forest_model.py"]):
                flash("Random Forest created.", "success")
            else:
                flash("Random Forest failed.", "danger")
        elif model_choice == "XGBoost":
            if run_command(["python", "./models/xgboost_model.py"]):
                flash("XGBoost created.", "success")
            else:
                flash("XGBoost failed.", "danger")
        else:
            flash("Unknown model choice.", "warning")
        return redirect(url_for("model_creation"))
    return render_template("model_creation.html", title="Model Creation")


# -----------------------------------------------------------------------------
# Shared Helper for Predicted Files
# -----------------------------------------------------------------------------
def safe_relative_path(file_path: Path, base: Path) -> str:
    """Computes relative path if possible."""
    try:
        return str(file_path.relative_to(base))
    except ValueError:
        return str(file_path)

def parse_predicted_file_details(file_path: Path, base: Path) -> dict:
    """Extracts details from a predicted file's path."""
    parts = file_path.parts
    model_name = "Unknown Model"
    if "linear_regression" in parts:
        model_name = "Linear Regression"
    elif "random_forest" in parts:
        model_name = "Random Forest"
    elif "xgboost" in parts:
        model_name = "XGBoost"

    fe_variant = "Unknown"
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
    return {
        "model_name": model_name,
        "fe_variant": fe_variant,
        "league_season": league_season,
        "link": link
    }


# -----------------------------------------------------------------------------
# Plotting Helpers
# -----------------------------------------------------------------------------
def df_to_bar_base64_png(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[x_col], df[y_col], color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def df_to_dist_base64_png(series: pd.Series, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series.dropna(), bins=15, color='orchid', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def df_to_scatter_base64_png(df: pd.DataFrame, actual_col: str, pred_col: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[actual_col], df[pred_col], alpha=0.6, color='teal')
    max_val = max(df[actual_col].max(), df[pred_col].max())
    ax.plot([0, max_val], [0, max_val], 'r--')
    ax.set_title(title)
    ax.set_xlabel(actual_col)
    ax.set_ylabel(pred_col)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# -----------------------------------------------------------------------------
# Enhanced Model Evaluation Route
# -----------------------------------------------------------------------------
@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation() -> str:
    """
    Enhanced model evaluation route with advanced charts and collapsible UI.
    """
    lr_file = Path("models/results/performance_metrics_linear_regression.csv")
    rf_file = Path("models/results/performance_metrics_random_forest.csv")
    xgb_file = Path("models/results/performance_metrics_xgboost.csv")

    model_metrics_map = {
        "Linear Regression": lr_file,
        "Random Forest": rf_file,
        "XGBoost": xgb_file
    }

    metrics = {}
    combined_metrics = []
    for model_name, csv_file in model_metrics_map.items():
        if csv_file.exists():
            try:
                df_m = pd.read_csv(csv_file)
                # We won't display a table here since we replaced it with visual
                # But we can store the DataFrame for reference or future expansions
                row_data = {"Model": model_name}
                for col in ["MAE", "MSE", "RMSE", "R2"]:
                    if col in df_m.columns:
                        row_data[col] = df_m.loc[0, col]
                combined_metrics.append(row_data)
            except Exception as e:
                logger.error(f"Error reading {model_name} metrics: {e}")
        # else: file not found, skip

    # Build predictions dictionary
    predictions = {"Linear Regression": {}, "Random Forest": {}, "XGBoost": {}}
    base = Path.cwd()
    for pred_dir in [
        Path("data/predictions/linear_regression"),
        Path("data/predictions/random_forest"),
        Path("data/predictions/xgboost")
    ]:
        if pred_dir.exists():
            for f in pred_dir.rglob("*"):
                if f.is_file():
                    info = parse_predicted_file_details(f, base)
                    predictions.setdefault(info["model_name"], {}).setdefault(info["fe_variant"], {})[
                        info["league_season"]] = info["link"]

    # Create an overall metrics bar chart for "MAE" if available
    overall_metrics_plot = None
    if combined_metrics:
        df_combined = pd.DataFrame(combined_metrics)
        if "MAE" in df_combined.columns:
            df_combined = df_combined.sort_values("Model")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_combined["Model"], df_combined["MAE"], color="salmon")
            ax.set_title("Model Comparison (MAE)")
            ax.set_xlabel("Model")
            ax.set_ylabel("MAE")
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            overall_metrics_plot = base64.b64encode(buffer.read()).decode("utf-8")

    # Initialize placeholders for advanced analysis
    search_params = {}
    search_results = None
    interesting_abs = None
    interesting_pct = None
    interesting_abs_plot = None
    interesting_pct_plot = None
    interesting_under_plot = None
    interesting_over_plot = None
    top_under_valued_table = None
    top_over_valued_table = None
    error_dist_plot = None
    scatter_plot = None

    # Handle form submission
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

        # Determine file path
        if view_type == "Raw":
            file_path = Path("data/updated") / selected_fe / f"updated_{selected_league}_{selected_season}.parquet"
        else:
            if selected_model == "Linear Regression":
                file_path = Path(
                    "data/predictions/linear_regression") / selected_fe / f"predicted_updated_{selected_league}_{selected_season}.parquet"
            elif selected_model == "Random Forest":
                file_path = Path(
                    "data/predictions/random_forest") / selected_fe / f"predicted_updated_{selected_league}_{selected_season}.parquet"
            elif selected_model in ("XGBoost", "XG Boost"):
                file_path = Path(
                    "data/predictions/xgboost") / selected_fe / f"predicted_updated_{selected_league}_{selected_season}.parquet"
                if not file_path.exists():
                    file_path = Path(
                        "data/predictions/xgboost") / f"predicted_updated_{selected_league}_{selected_season}.parquet"
            else:
                file_path = None

        if not file_path or not file_path.exists():
            flash(f"Data file not found: {file_path}", "danger")
            search_results = f"Data file not found: {file_path}"
        else:
            try:
                df = pd.read_parquet(file_path)
                if search_term:
                    df = df[df["player"].str.lower().str.contains(search_term)]
                # Prepare table view
                cols = ["player", "position", "squad", "Market Value", "predicted_market_value"]
                keep_cols = [c for c in cols if c in df.columns]
                df = df[keep_cols]
                if "predicted_market_value" in df.columns:
                    df.rename(columns={"predicted_market_value": "Predicted Price"}, inplace=True)
                search_results = df.to_html(classes="table table-striped table-bordered", index=False)

                # Advanced analysis
                if "Market Value" in df.columns and "Predicted Price" in df.columns:
                    df["Absolute Difference"] = (df["Predicted Price"] - df["Market Value"]).abs()
                    df["Error"] = df["Predicted Price"] - df["Market Value"]
                    df["Percentage Difference"] = df.apply(
                        lambda row: abs((row["Predicted Price"] - row["Market Value"]) / row["Market Value"] * 100)
                        if row["Market Value"] != 0 else np.nan,
                        axis=1
                    )

                    top_abs = df.nlargest(5, "Absolute Difference").copy()
                    if not top_abs.empty:
                        interesting_abs = top_abs.to_html(classes="table table-striped table-bordered", index=False)
                        top_abs["player"] = top_abs["player"].astype(str)
                        interesting_abs_plot = df_to_bar_base64_png(
                            top_abs, "player", "Absolute Difference", "Top 5 Absolute Diff"
                        )

                    top_pct = df.nlargest(5, "Percentage Difference").copy()
                    if not top_pct.empty:
                        interesting_pct = top_pct.to_html(classes="table table-striped table-bordered", index=False)
                        top_pct["player"] = top_pct["player"].astype(str)
                        interesting_pct_plot = df_to_bar_base64_png(
                            top_pct, "player", "Percentage Difference", "Top 5 Percentage Diff"
                        )

                    df_neg = df[df["Error"] < 0].nsmallest(5, "Error").copy()
                    if not df_neg.empty:
                        top_under_valued_table = df_neg.to_html(classes="table table-striped table-bordered",
                                                                index=False)
                        interesting_under_plot = df_to_bar_base64_png(
                            df_neg, "player", "Error", "Top 5 Under-Valued (Negative Error)"
                        )

                    df_pos = df[df["Error"] > 0].nlargest(5, "Error").copy()
                    if not df_pos.empty:
                        top_over_valued_table = df_pos.to_html(classes="table table-striped table-bordered",
                                                               index=False)
                        interesting_over_plot = df_to_bar_base64_png(
                            df_pos, "player", "Error", "Top 5 Over-Valued (Positive Error)"
                        )

                    # Distribution and scatter
                    error_dist_plot = df_to_dist_base64_png(df["Error"], "Distribution of Error")
                    scatter_plot = df_to_scatter_base64_png(
                        df, "Market Value", "Predicted Price", "Market Value vs. Predicted Price"
                    )

            except Exception as e:
                flash(f"Error loading data: {e}", "danger")
                search_results = f"Error: {e}"

    return render_template(
        "model_evaluation.html",
        title="Model Evaluation",
        # We removed old metrics table references and replaced with a single bar chart
        metrics={},  # or an empty dict if you're no longer showing text-based metrics
        predictions=predictions,
        search_results=search_results,
        search_params=search_params,
        interesting_abs=interesting_abs,
        interesting_pct=interesting_pct,
        interesting_abs_plot=interesting_abs_plot,
        interesting_pct_plot=interesting_pct_plot,
        overall_metrics_plot=overall_metrics_plot,
        top_under_valued_table=top_under_valued_table,
        top_over_valued_table=top_over_valued_table,
        interesting_under_plot=interesting_under_plot,
        interesting_over_plot=interesting_over_plot,
        error_dist_plot=error_dist_plot,
        scatter_plot=scatter_plot
    )

@app.route("/run_all", methods=["GET", "POST"])
def run_all() -> str:
    if request.method == "POST":
        steps = [
            (["python", "./preprocessing/web_scrape.py"], "Web scraping"),
            (["python", "./preprocessing/preprocessing.py"], "Preprocessing"),
            (["python", "./preprocessing/player_value.py"], "Player value update"),
            (["python", "./models/linear_regression_model.py"], "Linear Regression training"),
            (["python", "./models/random_forest_model.py"], "Random Forest training"),
            (["python", "./models/xgboost_model.py"], "XGBoost training")
        ]
        for cmd, description in steps:
            if not run_command(cmd):
                flash(f"{description} failed.", "danger")
                return redirect(url_for("run_all"))
            else:
                flash(f"{description} completed.", "success")
        flash("Full pipeline executed.", "success")
        return redirect(url_for("index"))
    return render_template("run_all.html", title="Run Full Pipeline")


if __name__ == "__main__":
    if not start_local_api():
        logger.error("Failed to start local API server.")
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000/", new=0)
    app.run(debug=True)
