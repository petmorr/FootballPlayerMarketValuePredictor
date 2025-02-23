import subprocess
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

from logging_config import configure_logger

logger = configure_logger("web_portal", "web_portal.log")

app = Flask(__name__)
app.secret_key = "replace_with_your_secret_key"

# Directory containing updated (preprocessed) datasets
UPDATED_DATA_DIR = Path("data/updated")

# --------------------------
# Helper Functions for Manual Transfer Value Input
# --------------------------

def get_clean_basename(file_path: str) -> str:
    """
    Returns the base name of the file without extensions.
    For example, "cleaned_Bundesliga_2019-2020.parquet" becomes "cleaned_Bundesliga_2019-2020".
    """
    p = Path(file_path)
    if len(p.suffixes) > 1:
        return p.name.split('.')[0]
    else:
        return p.stem


def load_missing_transfer_values():
    """
    Scan each updated dataset (Parquet or gzipped CSV) in UPDATED_DATA_DIR for rows
    where the "market value" column is null.
    Returns a list of dictionaries, each containing details for one missing entry.
    """
    missing_entries = []
    for file in UPDATED_DATA_DIR.glob("cleaned_*"):
        base_name = get_clean_basename(file.name)  # e.g., "cleaned_Bundesliga_2019-2020"
        parts = base_name.split("_")
        if len(parts) >= 3:
            league = parts[1]
            season = parts[2]
        else:
            league = "Unknown"
            season = "Unknown"
        try:
            if file.suffix == ".parquet":
                df = pd.read_parquet(file)
            elif file.name.endswith(".csv.gz"):
                df = pd.read_csv(file, compression="gzip", encoding="utf-8")
            else:
                continue
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue

        # Normalize column names to lower-case for consistency.
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
    Group missing entries by dataset.
    Returns a dictionary: keys are dataset names, values are lists of entries.
    """
    grouped = {}
    for entry in missing_entries:
        ds = entry["dataset"]
        grouped.setdefault(ds, []).append(entry)
    return grouped


def update_transfer_value_in_parquet(dataset, season, updates):
    """
    For the given dataset (e.g. "Bundesliga_2019-2020") and season,
    open the corresponding parquet file in UPDATED_DATA_DIR, update rows (matching on player and team,
    case-insensitive) with the provided manual transfer value, and write back the file.
    """
    filename = f"cleaned_{dataset}.parquet"
    file_path = UPDATED_DATA_DIR / filename
    if not file_path.exists():
        logger.error(f"File {file_path} not found for updating transfer value.")
        return False
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Error reading parquet file {file_path}: {e}")
        return False

    df.columns = [col.lower() for col in df.columns]

    for update in updates:
        player = update["player"]
        team = update["team"]
        new_value = update["manual_transfer_value"]
        mask = (df["player"].str.lower() == player.lower()) & (df["squad"].str.lower() == team.lower())
        if mask.sum() == 0:
            logger.warning(f"No matching row found for player '{player}' in team '{team}' in file {file_path}.")
            continue
        df.loc[mask, "market value"] = new_value
        logger.info(f"Updated {player} ({team}) in {dataset} with new value {new_value}.")
    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        logger.error(f"Error writing updated parquet file {file_path}: {e}")
        return False
    return True

# --------------------------
# Routes
# --------------------------

@app.route("/")
def index():
    return render_template("index.html", title="Home")


# ----- Combined Model Preprocessing Route -----
@app.route("/model_preprocessing", methods=["GET", "POST"])
def model_preprocessing():
    """
    This route combines web scraping, model preprocessing, and player value extraction.
    When the user clicks the "Run Model Preprocessing" button, the backend scripts are executed
    in order. After that, the system checks for any players missing transfer values.
    If missing values are found, the user is prompted to provide manual inputs.
    """
    if request.method == "POST":
        try:
            subprocess.run(["python", "./preprocessing/web_scrape.py"], check=True)
            subprocess.run(["python", "./preprocessing/preprocessing.py"], check=True)
            subprocess.run(["python", "./preprocessing/player_value.py"], check=True)
            flash("Preprocessing completed successfully.", "success")
        except Exception as e:
            flash("Error during preprocessing.", "danger")
        # Then check for missing transfer values, etc.
        return redirect(url_for("index"))
    return render_template("model_preprocessing.html", title="Model Preprocessing")


# ----- Manual Input for Missing Transfer Values -----
@app.route("/manual_input", methods=["GET", "POST"])
def manual_input():
    if request.method == "POST":
        # Process manual inputs for missing transfer values.
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

        # Group updates by dataset.
        grouped_updates = {}
        for update in updates:
            ds = update["dataset"]
            grouped_updates.setdefault(ds, []).append(update)

        # For each dataset, update the corresponding parquet file.
        for ds, upd_list in grouped_updates.items():
            success = update_transfer_value_in_parquet(ds, upd_list[0]["season"], upd_list)
            if not success:
                flash(f"Failed to update dataset {ds}.", "danger")
            else:
                flash(f"Successfully updated dataset {ds}.", "success")
        return redirect(url_for("manual_input"))
    else:
        missing_entries = load_missing_transfer_values()
        grouped_missing = group_missing_entries(missing_entries)
        return render_template("manual_input.html", grouped_missing=grouped_missing,
                               title="Manual Transfer Value Input")


# ----- Model Creation (Placeholder) -----
@app.route("/model_creation", methods=["GET", "POST"])
def model_creation():
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        flash(f"Model creation triggered for {model_choice} (functionality not yet implemented).", "info")
        logger.info(f"Model creation triggered: {model_choice} (placeholder).")
        return redirect(url_for("model_creation"))
    return render_template("model_creation.html", title="Model Creation")


# ----- Model Evaluation (Placeholder) -----
@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation():
    if request.method == "POST":
        flash("Model evaluation functionality is not yet implemented.", "info")
        logger.info("Model evaluation triggered (placeholder).")
        return redirect(url_for("model_evaluation"))
    return render_template("model_evaluation.html", title="Model Evaluation")


# ----- Run Full Pipeline (Placeholder) -----
@app.route("/run_all", methods=["GET", "POST"])
def run_all():
    if request.method == "POST":
        flash("Full pipeline run functionality is not yet implemented.", "info")
        logger.info("Full pipeline run triggered (placeholder).")
        return redirect(url_for("index"))
    return render_template("run_all.html", title="Run Full Pipeline")

# --------------------------
# Main Entry Point
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)