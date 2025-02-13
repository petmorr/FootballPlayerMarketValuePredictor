from flask import Flask, render_template, request, redirect, url_for, flash

from logging_config import configure_logger

logger = configure_logger("web_portal", "web_portal.log")

app = Flask(__name__)
app.secret_key = "replace_with_your_secret_key"


# --------------------------
# Home / Index Route
# --------------------------
@app.route("/")
def index():
    return render_template("index.html", title="Home")


# --------------------------
# Web Scraping Route (Placeholder)
# --------------------------
@app.route("/web_scraping", methods=["GET", "POST"])
def web_scraping():
    if request.method == "POST":
        flash("Web scraping functionality is not yet implemented.")
        logger.info("Web scraping triggered (placeholder).")
        return redirect(url_for("web_scraping"))
    return render_template("web_scraping.html", title="Web Scraping")


# --------------------------
# Preprocessing Route (Placeholder)
# --------------------------
@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    if request.method == "POST":
        flash("Preprocessing functionality is not yet implemented.")
        logger.info("Preprocessing triggered (placeholder).")
        return redirect(url_for("preprocessing"))
    return render_template("preprocessing.html", title="Preprocessing")


# --------------------------
# Player Value / Manual Input Route
# --------------------------
@app.route("/player_value", methods=["GET", "POST"])
def player_value():
    if request.method == "POST":
        # Here, you would process the user input for a single player.
        player_name = request.form.get("player_name")
        team_name = request.form.get("team_name")
        dataset = request.form.get("dataset")
        season = request.form.get("season")
        closest_date = request.form.get("closest_date")
        transfer_value = request.form.get("transfer_value")
        flash(
            f"Manual input received for {player_name} (Team: {team_name}, {dataset}, {season}, Closest Date: {closest_date}) with transfer value: {transfer_value}.")
        logger.info(
            f"Manual player value input: {player_name}, Team: {team_name}, Dataset: {dataset}, Season: {season}, Closest Date: {closest_date}, Value: {transfer_value}")
        return redirect(url_for("player_value"))
    return render_template("player_value.html", title="Player Market Value")


# --------------------------
# Model Creation / Tuning Route (Placeholder)
# --------------------------
@app.route("/model_creation", methods=["GET", "POST"])
def model_creation():
    if request.method == "POST":
        model_choice = request.form.get("model_choice")
        flash(f"Model creation triggered for {model_choice} (functionality not yet implemented).")
        logger.info(f"Model creation triggered: {model_choice} (placeholder).")
        return redirect(url_for("model_creation"))
    return render_template("model_creation.html", title="Model Creation")


# --------------------------
# Model Evaluation Route (Placeholder)
# --------------------------
@app.route("/model_evaluation", methods=["GET", "POST"])
def model_evaluation():
    if request.method == "POST":
        flash("Model evaluation functionality is not yet implemented.")
        logger.info("Model evaluation triggered (placeholder).")
        return redirect(url_for("model_evaluation"))
    return render_template("model_evaluation.html", title="Model Evaluation")


# --------------------------
# Run Full Pipeline Route (Placeholder)
# --------------------------
@app.route("/run_all", methods=["GET", "POST"])
def run_all():
    if request.method == "POST":
        flash("Full pipeline run functionality is not yet implemented.")
        logger.info("Full pipeline run triggered (placeholder).")
        return redirect(url_for("index"))
    return render_template("run_all.html", title="Run Full Pipeline")


# --------------------------
# Manual Input Route for Missing Transfer Values
# --------------------------
@app.route("/manual_input", methods=["GET", "POST"])
def manual_input():
    if request.method == "POST":
        # Process the manual inputs.
        # For simplicity, assume that the form submits lists of values.
        players = request.form.getlist("player")
        teams = request.form.getlist("team")
        datasets = request.form.getlist("dataset")
        seasons = request.form.getlist("season")
        closest_dates = request.form.getlist("closest_date")
        manual_values = request.form.getlist("manual_value")

        # In a complete implementation, you would update the dataset accordingly.
        flash("Manual input received for transfer values (functionality not yet fully implemented).")
        logger.info(f"Manual input received for players: {players} with values: {manual_values}")
        return redirect(url_for("manual_input"))

    # Sample data for demonstration purposes.
    # In your full system, this would be dynamically generated from the updated dataset.
    sample_missing = [
        {
            "player": "Miloš Veljković",
            "team": "SV Werder Bremen",
            "dataset": "Bundesliga_2019-2020",
            "season": "2019-2020",
            "closest_date": "2020-05-20",
            "current_value": "N/A"
        },
        {
            "player": "Obite N'Dicka",
            "team": "Eint Frankfurt",
            "dataset": "Bundesliga_2021-2022",
            "season": "2021-2022",
            "closest_date": "2022-06-15",
            "current_value": "N/A"
        },
        {
            "player": "Evan N'Dicka",
            "team": "Eint Frankfurt",
            "dataset": "Bundesliga_2021-2022",
            "season": "2021-2022",
            "closest_date": "2022-06-15",
            "current_value": "N/A"
        }
    ]
    return render_template("manual_input.html", players=sample_missing, title="Manual Transfer Value Input")


# --------------------------
# Main Entry Point
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
