import os
import subprocess

# Define the scripts to be executed in sequence
SCRIPTS = [
    "web_scrape.py",  # Script to scrape player data
    "preprocessing.py",  # Script to preprocess scraped data
    "player_value.py"  # Script to add market values to the preprocessed data
]

def execute_script(script_name):
    """
    Execute a Python script using subprocess.

    Args:
        script_name (str): Name of the Python script to execute.

    Raises:
        FileNotFoundError: If the script is not found.
    """
    script_path = os.path.join(os.getcwd(), script_name)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script {script_name} not found in the current directory.")

    try:
        print(f"Executing {script_name}...")
        result = subprocess.run(["python", script_name], capture_output=True, text=True)

        # Log the output and errors
        if result.returncode == 0:
            print(f"{script_name} completed successfully.\nOutput:\n{result.stdout}")
        else:
            print(f"{script_name} encountered an error.\nError:\n{result.stderr}")
    except Exception as e:
        print(f"An error occurred while executing {script_name}: {e}")

def main():
    """
    Main function to execute all scripts sequentially.
    """
    print("Starting the execution of all scripts...")
    for script in SCRIPTS:
        try:
            execute_script(script)
        except FileNotFoundError as fnf_error:
            print(fnf_error)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("All scripts have been executed.")

if __name__ == "__main__":
    main()