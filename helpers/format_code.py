import subprocess
import os


def format_code_with_black(folder_path):
    try:
        if os.path.exists(folder_path):
            subprocess.run(["python", "-m", "black", folder_path], check=True)
            print(f"Successfully formatted code in {folder_path}")
        else:
            print(f"Path does not exist: {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during formatting: {e}")


# Formating
format_code_with_black("helpers")
format_code_with_black("pipelines")
