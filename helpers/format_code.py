import subprocess
import os

def format_code_with_black(folder_path):
    try:
        subprocess.run(["black", folder_path], check=True)
        print(f"Successfully formatted code in {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during formatting: {e}")

# Formating
for file in os.listdir(helpers):
    format_code_with_black(os.getcwd() + "\\helpers\\"+ file)
format_code_with_black(os.getcwd() + "\\pipelines")