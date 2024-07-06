import json
import numpy as np
import pandas as pd
import os

folder_path = 'D:/BikeMap'

folders = os.listdir(folder_path)

json_files = []

for folder in folders:
    folder_path_json = os.path.join(folder_path, folder)
    if os.path.isdir(folder_path_json):
        files = os.listdir(folder_path_json)
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(f"{folder}/{file}")
                
def write_to_file(File_name):
    input_filename = f"D:/BikeMap/{File_name}"
    output_filename = f"CSV/CSV_All/{File_name.replace('/', '-').replace('.json', '')}.csv"
    
    # Check if the CSV file already exists
    if os.path.exists(output_filename):
        print(f"Skipping {File_name} as CSV already exists")
        return
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Skipping {File_name} due to file error: {e}")
        return

    Elevation_curve = data.get("elevation_curve")
    Distance = data.get("distance")

    if Elevation_curve is None or Distance is None:
        print(f"Skipping {File_name} due to missing data")
        return

    Distance_curve = np.linspace(0, Distance, len(Elevation_curve))
    df = pd.DataFrame({'Distance': Distance_curve, 'Elevation': Elevation_curve})
    
    try:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df.to_csv(output_filename, index=False)
        print(f"{File_name} saved successfully")
    except Exception as e:
        print(f"Failed to save {File_name} due to error: {e}")

for File_name in json_files:
    write_to_file(File_name)
