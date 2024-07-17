import requests
import json
from glob import glob

def read_json_file(filepath):
    """Read JSON data from the given file path"""
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json_to_file(data, filepath):
    """Save data as a JSON file"""
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def get_berufs_details(berufs_ids, client_id, output_directory):
    """Obtain detailed information for each profession by profession ID list and Client-ID through the API and save it as JSON files"""
    headers = {"X-API-Key": client_id}
    base_url = "https://rest.arbeitsagentur.de/infosysbub/bnet/pc/v1/berufe"

    for berufs_id in berufs_ids:
        url = f"{base_url}/{berufs_id}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Build output file path
            output_filepath = f"{output_directory}/beruf_{berufs_id}.json"
            # Save the retrieved data to a JSON file
            save_json_to_file(response.json(), output_filepath)
            print(f"Details for profession ID {berufs_id} have been saved to {output_filepath}")
        else:
            print(f"Error when requesting profession ID {berufs_id}, status code: {response.status_code}")

def process_files_in_directory(directory, client_id, output_directory):
    """Process all matching files in the directory and save the results as JSON files"""
    file_pattern = f"{directory}/result__page_*.json"
    for filepath in glob(file_pattern):
        json_data = read_json_file(filepath)
        berufs_ids = [beruf["id"] for beruf in json_data["_embedded"]["berufSucheList"]]
        get_berufs_details(berufs_ids, client_id, output_directory)

# Folder path
directory = 'result'
# Output folder path
output_directory = 'beruf_detail'

# Client-ID
client_id = "d672172b-f3ef-4746-b659-227c39d95acf"

# Process all files in the folder
process_files_in_directory(directory, client_id, output_directory)
