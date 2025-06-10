import requests
import json
import os
import re
import csv
import time
import pandas as pd
import glob
from bs4 import BeautifulSoup


def get_aw_courses(filepath):
    #generated_token = get_tocken(filepath)
    headers = {
                 "X-API-Key": "infosysbub-wbsuche",
        }

    # Read ID list from provider_id.txt
    id_list_path = "Dataset\Course_dataset\provider_list.txt"
    if not os.path.isfile(id_list_path):
        print("provider_id.txt does not exist.")
        return
    
    with open(id_list_path, "r") as id_file:
        id_list = id_file.read().strip().split("\n")


    # Attempt to read the last id and page from last_position.txt file
    last_position_path = os.path.join(filepath, "last_position.txt")
    start_id = None
    start_page = None
    if os.path.exists(last_position_path):
        with open(last_position_path, "r") as f:
            lines = f.read().strip().split("\n")
            if len(lines) == 2:
                start_id, start_page = lines

    # Base URL
    base_url = "https://rest.arbeitsagentur.de/infosysbub/wbsuche/pc/v2/bildungsangebot"

    # Initialization
    found_start = False if start_id else True
   
    for id in  id_list:
        if not found_start:
            if id == start_id:
                found_start = True
            else:
                continue
        
        #current_page = int(start_page) if id == start_id else 0
        current_page = 0
        params = {
            "page": int(current_page),
            "ban": id

        }


        while True:
            try:
                response = requests.get(base_url, headers=headers, params=params)

                if response.status_code != 200:
                    print(f"GET request failed, status code: {response.status_code}, ID: {id}")
                    retry_count += 1
                    
                    if retry_count >= 10:
                        print("Retry limit reached, saving last position and exiting.")
                        with open(last_position_path, "w") as f:
                            f.write(f"{id}\n{current_page}")
                        return
                    
                    headers = {
                        "OAuthAccessToken": generated_token
                        }
                    print("Wait for 3 minutes before retrying...")
                    time.sleep(180)
                    continue

                retry_count = 0

                wb_data = response.json()
                
                
                file_name = "_".join(f"{k}_{v}" for k, v in params.items() if k != "page")
                folder_name = os.path.join(filepath, f"result_{id}")
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                with open(os.path.join(folder_name, f"result_{file_name}_page_{params['page']}.json"), "w") as result_file:
                    json.dump(wb_data, result_file, indent=4)
                print(f"Result has been saved to result_{file_name}_page_{params['page']}.json file")


                next_page_link = wb_data.get("_links", {}).get("next", None)
                if next_page_link:
                    current_page += 1
                    params["page"] = str(current_page)
                else:
                    break
            except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
                    retry_count += 1
                    print(f"Retry #{retry_count}")

                    if retry_count > 10:
                        print("Max retries reached. Exiting...")
                        with open(os.path.join(filepath,"last_position.txt"), "w") as f:
                            f.write(f"{id}\n{current_page}")
                        exit(1)
                    generated_token = get_tocken()
                    headers = {
                        "OAuthAccessToken": generated_token
                        }
                    print(f"Sleeping for 180 seconds before retrying...")
                    time.sleep(180)





def aw_tocsv(filepath):
    parent_folder = filepath
    csv_folder = os.path.join(parent_folder, 'data_csv')

    for folder_name, subfolders, filenames in os.walk(parent_folder):
    
        match = re.match(r'^result_(\d+)$', os.path.basename(folder_name))
        if not match:
            continue
        
        id = match.group(1)
    
        pattern = r'^result_.*.json$'
        matched_files = [f for f in filenames if re.match(pattern, f)]

        
        written_ids = set()

        with open(os.path.join(csv_folder,f'result_{id}.csv'), 'w', newline='', encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            
            csvwriter.writerow(['ID', 'Title', 'Content', 'Keywords','Link'])
            for file_name in matched_files:
                with open(os.path.join(folder_name, file_name), 'r') as json_file:
                    print(file_name)
                    data = json.load(json_file)
                    
                    termine_items = data.get("_embedded", {}).get("termine", [])
                    for item in termine_items:
                        offer_data = item["angebot"]
                        id = offer_data["id"]
                    
                        if id in written_ids:
                            continue
                        title = offer_data["titel"]
                        content = offer_data["inhalt"]
                        link = offer_data["link"]
                        keywords = ', '.join([word["suchwort"] for word in offer_data["suchworte"]])
                        csvwriter.writerow([id, title, content, keywords, link])
                        
                        written_ids.add(id)

def aw_preprocess(filename):
    # Set the folder path where the CSV files are located
    folder_path = os.path.join(filename, 'data_csv')

    # Read all CSV files in the folder
    all_files = glob.glob(folder_path + "/*.csv")

    # Create an empty DataFrame to hold the merged data
    all_data = pd.DataFrame()

    # Read each file one by one
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        # Append data to the all_data DataFrame
        all_data = pd.concat([all_data, df])

    # Remove duplicates based on each column individually
    for column in ['ID', 'Title', 'Content', 'Keywords']:
        all_data = all_data.drop_duplicates(subset=column, keep='first')
    
    # Clean HTML content in the 'Content' column after removing duplicates
    all_data['Content'] = all_data['Content'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    all_data['Content'] = all_data['Content'] + ' ' + all_data['Keywords']
    all_data.drop(columns=['Keywords'], inplace=True)
    # Save to a new CSV file
    all_data.to_csv('AW_courses.csv', index=False)

def ensure_directory_exists(directory):
    """Ensure directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def aw_get_courses(filepath):
    ensure_directory_exists(filepath)
    get_aw_courses(filepath)
    aw_tocsv(filepath)
    aw_preprocess(filepath)