import requests
import json
import os
from glob import glob
import csv
from bs4 import BeautifulSoup
import fire

def ensure_directory_exists(directory):
    """Ensure directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_occupation_list(filepath):

    client_id = "d672172b-f3ef-4746-b659-227c39d95acf"

    # Headers including the Client-ID as 'X-API-Key'
    headers = {
        "X-API-Key": client_id
    }

    # Base URL
    base_url = "https://rest.arbeitsagentur.de/infosysbub/bnet/pc/v1/berufe"
    current_page = 0
    params = {
        "page": current_page,
    }
    result_filepath = os.path.join(filepath,"result")
    ensure_directory_exists(result_filepath)
    while True:
        try:
            response = requests.get(base_url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"GET request failed, status code: {response.status_code}")
                break

            wb_data = response.json()


            with open(os.path.join(result_filepath,f"result_page_{params['page']}.json"), "w") as result_file:
                json.dump(wb_data, result_file, indent=4)
            print(f"Results have been saved to result_page_{params['page']}.json file")

            next_page_link = wb_data.get("_links", {}).get("next", None)
            if next_page_link:
                current_page += 1
                params["page"] = str(current_page)
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break



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

def get_occupation_detail(filepath):
    # Folder path
    directory = os.path.join(filepath,'result')
    # Output folder path
    output_directory = os.path.join(filepath,'beruf_detail')
    ensure_directory_exists(output_directory)

    # Client-ID
    client_id = "d672172b-f3ef-4746-b659-227c39d95acf"

    # Process all files in the folder
    process_files_in_directory(directory, client_id, output_directory)


def json_to_csv(filepath):
    # Path to the directory containing JSON files
    directory_path = os.path.join(filepath,'beruf_detail')

    # Path for the output CSV file
    output_csv_file = os.path.join(filepath,'occupation_info.csv')

    # Open a CSV file for writing
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Writing header to CSV file
        # id,short name,full name,KldB codes,DKZ codes,entry requirements,training opportunities,essential skills,optional skills,description
        writer.writerow([
            'id','short name','full name','KldB codes','DKZ codes','entry requirements','training opportunities','essential skills','optional skills','description'
        ])

        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):  # Check if the file is a JSON file
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Iterate through each item if JSON is a list of entries
                    for item in data:
                        if not item.get('bkgr')['typ']['id'] == 't':
                            continue
                        # Extract entry requirements from the 'infofelder' section where 'ueberschrift' matches 'Zugangsberufe/Zugangstätigkeiten'
                        entry_requirements = []
                        essential_skills = []
                        optional_skills = []
                        for field in item.get('infofelder', []):
                            # get entry requirements
                            if field.get('ueberschrift', '') == 'Zugangsberufe/Zugangstätigkeiten':
                                content_html = field.get('content', '')
                                soup = BeautifulSoup(content_html, 'html.parser')
                                # Extracting each item
                                for extsysref in soup.find_all('ba-berufepool-extsysref'):
                                    entry_dict = {
                                        'data_idref': extsysref.get('data-idref', ''),
                                        'data_ext_system': extsysref.get('data-ext-system', ''),
                                        'text': extsysref.get_text(strip=True)
                                    }
                                    entry_requirements.append(entry_dict)
                            # get essential skills and optional skills
                            if field.get('ueberschrift', '') == 'Kompetenzen':
                                content_html = field.get('content', '')
                                soup = BeautifulSoup(content_html, 'html.parser')
                                # Distinguish between core skills and other skills by section
                                sections = soup.find_all('section')
                                
                                # Assuming the first section contains core skills and the second contains other skills
                                if sections:
                                    core_skill_elements = sections[0].find_all('ba-berufepool-extsysref')
                                    for element in core_skill_elements:
                                        skill_id = element.get('data-idref')
                                        skill_name = element.text.strip()
                                        essential_skills.append({'id': skill_id, 'skill': skill_name})
                                
                                if len(sections) > 1:
                                    other_skill_elements = sections[1].find_all('ba-berufepool-extsysref')
                                    for element in other_skill_elements:
                                        skill_id = element.get('data-idref')
                                        skill_name = element.text.strip()
                                        optional_skills.append({'id': skill_id, 'skill': skill_name})
                            # description
                            if field.get('ueberschrift', '') == 'Aufgaben und Tätigkeiten kompakt':
                                html_content = field.get('content', '')
                                soup = BeautifulSoup(html_content, 'html.parser')
                                description = soup.get_text(strip=True)

                        # get training opportunities
                        training_opportunities = []
                        for trainings in item.get('aufstiegsweiterbildungen', []):
                            training_opportunitie = {
                                'training_id': trainings.get('id', ''),
                                'training_name': trainings.get('kurzBezeichnungNeutral', ''),
                                'training_type': trainings.get('aufstiegsart', '')['id'],
                            }
                            training_opportunities.append(training_opportunitie)                    

                        writer.writerow([
                            item.get('id', ''), # id
                            item.get('kurzBezeichnungNeutral', ''), # short name
                            item.get('bezeichnungNeutral', ''), # full name
                            item.get('kldb2010', ''), # KldB codes
                            item.get('codenr', ''), # DKZ codes
                            entry_requirements, # entry requirements
                            training_opportunities,# training opportunities
                            essential_skills, # essential skills
                            optional_skills, # optional skills
                            description# description
                        ])

    print("Data extraction and CSV writing complete.")

def get_berufenet(filename):
    get_occupation_list(filename)
    get_occupation_detail(filename)
    json_to_csv(filename)

