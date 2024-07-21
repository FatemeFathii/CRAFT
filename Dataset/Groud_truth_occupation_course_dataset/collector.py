#read csv from ./Data/data_collection/Berufenet/berufe_info.csv
import pandas as pd
import requests
import json
import os
def ensure_directory_exists(directory):
    """Ensure directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_oauth_token():
    # API endpoint URL
    token_url = "https://rest.arbeitsagentur.de/oauth/gettoken_cc"
    
    # Client credentials
    client_id = "38053956-6618-4953-b670-b4ae7a2360b1"
    client_secret = "c385073c-3b97-42a9-b916-08fd8a5d1795"
    grant_type = "client_credentials"
    
    # Prepare request data
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": grant_type
    }
    
    # Send request and get response
    response = requests.post(token_url, data=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print("Token request failed:", response.text)
        return None

def query_weiterbildungssuche_api(filepath, token, params,beruf_id):
    # Set API URL
    api_url = "https://rest.arbeitsagentur.de/infosysbub/wbsuche/pc/v2/bildungsangebot"
    
    # Prepare request headers
    result_path = os.path.join(filepath, "aw_course")
    ensure_directory_exists(result_path)
    current_page = 0
    rety_count = 0
    while True:
        headers = {
        "Authorization": f"Bearer {token}"
    }
        # Send GET request
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            rety_count = 0
            wb_data = response.json()
            file_name = "_".join(f"{k}_{v}" for k, v in params.items() if k != "page")
            with open(os.path.join(result_path,f"result_{beruf_id}_{file_name}_page_{params['page']}.json"), "w") as result_file:
                json.dump(wb_data, result_file, indent=4)
            print(f"result saved to result_{beruf_id}_{file_name}_page_{params['page']}.json")


            next_page_link = wb_data.get("_links", {}).get("next", None)
            if next_page_link:
                current_page += 1
                params["page"] = str(current_page)
            else:
                break
           
        else:
            print("API request failed:", response.text)
            token = get_oauth_token()
            rety_count += 1
            if rety_count >= 10:
                print("retry count exceed 10, exit")
                exit(1)
            continue
            
            
def get_aw_courses(filepath):
    berufe_info = pd.read_csv(os.path.join(filepath,'occupation_info.csv'))
    # get id and entry requirement
    token = get_oauth_token()
    flag = 0
    for index, row in berufe_info.iterrows():
        if flag ==0 and row['id'] != 89948:
            continue
        flag = 1
        BA_berufe = set()
        entry_requirement = row['entry requirements']
        corrected_json_string = entry_requirement.replace("'", '"')
        entry_requirement_json = json.loads(corrected_json_string)
        for js in entry_requirement_json:
            BA_berufe.add(str(js['data_idref']))
        for id in BA_berufe:
            params = {
                'page': 0,
                "ids": id
                }
            query_weiterbildungssuche_api(filepath, token, params,row['id'])