import requests
import json
import os
import time

def get_tocken():
    # Client Credentials
    client_id = "38053956-6618-4953-b670-b4ae7a2360b1"
    client_secret = "c385073c-3b97-42a9-b916-08fd8a5d1795"
    grant_type = "client_credentials"

    # URL for obtaining the token
    token_url = "https://rest.arbeitsagentur.de/oauth/gettoken_cc"

    # Payload for token request
    token_payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": grant_type
    }

    # Sending POST request to obtain the token
    response = requests.post(token_url, data=token_payload)

    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get("access_token")
        
        # Saving the token to a file
        with open("token.txt", "w") as token_file:
            token_file.write(token)
        
        print("Token saved to token.txt")

        # Using the generated token for subsequent requests
        headers = {
            "Authorization": f"Bearer {token}"
        }

        # URL for the GET request
        get_url = "https://rest.arbeitsagentur.de/infosysbub/wbsuche/pc/v1/bildungsangebot"

        # Sending a GET request with the token in the header
        response = requests.get(get_url, headers=headers)

        if response.status_code == 200:
            # Process the response data here
            print("GET request successful!")
            return token
        else:
            print("GET request failed:", response.status_code)
    else:
        print("Token request failed:", response.status_code)
    

generated_token = get_tocken()
headers = {
            "OAuthAccessToken": generated_token
    }

# Read ID list from id_list.txt
with open("provider_id.txt", "r") as id_file:
    id_list = id_file.read().strip().split("\n")


# Attempt to read the last id and page from last_position.txt file
start_id = None
start_page = None
if os.path.exists("last_position.txt"):
    with open("last_position.txt", "r") as f:
        lines = f.read().strip().split("\n")
        if len(lines) == 2:
            start_id, start_page = lines

# Base URL
base_url = "https://rest.arbeitsagentur.de/infosysbub/wbsuche/pc/v1/bildungsangebot"

# Initialization
found_start = False if start_id else True
dir_name = "result"

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
                    print("Retry limit reached, terminating program")
                    with open("last_position.txt", "w") as f:
                        f.write(f"{id}\n{current_page}")
                    exit(1)
                
                generated_token = get_tocken()
                headers = {
                    "OAuthAccessToken": generated_token
                    }
                print("Wait for 3 minutes before retrying...")
                time.sleep(180)
                continue

            retry_count = 0

            wb_data = response.json()
            
            file_name = "_".join(f"{k}_{v}" for k, v in params.items() if k != "page")
            folder_name = os.path.join(f"result_{id}")
            if not os.path.exists(dir_name,folder_name):
                os.mkdir(folder_name)
            with open(f"result/result_{id}/result_{file_name}_page_{params['page']}.json", "w") as result_file:
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
                    with open("last_position.txt", "w") as f:
                        f.write(f"{id}\n{current_page}")
                    exit(1)
                generated_token = get_tocken()
                headers = {
                    "OAuthAccessToken": generated_token
                    }
                print(f"Sleeping for 180 seconds before retrying...")
                time.sleep(180)
