import requests
import json
import os
import time
import urllib.parse

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

while True:
    try:
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"GET request failed, status code: {response.status_code}, ID: {id}")
            break

        wb_data = response.json()


        with open(f"result/result_page_{params['page']}.json", "w") as result_file:
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
