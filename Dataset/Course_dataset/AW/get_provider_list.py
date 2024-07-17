import requests
import json
import os

def get_token():
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

        return token

generated_token = get_token()

headers = {
    "Authorization": f"Bearer {generated_token}"
}

# Base URL
base_url = "https://rest.arbeitsagentur.de/infosysbub/wbsuche/pc/v1/bildungsangebot"
try:
    response = requests.get(base_url, headers=headers)

    if response.status_code != 200:
        print(f"GET request failed, status code: {response.status_code}")
    else:
        # Parsing the response
        data = response.json()
        provider_data = data.get("aggregations", {}).get("ANBIETER", {})

        # Extracting provider IDs
        provider_ids = list(provider_data.keys())

        # Save provider IDs to a text file
        with open('provider_list.txt', 'w') as f:
            for id in provider_ids:
                f.write(id + '\n')
            print("Provider IDs saved to provider_list.txt")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
