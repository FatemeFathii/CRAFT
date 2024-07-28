from config import *
import pandas as pd
import requests




def build_skill_query(skill):
    skill_query =  skill['preferredLabel']['de'] +" " + skill['preferredLabel']['en']+" "+ skill['description']['de']+ " "+ skill['description']['en']
    if skill['borderConcept']['broaderHierarchyConcept']:
        broader_hierarchy_concept_str = ", ".join(skill['borderConcept']['broaderHierarchyConcept'])
        skill_query += " " + broader_hierarchy_concept_str
    else:
        pass
    if skill['borderConcept']['broaderSkill']:
        broaderSkill_str = ", ".join(skill['borderConcept']['broaderSkill'])
        skill_query += " " + broaderSkill_str
    else:
        pass
    if skill['alternativeLabel']['de']:
        alternativeLabel_de_str = ", ".join(skill['alternativeLabel']['de'])
        skill_query += " " + alternativeLabel_de_str
    else:
        pass
    if skill['alternativeLabel']['en']:
        alternativeLabel_en_str = ", ".join(skill['alternativeLabel']['en'])
        skill_query += " " + alternativeLabel_en_str
    else:
        pass 
    return skill_query



def build_occupation_query(occupation):
    occupation_name_de = occupation['preferredLabel'].get('de','')
    occupation_dsp = occupation['description'].get('de','').get('literal','')
    occupation_query = occupation_name_de +" " + occupation['preferredLabel'].get('en','')+" "+ occupation['description'].get('de','').get('literal','') + " "+ occupation_dsp
    '''
    if occupation['_links']['broaderIscoGroup']:
        for group in occupation['_links']['broaderIscoGroup']:
            occupation_query += " " + group['title']
    else:
        pass
        '''
    return occupation_name_de,occupation_dsp,occupation_query      

# Get occupations from a CSV
def get_occupations_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df[['preferredLabel', 'conceptUri']].set_index('preferredLabel').to_dict()['conceptUri']


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

def query_weiterbildungssuche_api(token, params):
    # Set API URL
    api_url = "https://rest.arbeitsagentur.de/infosysbub/wbsuche/pc/v2/bildungsangebot"
    
    # Prepare request headers
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Send GET request
    response = requests.get(api_url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print("API request failed:", response.text)
        return None

def get_courses_from_BA(ids):
    # Get OAuth token
    token = get_oauth_token()
    if token:
        # Set query parameters
        params = {
            "ids": list(ids)
        }
        # Use token to query the API
        result = query_weiterbildungssuche_api(token, params)
        return result


def get_occupation_detial(oc_uri):
    uri = f'https://ec.europa.eu/esco/api/resource/occupation?selectedVersion=v1.0.9&language=en&uri={oc_uri}'
    try:
            occupation_response = requests.get(uri)
            occupation_response.raise_for_status()
            occupation_json = occupation_response.json()
            return occupation_json
            
    except requests.RequestException as e:
        print(f"Error while fetching skill details: {e}")




