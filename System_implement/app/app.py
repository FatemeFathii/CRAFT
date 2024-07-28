import gradio as gr
import pandas as pd
import redis
import json
import requests
from config import *
import functools
from embedding_setup import retriever, find_similar_occupation, compare_docs_with_context,generate_exp,generate_prompt_exp
from data_process import  get_occupations_from_csv, get_courses_from_BA, get_occupation_detial, build_occupation_query
with open('/app/data/redis_data.json', 'r') as file:
    data_dict = json.load(file)
#r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

skill_details_mapping = {}


# Function to retrieve documents based on selected skills
def retrieve_documents(occupation,skills):
    output = []
    output.append(f"<div style=\"text-align: center; font-size: 24px;\">Empfehlungsergebnisse:</div>")
    oc_uri = occupations.get(occupation, "")
    skill_query = ''
    if isinstance(oc_uri, int):
        df = pd.read_csv("/app/data/berufe_info.csv")
        target_occupation = df[df['id'] == oc_uri]
        target_occupation_name = target_occupation['short name'].values[0]
        target_occupation_dsp = target_occupation['description'].values[0]
        target_occupation_query = target_occupation_name + ' ' + target_occupation_dsp
        target_occupation_query = target_occupation_query
    else:
        target_occupation = get_occupation_detial(oc_uri)
        target_occupation_name, target_occupation_dsp, target_occupation_query = build_occupation_query(target_occupation)
    for german_label in skills:
        skill_query += german_label + ' '
    query = 'target occupation: ' + target_occupation_query + ' Skills gap:' + skill_query
    llama_query = 'info:' + target_occupation_name + ' ' + 'Skills gap:' + skill_query
    print(query)
    docs = retriever.get_relevant_documents(query)

    partial_compare_docs = functools.partial(compare_docs_with_context, target_occupation_name=target_occupation_name, target_occupation_dsp=target_occupation_dsp,skill_gap = skill_query)
    sorted_docs = sorted(docs, key=functools.cmp_to_key(partial_compare_docs), reverse=True)

    
    batch_prompts = []
    for doc in sorted_docs[:5]:
        doc_name = doc.metadata.get('name', 'Unnamed Document')
        doc_skill = doc.metadata.get('skills', '')
        input_text = f"target occupation: {llama_query}\n Recommended course: name: {doc_name}, learning objectives: {doc_skill[:2000]}"
        prompt = generate_prompt_exp(input_text)
        batch_prompts.append(prompt)

    # Evaluate the current batch of prompts
    batch_output = generate_exp(batch_prompts)
    output.append(f"<b>Zielberuf:</b> {target_occupation_name}")
    output.append(f"<b>Qualifikationslücke:</b> {skill_query}")
    output.append(f"<b>Empfohlene Kurse:</b>")
    for i in range(5):
        doc = sorted_docs[i]
        doc_name = doc.metadata.get('name', 'Unnamed Document')
        doc_url = doc.metadata.get('url', '#')
        doc_skill = doc.metadata.get('skills', '')
        output.append(f"<a href='{doc_url}' target='_blank'>{doc_name}</a>") 
        output.append(f"<b>Empfehlungsgrund:</b> {batch_output[i]}")
    

    output.append(f"<br>")
    return "<br>".join(output)


def get_candidate_courses(occupation, skills):
    output = []
    output.append(f"<div style=\"text-align: center; font-size: 24px;\">Empfehlungsergebnisse:</div>")
    df_lookup = pd.read_csv('/app/data/kldb_isco_lookup.csv')
    df_berufe = pd.read_csv('/app/data/berufe_info.csv')
    occupation_codes = set()
    kldB_set = set()
    occupation_hrefs = set()
    BA_berufe = set()
    oc_uri = occupations.get(occupation, "")
    target_occupation = get_occupation_detial(oc_uri)
    target_occupation_query = build_occupation_query(target_occupation)
    
    for german_label in skills:
        skill = skill_details_mapping.get(german_label, {})
        uri = f'https://ec.europa.eu/esco/api/resource/skill?selectedVersion=v1.0.9&language=en&uri={skill["uri"]}'
        try:
            skill_response = requests.get(uri)
            skill_response.raise_for_status()
            skill_json = skill_response.json()
            
            # Combine essential and optional occupations
            skill_related_occupations = (skill_json['_links'].get('isEssentialForOccupation', []) +
                                          skill_json['_links'].get('isOptionalForOccupation', []))
            
            for occupation in skill_related_occupations:
                href = occupation.get('href')
                if href:
                    occupation_hrefs.add(href)
        except requests.RequestException as e:
            print(f"Error while fetching skill details: {e}")
                
    for href in occupation_hrefs:
        try:
            occupation_response = requests.get(href)
            occupation_response.raise_for_status()
            occupation_details = occupation_response.json()
            
            code = occupation_details.get('code')
            if code:
                occupation_codes.add(code.split('.')[0])
        except requests.RequestException as e:
            print(f"Error while fetching occupation details: {e}")
            
    for isco_code in occupation_codes:
        kldB_codes = df_lookup[df_lookup['isco08'] == int(isco_code)]['kldb2010'].values
        for code in kldB_codes:
            kldB_set.add(str(code))
    dfs = []
    for kldb in kldB_set:
        berufe = df_berufe[df_berufe['KldB codes']=='B '+kldb]
        dfs.append(berufe)

    merged_df = pd.concat(dfs, ignore_index=True)  
    top_k_berufe = find_similar_occupation(target_occupation_query,merged_df,5,'cosine')
    for beruf in top_k_berufe:
        entry_requirement = beruf.metadata['entry_requirements']
        corrected_json_string = entry_requirement.replace("'", '"')
        entry_requirement_json = json.loads(corrected_json_string)
        for js in entry_requirement_json:
            BA_berufe.add(str(js['data_idref']))
                
    result = get_courses_from_BA(BA_berufe)
    courses = result
    for course in courses['_embedded']['termine']:
        output.append(f"<a href='{course['angebot']['link']}' target='_blank'>{course['angebot']['titel']}</a>") 

    return "<br>".join(output)


def get_occupation_skills(oc_uri):
    #skills_json = r.get(oc_uri)
    skills_json = data_dict.get(oc_uri, None)
    skill_labels = []
    if skills_json:
        skills = json.loads(skills_json)
        for skill in skills:
            german_label = skill['preferredLabel']['de']
            skill_details_mapping[german_label] = skill
            skill_labels.append(german_label)
        return skill_labels
    else:
        return skill_labels
    
def get_occupation_skills_BA(oc_uri):
    df = pd.read_csv("/app/data/berufe_info.csv")
    essential_skills = df[df['id'] == oc_uri]['essential skills'].values
    optional_skills = df[df['id'] == oc_uri]['optional skills'].values
    combined_skills = essential_skills[0][:-1] + ',' + optional_skills[0][1:]
    combined_skills = combined_skills.replace("'", "\"")
    skills = json.loads(combined_skills)
    skill_labels = []
    for skill in skills:
        german_label = skill['skill']
        skill_details_mapping[german_label] = skill
        skill_labels.append(german_label)
    return skill_labels

# Function to update the skills dropdown
def update_skills(occupation):
    oc_uri = occupations.get(occupation, "")
    if isinstance(oc_uri, int):
        skills = get_occupation_skills_BA(oc_uri)
        return gr.Dropdown(skills,label="aktuelle Fähigkeiten", multiselect=True,info='Bitte wählen Sie die Fähigkeiten aus, die Sie derzeit besitzen')
    else:
        skills = get_occupation_skills(oc_uri)
        return gr.Dropdown(skills,label="aktuelle Fähigkeiten", multiselect=True,info='Bitte wählen Sie die Fähigkeiten aus, die Sie derzeit besitzen')
    return 

def update_skillgap(occupation, current_skills):
    oc_uri = occupations.get(occupation, "")
    if isinstance(oc_uri, int):
        ocupation_skills = get_occupation_skills_BA(oc_uri)
    else:
        ocupation_skills = get_occupation_skills(oc_uri)
    skill_gap = [skill for skill in ocupation_skills if skill not in current_skills]
    
    return gr.Dropdown(skill_gap, label="Qualifikationslücke", multiselect=True, info='Bitte wählen Sie die Fähigkeiten aus, die Sie lernen möchten.')

if __name__ == "__main__":
    # Load occupations from CSV
    occupations_esco = get_occupations_from_csv(CSV_FILE_PATH)
    df = pd.read_csv("/app/data/berufe_info.csv")
    occupations_BA = df[['short name', 'id']].set_index('short name').to_dict()['id']
    occupations = {**occupations_esco, **occupations_BA}
    # Gradio interface
    with gr.Blocks(title="MyEduLife Kursempfehlungssystem") as demo:
        with gr.Row():
            with gr.Column():
                occupation_dropdown = gr.Dropdown(list(occupations.keys()), label="Zielberuf",info='Bitte wählen Sie Ihren Zielberuf aus.')
                currentskill_dropdown = gr.Dropdown([],label="aktuelle Fähigkeiten", multiselect=True,info='Bitte wählen Sie die Fähigkeiten aus, die Sie derzeit besitzen')
                sb_btn = gr.Button("Absenden")
                skillgap_dropdown = gr.Dropdown([],label="Fähigkeiten", multiselect=True,info='Bitte wählen Sie die Fähigkeiten aus, die Sie lernen möchten.')
                # Use gr.HTML to display the HTML content
                button = gr.Button("Kursempfehlungen")
            with gr.Column():
                documents_output = gr.HTML()

        occupation_dropdown.change(update_skills, inputs=occupation_dropdown, outputs=currentskill_dropdown)

        sb_btn.click(
                    update_skillgap, 
                    inputs=[occupation_dropdown,currentskill_dropdown], 
                    outputs=skillgap_dropdown
                )

        button.click(
                    retrieve_documents, 
                    inputs=[occupation_dropdown,skillgap_dropdown],
                    outputs=documents_output
                    )
    print('Initialization completed')
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
    
