import json
import glob
import pandas as pd
import os

def process_aw(filepath):
    result_path = os.path.join(filepath, "aw_course")

    # Load Berufenet occupation info
    berufe_info = pd.read_csv(os.path.join(filepath, 'occupation_info.csv'))
    final_df = pd.DataFrame(columns=['occupation_id', 'occupation_name', 'occupation_dsp', 'occupation_skills', 'pre_require_id', 'course_id', 'course_name', 'course_dsp'])

    # Initialize a dictionary to track records for each pre_require_id based on course_name and course_dsp
    unique_tracker = {}

    for filepath in glob.glob(result_path + 'result_*.json'):
        with open(filepath, 'r', encoding='utf-8') as file:
            occupation_id = int(filepath.split('/')[-1].split('_')[1])
            occupation_name = berufe_info[berufe_info['id'] == occupation_id]['short name'].values[0]
            occupation_dsp = berufe_info[berufe_info['id'] == occupation_id]['description'].values[0]
            occupation_skills = berufe_info[berufe_info['id'] == occupation_id]['essential skills'].values[0]
            pre_require_id = filepath.split('/')[-1].split('_')[3]
            datas = json.load(file)
            if '_embedded' in datas:
                for data in datas['_embedded'].get('termine', []):
                    course_id = data['angebot']['id']
                    course_name = data['angebot']['titel']
                    course_dsp = data['angebot']['inhalt']
                    course_link = data['angebot']["link"]
                    course_kw = ''
                    for keyword in data['angebot']['suchworte']:
                        course_kw += ' '+ keyword['suchwort']
                    
                    # Check uniqueness based on course_name and course_dsp
                    unique_key = f"{course_name}_{course_dsp}"
                    
                    # Initialize tracking for this pre_require_id if not already done
                    if pre_require_id not in unique_tracker:
                        unique_tracker[pre_require_id] = set()
                    
                    # Check for uniqueness and limit to 100 records
                    if unique_key not in unique_tracker[pre_require_id] and len(unique_tracker[pre_require_id]) < 100:
                        unique_tracker[pre_require_id].add(unique_key)
                        
                        temp_df = pd.DataFrame({'occupation_id': [occupation_id],
                                                'occupation_name': [occupation_name],
                                                'occupation_dsp': [occupation_dsp ],
                                                'occupation_skills': [occupation_skills],
                                                'pre_require_id': [pre_require_id],
                                                'course_id': [course_id],
                                                'course_name': [course_name],
                                                'course_dsp': [course_dsp + course_kw],
                                                'course_link': [course_link]})
                        final_df = pd.concat([final_df, temp_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['occupation_id', 'course_id'])
    final_df.to_csv(os.path.join(filepath,'occupation_course_info_aw.csv'), index=False)
