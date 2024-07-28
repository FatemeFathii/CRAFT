import pandas as pd
import json
import os

def get_ab_courses(filepath):
    df_occupations = pd.read_csv(os.path.join(filepath,'occupation_info.csv'))

    final_df = pd.DataFrame(columns=['occupation_id', 'occupation_name', 'occupation_dsp', 'occupation_skills', 'pre_require_id', 'course_id', 'course_name', 'course_dsp'])

    # iterate over each row in the DataFrame
    for index, row in df_occupations.iterrows():
        occupation_id = row['id']
        occupation_name = row['short name']
        occupation_description = row['description']
        occupation_skills = row['essential skills']
        entry_requirement = row['entry requirements']
        corrected_json_string = entry_requirement.replace("'", '"')
        entry_requirement_json = json.loads(corrected_json_string)
        for js in entry_requirement_json:
            course_id = js['data_idref']
            try:
                with open(os.path.join(filepath,f'beruf_detail/beruf_{course_id}.json'), 'r', encoding='utf-8') as file:
                    datas = json.load(file)
                    for data in datas:
                        if str(data['id']) == course_id:
                            print(f'Processing course {course_id}')
                            course_name = data['kurzBezeichnungNeutral']
                            course_link = 'https://web.arbeitsagentur.de/berufenet/beruf/' + str(course_id)
                            for info in data['infofelder']:
                                if info['ueberschrift'] == 'Ausbildungsinhalte' or info['ueberschrift'] == 'Weiterbildungsinhalte':
                                    course_dsp = info['content']
                        
                            temp_df = pd.DataFrame({'occupation_id': [occupation_id],
                                                    'occupation_name': [occupation_name],
                                                    'occupation_dsp': [occupation_description],
                                                    'occupation_skills': [occupation_skills],
                                                    'pre_require_id': [course_id],
                                                    'course_id': [course_id],
                                                    'course_name': [course_name],
                                                    'course_dsp': [course_dsp ],
                                                    'course_link': [course_link]})
                            final_df = pd.concat([final_df, temp_df], ignore_index=True)
            except:
                print(f'No file found for beruf_{course_id}.json')
                continue
    final_df.to_csv(os.path.join(filepath,'occupation_course_info_ab.csv'), index=False)