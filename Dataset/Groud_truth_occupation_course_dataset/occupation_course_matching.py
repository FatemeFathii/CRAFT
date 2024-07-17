import pandas as pd
import json
df_occupations = pd.read_csv('./Data/data_collection/Berufenet/berufe_info.csv')

final_df = pd.DataFrame(columns=['occupation_id', 'occupation_name', 'occupation_dsp', 'pre_require_id', 'course_id', 'course_name', 'course_dsp'])

# iterate over each row in the DataFrame
for index, row in df_occupations.iterrows():
    occupation_id = row['id']
    occupation_name = row['short name']
    occupation_description = row['description']
    entry_requirement = row['entry requirements']
    corrected_json_string = entry_requirement.replace("'", '"')
    entry_requirement_json = json.loads(corrected_json_string)
    for js in entry_requirement_json:
        course_id = js['data_idref']
        try:
            with open(f'./Data/data_collection/Berufenet/beruf_detail/beruf_{course_id}.json', 'r', encoding='utf-8') as file:
                datas = json.load(file)
                for data in datas:
                    if str(data['id']) == course_id:
                        print(f'Processing course {course_id}')
                        course_name = data['kurzBezeichnungNeutral']
                        for info in data['infofelder']:
                            if info['ueberschrift'] == 'Ausbildungsinhalte' or info['ueberschrift'] == 'Weiterbildungsinhalte':
                                course_dsp = info['content']
                    
                        temp_df = pd.DataFrame({'occupation_id': [occupation_id],
                                                'occupation_name': [occupation_name],
                                                'occupation_dsp': [occupation_description],
                                                'pre_require_id': [course_id],
                                                'course_id': [course_id],
                                                'course_name': [course_name],
                                                'course_dsp': [course_dsp ]})
                        final_df = pd.concat([final_df, temp_df], ignore_index=True)
        except:
            print(f'No file found for beruf_{course_id}.json')
            continue
final_df.to_csv('occupation_course_info_regulation.csv', index=False)