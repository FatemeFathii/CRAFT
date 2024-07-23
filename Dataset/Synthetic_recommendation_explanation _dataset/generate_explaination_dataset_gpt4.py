import pandas as pd
import json
import os
# Load the DataFrames
df_1 = pd.read_csv('./escoai/data/udemy_courses_final.csv')
df_2 = pd.read_csv('./model_training/training_data_balanced/all_course_info.csv')
# Select relevant columns and rename them for consistency
df_1 = df_1[['course_id', 'title', 'course_skills']].rename(columns={'title': 'course_name', 'course_skills': 'course_ctnski_edu'})
df_2 = df_2[['course_id', 'course_name', 'course_skills_edu']]
df = pd.concat([df_1, df_2])
df_o = pd.read_csv('./Data/data_collection/Berufenet/berufe_info.csv')

# Prepare the JSON file content
json_entries = []
# Define the directory containing the CSV files
directory = './explainable_recommendation/Data/lora_data'
dfs = []
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        dfs.append(pd.read_csv(file_path))

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

for index, review in combined_df.iterrows():
    # Find the matching course by 'course_id'
    course = df[df['course_id'] == review['course_id']].iloc[0]
    occupation_name = df_o[df_o['id'] == review['occupation_id']]['short name'].values
    essential_skills = df_o[df_o['id'] == review['occupation_id']]['essential skills'].values
    combined_skills = essential_skills[0]
    combined_skills = combined_skills.replace("'", "\"")
    combined_skills = combined_skills.replace("Horsd\"oeuvr", "Horsd'oeuvr")
    print(combined_skills)
    skills = json.loads(combined_skills)
    skill_labels = []
    for skill in skills:
        german_label = skill['skill']
        skill_labels.append(german_label)
    input_text = f"target occupation: {occupation_name[0]} Skills Gap: {','.join(skill_labels)}\n courses: name: {course['course_name']}, learning objectives: {course['course_skills_edu']}"

    # Create the JSON entry
    json_entry = {
        "instruction": "As an education expert, you have been provided with target occupations and recommended course information. Your task is to explain the recommendation in German.",
        "input": input_text,
        "output": review['respond']
    }
    json_entries.append(json_entry)

# Write the JSON entries to a file
with open('./explainable_recommendation/Data/train_llama_gpt.json', 'w') as f:
    json.dump(json_entries, f)
