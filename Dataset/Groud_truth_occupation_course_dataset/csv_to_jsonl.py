# get array of courses from courses.csv with columns name,description,url
import json
import pandas as pd
import random
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

def jsonl_preparation(csv_file, output_folder, random_seed=36, num_negatives=14):
    instruction_prompt = """
                    As an education expert, you have been provided with a target occupation, a skill gap, and information on two candidate courses. Your task is to determine which course better matches the target occupation and skill gap. Please respond with 'First' or 'Second' to indicate your recommendation. """

    random.seed(random_seed)
    df = pd.read_csv(csv_file)


    grouped = df.groupby('occupation_id')['course_id'].apply(list).reset_index()


    query_to_pos = dict(zip(grouped['occupation_id'], grouped['course_id']))


    all_pos = set(df['course_id'])


    def generate_negatives(query, pos_texts, all_texts, num_negatives=14):
        possible_negatives = list(all_texts - set(pos_texts))
        return random.sample(possible_negatives, min(num_negatives, len(possible_negatives)))


    final_data = []
    #query_instruction = 'Gib einen Beruf mit den dazugehörigen Fähigkeiten an und empfehle passende Kurse: '
    for query, pos_list in query_to_pos.items():
        neg_list = generate_negatives(query, pos_list, all_pos,num_negatives)
        final_data.append({"query":  query, "pos": pos_list, "neg": neg_list})

  
    retrival_json = []
    json_file = []
    for example in final_data:
        occupation_id = example['query']
        occupation = df[df['occupation_id'] == occupation_id].iloc[0]
        for pos in example['pos']:
            course_pos = df[df['course_id'] == pos].iloc[0]
            for neg in example['neg']:
                course_neg = df[df['course_id'] == neg].iloc[0]
                retrival_json.append({"query":  example['query'], "pos": course_pos['course_skills'], "neg": course_neg['course_skills']})
                if random.choice([True, False]):
                    A, B = course_pos, course_neg
                    output_key = 'First'
                else:
                    A, B = course_neg, course_pos
                    output_key = 'Second'
                # Create the input text
                input_text = (f"target occupation: name: {occupation['occupation_name']} decription: {occupation['occupation_dsp']}"
                            f"skill gap: {occupation['occupation_skills']} "
                            f"candidate courses: First: name: {A['course_name']}, decription:{A['course_content_limited']} "
                            f"Second: name: {B['course_name']}, decription:{B['course_content_limited']}")

                # The positive course will always be referred to by `output_key`
                output = output_key
                entry = {
                "instruction": instruction_prompt,
                "input": input_text,
                "output": output,
                }
                json_file.append(entry)
    with open(os.path.join(output_folder,'LLM_data.json'), 'w') as f:
        json.dump(json_file, f)
    with open(os.path.join(output_folder,'retriever_data.jsonl'), 'w') as f:
        for data in retrival_json:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            # Convert the string response to a StringIO object for reading as file-like object
       


   

