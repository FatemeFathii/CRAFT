import pandas as pd
import json
import os
import csv
from io import StringIO
import os
from openai import OpenAI
#from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
import fire



def get_explanation_from_gpt(LLM_key, occupation_course_data, filepath):
    runnable = RunnableParallel(
        target_occupation=lambda x: x["target_occupation"],
        skill_gap=lambda x: x["skill_gap"],
        course_rec = lambda x: x["course_rec"]
    )

   
    chat_openai = ChatOpenAI(model="gpt-4o",temperature=0.3,max_tokens=256,api_key=LLM_key)
    template_re = """
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

    ### Instruction:
    As an education expert, you have been provided with information on target occupations and skills gaps, along with recommended course details. Your task is to explain the recommendation in German in one paragraph, focusing on how the course's learning outcomes and target skills relate to the identified skills gaps.

    ### Input:
    Target Occupation: {target_occupation}
    Skills Gap: {skill_gap}
    Recommended course: {course_rec}

    ### Response:
    """

    prompt_re = ChatPromptTemplate.from_template(template_re)
    chain_re = (
        runnable
        | prompt_re
        | chat_openai
        | StrOutputParser()
    )

    df = pd.read_csv(occupation_course_data)
   

    json_entries = []
    for index, row in df.iterrows():
        final_result = []
        respond = chain_re.invoke({
                    "course_rec": 'name: ' + row['course_name']  + ' learning outcomes: ' + row['course_skills'] ,  
                    "target_occupation": row['occupation_name'],
                    "skill_gap": row['occupation_skills'],
                })

            
        final_result.append({
                            'course_id': row['course_id'],  # 'course_id' corresponds to the header in the CSV
                            'occupation_id': row['occupation_id'],  # Assuming occupation_id is passed or available here
                            'respond' : respond
                        })
        input_text = f"target occupation: {row['occupation_name']} Skills Gap: {row['occupation_skills']}\n courses: name: {row['course_name']}, learning objectives: {row['course_skills']}"

        # Create the JSON entry
        json_entry = {
            "instruction": "As an education expert, you have been provided with target occupations and recommended course information. Your task is to explain the recommendation in German.",
            "input": input_text,
            "output": respond
        }
        json_entries.append(json_entry)

    # Write the JSON entries to a file
    with open(os.path.join(filepath,'train_llama_gpt.json'), 'w') as f:
        json.dump(json_entries, f)

    df_results = pd.DataFrame(final_result)



    df_results.to_csv(os.path.join(filepath,'explanation_training_data.csv'), index=False)



if __name__ == "__main__":
    fire.Fire(get_explanation_from_gpt)