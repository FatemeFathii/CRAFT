import fire
from berufenet import get_berufenet
from collector import get_aw_courses
from occupation_course_matching import get_ab_courses
from preprocess import process_aw
import os
import pandas as pd
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.pipelines.token_classification import TokenClassificationPipeline

def ensure_directory_exists(directory):
    """Ensure directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class EntityMerger:
    """Helper class to make merging of B- I- tokens from hf-transformer easier."""
    def __init__(self, text, label, start, end):
        self.text = text
        self.label = label.replace("B-", "")
        self.start = start
        self.end = end
    
    def __repr__(self) -> str:
        return f"<Entity {self.label} {self.text[self.start:self.end]}>"

    def append_hf_tok(self, tok: dict) -> None:
        self.end = tok['end']


def skill_extractor(text: str, model_name):

    # The entities that come out of hf_model are ordered, so we can loop like this

    tfm_model: TokenClassificationPipeline = pipeline("ner", model=model_name)
    entities = []
    current = None
    for ex in tfm_model(text):
        if ex['entity'][0] == 'B':
            if current: 
                entities.append(current)
            current = EntityMerger(text=text, label=ex['entity'], start=ex['start'], end=ex['end'])
        else:
            if current is not None:
                # Theoretically the model could output an `I` without a `B` before. 
                # Just to make sure that isn't hapening at the start we add this if.
                current.append_hf_tok(ex)
    if current is not None:
        entities.append(current)
    return ','.join([str(e.text[e.start:e.end]) for e in entities])

    

def process_data(filepath='./Dataset/Groud_truth_occupation_course_dataset/data', model_name='wt3639/NER_skill_extractor'):
    """
    Process and merge data from AW and AB sources, extract skills using NER.
    
    Args:
    filepath (str): Path to the dataset file.
    model_name (str): Model identifier for the NER model.
    """
    # Load and process datasets
    get_berufenet(filepath)
    get_aw_courses(filepath)
    process_aw(filepath)
    get_ab_courses(filepath)
    data1 = pd.read_csv(os.path.join(filepath,'occupation_course_info_aw.csv'))
    data2 = pd.read_csv(os.path.join(filepath,'occupation_course_info_ab.csv'))
    
    # Merge data ensuring at least 5 records per occupation_id
    complete_data = data2.copy()
    for occ_id in complete_data['occupation_id'].unique():
        if len(complete_data[complete_data['occupation_id'] == occ_id]) < 5:
            additional_rows_needed = 5 - len(complete_data[complete_data['occupation_id'] == occ_id])
            additional_rows = data1[data1['occupation_id'] == occ_id].head(additional_rows_needed)
            complete_data = pd.concat([complete_data, additional_rows], ignore_index=True)
    
    # Extract skills and add as a new column
    complete_data['course_skills'] = complete_data['course_dsp'].apply(lambda x: skill_extractor(x, model_name))
    # Save merged dataset
    # columns: ['occupation_id', 'occupation_name', 'occupation_dsp', 'occupation_skills', 'pre_require_id', 'course_id', 'course_name', 'course_dsp', 'course_skills']
    complete_data.to_csv(os.path.join(filepath,'occupation_course_info.csv'), index=False)

    # Save another version with only specific columns and removed duplicates
    course_info = complete_data[['course_id', 'course_name', 'course_dsp', 'course_skills']].drop_duplicates()
    course_info.to_csv(os.path.join(filepath,'course_info.csv'), index=False)

if __name__ == '__main__':
    fire.Fire(process_data)
