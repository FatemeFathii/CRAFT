import pandas as pd
import numpy as np
import os
import fire

def create_and_divide_jsonl_from_csv(paths, fractions, num_files=8, filepath='./Dataset/Skill_entities_annotation_dataset/data'):
    if sum(fractions) != 1:
        raise ValueError("The sum of sampling fractions must be 1.")
    if len(paths) != len(fractions):
        raise ValueError("The number of paths and fractions must be equal.")
    
    combined_samples = pd.DataFrame()
    
    # Load and sample data from each file according to the specified fractions
    for path, fraction in zip(paths, fractions):
        data = pd.read_csv(path)
        sample = data.sample(frac=fraction, random_state=1)  # Ensure reproducibility
        combined_samples = pd.concat([combined_samples, sample])
    
    # Determine the size of each split part
    part_size = int(np.ceil(len(combined_samples) / num_files))
    
    # Split data into num_files parts and create separate JSONL files
    for i in range(num_files):
        part = combined_samples.iloc[i * part_size:(i + 1) * part_size]
        jsonl_strings = part['Content'].apply(lambda x: f'{{"text": "{x}"}}')
        
        output_path = f'course_description_{i+1}.jsonl'
        with open(os.path.join(filepath,output_path), 'w') as file:
            for item in jsonl_strings:
                file.write(item + '\n')
        
        print(f'Part {i+1} JSONL file created at {output_path}')


if __name__ == "__main__":
    fire.Fire(create_and_divide_jsonl_from_csv)
