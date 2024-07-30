import fire
import pandas as pd
import json
import os

def generate_corpora(*files, output):
    # List to store all the json lines
    json_lines = []
    
    # Process each file
    for file in files:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Iterate over each column value in the row
            for value in row:
                # Check if the content is text (ignore numeric values)
                if isinstance(value, str):
                    # Create a JSON line and append to the list
                    json_lines.append(json.dumps({'text': value}))
    
    # Write all json lines to a .jsonl file
    with open(os.path.join(output,'edu_corpora.jsonl'), 'w') as f:
        for line in json_lines:
            f.write(line + '\n')

# Main function to integrate with Fire
if __name__ == '__main__':
    fire.Fire(generate_corpora)
