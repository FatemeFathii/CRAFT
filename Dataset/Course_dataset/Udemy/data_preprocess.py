import pandas as pd
from bs4 import BeautifulSoup
import spacy
import re
import tokenizer_function
# Define the function to process text
def process_text_udemy(row):
    if pd.isna(row['description']):
        content = ""
    else:
        # remove HTML tags
        content = BeautifulSoup(row['description'], "html.parser").get_text(separator='. ')
    
    # add keywords
    full_text = row['title'] + ". " + row['headline'] + ". " + row['objectives'] + ". " + content
    return full_text


# Read the CSV file
df = pd.read_csv('course_data_with_details.csv')

# Initialize columns in the new DataFrame
df_new = pd.DataFrame()
df_new['course_id'] = df['id']
df_new['course_content'] = df.apply(process_text_udemy, axis=1)
df_new['title'] = df['title']
df_new['url'] = df['url']


# Save the new DataFrame to a CSV file
df_new.to_csv('Udemy_courses.csv', index=False)
