import pandas as pd
import glob
from bs4 import BeautifulSoup

# Set the folder path where the CSV files are located
folder_path = 'data_csv'  # Replace with your folder path

# Read all CSV files in the folder
all_files = glob.glob(folder_path + "/*.csv")

# Create an empty DataFrame to hold the merged data
all_data = pd.DataFrame()

# Read each file one by one
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    # Append data to the all_data DataFrame
    all_data = pd.concat([all_data, df])

# Remove duplicates based on each column individually
for column in ['ID', 'Title', 'Content', 'Keywords']:
    all_data = all_data.drop_duplicates(subset=column, keep='first')

# Clean HTML content in the 'Content' column after removing duplicates
all_data['Content'] = all_data['Content'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
all_data['Content'] = all_data['Content'] + ' ' + all_data['Keywords']
all_data.drop(columns=['Keywords'], inplace=True)
# Save to a new CSV file
all_data.to_csv('AW_courses.csv', index=False)
