import os
import json
import re
import csv


parent_folder = 'result'


for folder_name, subfolders, filenames in os.walk(parent_folder):
   
    match = re.match(r'^result_(\d+)$', os.path.basename(folder_name))
    if not match:
        continue
    
    id = match.group(1)
   
    pattern = r'^result_.*.json$'
    matched_files = [f for f in filenames if re.match(pattern, f)]

    
    written_ids = set()

    with open(f'data_csv/result_{id}.csv', 'w', newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(['ID', 'Title', 'Content', 'Keywords','Link'])
        for file_name in matched_files:
            with open(os.path.join(folder_name, file_name), 'r') as json_file:
                print(file_name)
                data = json.load(json_file)
                
                termine_items = data.get("_embedded", {}).get("termine", [])
                for item in termine_items:
                    offer_data = item["angebot"]
                    id = offer_data["id"]
                   
                    if id in written_ids:
                        continue
                    title = offer_data["titel"]
                    content = offer_data["inhalt"]
                    link = offer_data["link"]
                    keywords = ', '.join([word["suchwort"] for word in offer_data["suchworte"]])
                    csvwriter.writerow([id, title, content, keywords, link])
                    
                    written_ids.add(id)
