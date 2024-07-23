import requests
import csv
import os
import pandas as pd
import pandas as pd
from bs4 import BeautifulSoup

def get_course_list(filepath):
    # Set the request URL and headers
    url = "https://www.udemy.com/api-2.0/courses/?page=1&page_size=100&language=de"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": "Basic alNmVm5jMGJZaWpxR3V2bHp1Sk4zZnFhNTl2c0MxRzdKS3FEZGV2NDp4MTFKejAwc2RXWU05Tk9JMnJWaGRvbnJhQ2l3cXZDSndXdkppYVplRUpTbGVsZXo1MFVtMm53eU5Cb20wQ3h1ZUFHdWVFbjFEWWx0eFJqUTRiOVhjUDZoVWxXc3B1eEdJVnF6ZFZ5RktaWFNTVjhic0g5Nk4yUVhVR0hkdDFKRA==",
        "Content-Type": "application/json;charset=utf-8",
    }

    # Set the initial page number and page size
    page_number = 1
    page_size = 100

    # Open a CSV file to write the course data
    with open(os.path.join(filepath,'course_data.csv'), mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['id', 'name', 'url', 'price', 'instructor']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Send GET requests with increasing page number until there are no more results
        while True:
            print(url)
            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                # Extract the course information from the response JSON data
                url = response.json()['next']
                courses = response.json()['results']
                if not courses:
                    break
                
                for course_data in courses:
                    # Extract the course information from the response JSON data
                    course_id = course_data['id']
                    course_name = course_data['title']
                    course_url = course_data['url']
                    course_price = course_data['price']
                    course_instructor = course_data['visible_instructors'][0]['display_name']

                    # Write the course information to the CSV file
                    writer.writerow({'id': course_id, 'name': course_name, 'url': course_url, 'price': course_price, 'instructor': course_instructor})
                
                # Increase the page number for the next request
                page_number += 1
            else:
                # Print an error message if the request was unsuccessful
                print("Error:", response.status_code)
                break


def get_course_detail(filepath):
    # Read the course data CSV file
    df = pd.read_csv(os.path.join(filepath,'course_data.csv'))

    # Set the headers for the Udemy API request
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": "Basic alNmVm5jMGJZaWpxR3V2bHp1Sk4zZnFhNTl2c0MxRzdKS3FEZGV2NDp4MTFKejAwc2RXWU05Tk9JMnJWaGRvbnJhQ2l3cXZDSndXdkppYVplRUpTbGVsZXo1MFVtMm53eU5Cb20wQ3h1ZUFHdWVFbjFEWWx0eFJqUTRiOVhjUDZoVWxXc3B1eEdJVnF6ZFZ5RktaWFNTVjhic0g5Nk4yUVhVR0hkdDFKRA==",
        "Content-Type": "application/json;charset=utf-8",
    }

    # Loop over the rows in the DataFrame and make a request for each course ID
    results = []
    for index, row in df.iterrows():
        # Replace the course ID in the URL with the ID from the current row
        url = f"https://www.udemy.com/api-2.0/courses/{row['id']}/?fields[course]=@all"
        # Make the request
        response = requests.get(url, headers=headers)
        # Check if the request was successful and extract the course information
        if response.status_code == 200:
            course_info = response.json()
            results.append(course_info)
        else:
            print(f"Error getting course {row['id']}: {response.status_code}")

    # Save the results to a new CSV file
    output_df = pd.DataFrame(results)
    output_df.to_csv(os.path.join(filepath,"course_data_with_details.csv"), index=False)

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




def data_preprocess(filepath):

    # Read the CSV file
    df = pd.read_csv(os.path.join(filepath,'course_data_with_details.csv'))

    # Initialize columns in the new DataFrame
    df_new = pd.DataFrame()
    df_new['ID'] = df['id']
    df_new['Content'] = df.apply(process_text_udemy, axis=1)
    df_new['Title'] = df['title']
    df_new['Link'] = df['url']


    # Save the new DataFrame to a CSV file
    df_new.to_csv(os.path.join(filepath,'Udemy_courses.csv'), index=False)

def udemy_get_courses(filepath):
    get_course_list(filepath)
    get_course_detail(filepath)
    data_preprocess(filepath)