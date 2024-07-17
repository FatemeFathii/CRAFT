import requests
import csv

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
with open('course_data.csv', mode='w', newline='', encoding='utf-8') as csv_file:
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
