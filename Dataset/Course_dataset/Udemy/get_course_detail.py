import pandas as pd
import requests

# Read the course data CSV file
df = pd.read_csv("course_data.csv")

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
output_df.to_csv("course_data_with_details.csv", index=False)
