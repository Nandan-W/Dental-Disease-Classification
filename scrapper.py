import os
import requests
from googleapiclient.discovery import build

api_key = ""
cse_id = ""

def google_image_search(query, num_results=10, start_index=1):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, searchType='image', num=num_results, start=start_index).execute()
    return res['items']

def download_image(url, folder_path, image_name):
    try:
        img_data = requests.get(url).content
        with open(os.path.join(folder_path, image_name), 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded {image_name}")
    except Exception as e:
        print(f"Failed to download {image_name}: {e}")

query = "healthy human teeth"
total_images_needed = 950
images_per_query = 10
queries_needed = total_images_needed // images_per_query  

folder_path = "images"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

downloaded_urls = set()

downloaded_images = 0
start_index = 10  

for query_num in range(queries_needed):
    images = google_image_search(query, num_results=images_per_query, start_index=start_index)
    for idx, image in enumerate(images):
        image_url = image['link']

        # Check if this URL has already been downloaded
        if image_url not in downloaded_urls:
            image_name = f"healthy_teeth_{downloaded_images + idx + 1}.jpg"
            download_image(image_url, folder_path, image_name)
            
            downloaded_urls.add(image_url)
        
    downloaded_images += len(images)
    start_index += images_per_query  
    print(f"Downloaded {downloaded_images} images out of {total_images_needed}")

if downloaded_images < total_images_needed:
    remaining_images = total_images_needed - downloaded_images
    additional_queries = (remaining_images + images_per_query - 1) // images_per_query  

    for _ in range(additional_queries):
        images = google_image_search(query, num_results=images_per_query, start_index=start_index)
        for idx, image in enumerate(images):
            image_url = image['link']

            # Check if this URL has already been downloaded
            if image_url not in downloaded_urls:
                image_name = f"healthy_teeth_{downloaded_images + idx + 1}.jpg"
                download_image(image_url, folder_path, image_name)
                
                # Add the URL to the set of downloaded URLs
                downloaded_urls.add(image_url)
        
        downloaded_images += len(images)
        start_index += images_per_query  
        print(f"Downloaded {downloaded_images} images out of {total_images_needed}")
