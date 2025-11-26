import requests

url = "http://192.168.1.165:8000/predict/"
image_path = r"C:\Users\Glaesha\Downloads\images (18).jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
