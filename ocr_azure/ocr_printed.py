import json
import os
import sys
import time
import cv2
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO

MAX_RETRIES = 10

def get_authy():
    endpoint = None
    key = None
    if 'AZURE_COMPUTER_VISION_ENDPOINT' in os.environ:
        endpoint = os.environ['AZURE_COMPUTER_VISION_ENDPOINT']
    if 'AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
        key = os.environ['AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY']
    if endpoint is None:
        print("ENDPOINT system environmental variable not found.")
        sys.exit()
    if key is None:
        print("KEY system environmental variable not found.")
        sys.exit()
    return endpoint, key


def exec(img):
    # Get authentication details
    ENDPOINT, KEY = get_authy()
    endpoint_url = ENDPOINT + "vision/v3.1/read/analyze"

    # Convert image to binary data
    success, data = cv2.imencode(".jpg", img)
    data = data.tobytes()

    headers = {
        'Ocp-Apim-Subscription-Key': KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {'language': 'en'}
    ready = False
    response = None
    retries = 0
    while response is None or ready is False:
        try:
            response = requests.post(
                url=endpoint_url,
                headers=headers,
                params=params,
                data=data
            )
            response.raise_for_status()
            if response.status_code == 429:
                print("Message: %s" % (response.json()))
                if retries <= MAX_RETRIES:
                    time.sleep(1)
                    retries += 1
                    continue
                else:
                    print('Error: failed after retrying!')
                    ready = True
            elif response.status_code == 200:
                ready = True
            else:
                print("Error code: %d" % (response.status_code))
                print("Message: %s" % (response.json()))
            ready = True
        except:
            print("Critical error on response. Retrying in 5...")
            time.sleep(5)

        print("Finished with response")
        json_analysis = response.json()
        # print(json.dumps(json_analysis, indent=4))
        print("Analysis Complete")

        if "analyzeResult" in json_analysis:
            print("Successful result, returning.")
            poll = False
        if "status" in json_analysis and json_analysis['status'] == 'failed':
            print("Retrying...")
            # print(json.dumps(json_analysis, indent=4))
            poll = False
        response.close()
    response.close()

    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    for line in line_infos:
        print(line)


def exec_orig():
    # Add your Computer Vision subscription key and endpoint to your environment variables.
    if 'AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
        subscription_key = os.environ['AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY']
    else:
        print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
        sys.exit()

    if 'AZURE_COMPUTER_VISION_ENDPOINT' in os.environ:
        endpoint = os.environ['AZURE_COMPUTER_VISION_ENDPOINT']

    ocr_url = endpoint + "vision/v3.1/ocr"

    # Set image_url to the URL of an image that you want to analyze.
    image_path = "C:\\Users\\Godonan\\Pictures\\testing.png"
    image_data = open(image_path, "rb").read()

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream'
    }
    params = {'language': 'unk', 'detectOrientation': 'true'}
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    print(response)
    response.raise_for_status()
    print(response)

    analysis = response.json()
    print(response)


    # Extract the word bounding boxes and text.
    # print(json.dumps(analysis, indent=4))
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)
    for word in word_infos:
        print(word["text"])