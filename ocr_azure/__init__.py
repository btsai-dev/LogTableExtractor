import json
import os
import sys
import requests
import time
import cv2
import numpy as np
# If you are using a Jupyter Notebook, uncomment the following line.
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image
from io import BytesIO

SUBSCRIPTION_KEY = None
ENDPOINT = None
MAX_RETRIES = 10

missing_env = False

if 'AZURE_COMPUTER_VISION_ENDPOINT' in os.environ:
    ENDPOINT = os.environ['AZURE_COMPUTER_VISION_ENDPOINT'] + "vision/v3.0/ocr"
else:
    print("From Azure Cognitive Service, retrieve your endpoint and subscription key.")
    print(
        "\nSet the COMPUTER_VISION_ENDPOINT environment variable, such as \"https://westus2.api.cognitive.microsoft.com\".\n")
    missing_env = True

if 'AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    SUBSCRIPTION_KEY = os.environ['AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("From Azure Cognitive Service, retrieve your endpoint and subscription key.")
    print(
        "\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable, such as \"1234567890abcdef1234567890abcdef\".\n")
    missing_env = True

if missing_env:
    print("**Restart your shell or IDE for changes to take effect.**")
    sys.exit()

def ocr_handwriting(img):
    def processRequest(json, data, headers, params):
        """
        Helper function to process the request to Project Oxford

        Parameters:
        json: Used when processing images from its URL. See API Documentation
        data: Used when processing image read from disk. See API Documentation
        headers: Used to pass the key information and the data type request
        """

        retries = 0
        result = None

        while True:
            print(ENDPOINT)
            response = requests.request('post', ENDPOINT, json=json, data=data, headers=headers, params=params)

            if response.status_code == 429:
                print("Message: %s" % (response.json()))
                if retries <= MAX_RETRIES:
                    time.sleep(1)
                    retries += 1
                    continue
                else:
                    print('Error: failed after retrying!')
                    break
            elif response.status_code == 202:
                result = response.headers['Operation-Location']
            else:
                print("Error code: %d" % (response.status_code))
                print("Message: %s" % (response.json()))
            break

        return result

    def getOCRTextResult(operationLocation, headers):
        """
        Helper function to get text result from operation location

        Parameters:
        operationLocation: operationLocation to get text result, See API Documentation
        headers: Used to pass the key information
        """

        retries = 0
        result = None

        while True:
            response = requests.request('get', operationLocation, json=None, data=None, headers=headers, params=None)
            if response.status_code == 429:
                print("Message: %s" % (response.json()))
                if retries <= MAX_RETRIES:
                    time.sleep(1)
                    retries += 1
                    continue
                else:
                    print('Error: failed after retrying!')
                    break
            elif response.status_code == 200:
                result = response.json()
            else:
                print("Error code: %d" % (response.status_code))
                print("Message: %s" % (response.json()))
            break

        return result

    def showResultOnImage(result, img):

        """Display the obtained results onto the input image"""
        img = img[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img, aspect='equal')

        lines = result['recognitionResult']['lines']

        for i in range(len(lines)):
            words = lines[i]['words']
            for j in range(len(words)):
                tl = (words[j]['boundingBox'][0], words[j]['boundingBox'][1])
                tr = (words[j]['boundingBox'][2], words[j]['boundingBox'][3])
                br = (words[j]['boundingBox'][4], words[j]['boundingBox'][5])
                bl = (words[j]['boundingBox'][6], words[j]['boundingBox'][7])
                text = words[j]['text']
                x = [tl[0], tr[0], tr[0], br[0], br[0], bl[0], bl[0], tl[0]]
                y = [tl[1], tr[1], tr[1], br[1], br[1], bl[1], bl[1], tl[1]]
                line = Line2D(x, y, linewidth=3.5, color='red')
                ax.add_line(line)
                ax.text(tl[0], tl[1] - 2, '{:s}'.format(text),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.show()

    success, data = cv2.imencode('.png', img)
    data = data.tobytes()

    # Computer Vision parameters
    params = {'mode': 'Handwritten'}

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = SUBSCRIPTION_KEY
    headers['Content-Type'] = 'application/octet-stream'

    json = None

    operationLocation = processRequest(json, data, headers, params)

    result = None
    if (operationLocation != None):
        headers = {}
        headers['Ocp-Apim-Subscription-Key'] = SUBSCRIPTION_KEY
        while True:
            time.sleep(1)
            result = getOCRTextResult(operationLocation, headers)
            if result['status'] == 'Succeeded' or result['status'] == 'Failed':
                break

    # Load the original image, fetched from the URL
    if result is not None and result['status'] == 'Succeeded':
        data8uint = np.fromstring(data, np.uint8)  # Convert string to an unsigned int array
        img = cv2.cvtColor(cv2.imdecode(data8uint, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        showResultOnImage(result, img)
