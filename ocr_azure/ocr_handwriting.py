import json
import os
import sys
import requests
import time
import cv2

import traceback

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


def execute(img):
    # Get authentication details
    ENDPOINT, KEY = get_authy()
    endpoint_url = ENDPOINT + "vision/v3.1/read/analyze"

    # Convert image to binary data
    success, data = cv2.imencode(".jpg", img)
    data = data.tobytes()

    # Set up headers
    headers = {
        'Ocp-Apim-Subscription-Key': KEY,
        'Content-Type': 'application/octet-stream'
    }

    ready = False
    response = None
    while response is None or ready is False:
        try:
            response = requests.post(
                url=endpoint_url,
                headers=headers,
                data=data
            )
            if response.status_code != requests.codes.accepted:
                print("Response NOT ACCEPTED")
                if response.status_code == requests.codes.too_many_requests:
                    print("Too many requests. Waiting.")
                else:
                    print("Response code: ", response.status_code)
                time.sleep(3)
                print("Retrying...")
                continue
            ready = True
        except Exception as e:
            print("Critical error on response POST. Retrying in 5...")
            print(e)
            # traceback.print_exc()
            time.sleep(5)


    # print("Raising server status.")
    # Server status details
    #

    # operation_url = response.headers["Operation-Location"]
    # print(operation_url)

    # The recognized text isn't immediately available, so poll to wait for completion.
    poll = True
    print("Beginning poll")
    counter = 0
    json_analysis = None
    while poll:
        counter += 1
        print("Polling attempt #", counter)
        time.sleep(2)
        print("Sleep finished")
        response_final = None
        while response_final is None:
            try:
                response_final = requests.get(
                    response.headers["Operation-Location"],
                    headers=headers,
                    timeout=5
                )
                if response_final.status_code != requests.codes.ok:
                    print("Response NOT OK")
                    if response_final.status_code == requests.codes.too_many_requests:
                        print("Too many requests. Waiting.")
                    else:
                        print("Response code: ", response_final.status_code)
                    time.sleep(3)
                    print("Retrying...")
                    continue
            except Exception as e:
                print(e)
                # traceback.print_exc()
                print("Critical error on response GET. Retrying in 5...")
                time.sleep(5)

        print("Finished with response")
        json_analysis = response_final.json()
        # print(json.dumps(json_analysis, indent=4))
        print("Analysis Complete")

        if "analyzeResult" in json_analysis:
            print("Successful result, returning.")
            poll = False
        if "status" in json_analysis and json_analysis['status'] == 'failed':
            print("Retrying...")
            poll = False
        response_final.close()
    response.close()


    # polygons = []
    # if "analyzeResult" in json_analysis:
    #     # Extract the recognized text, with bounding boxes.
    #     polygons = [(line["boundingBox"], line["text"])
    #                 for line in analysis["analyzeResult"]["readResults"][0]["lines"]]

    return json_analysis
