import json
import os
import cv2
import time
import table_extractor as tabex
import ocr_azure.ocr_handwriting as ocrhand
import ocr_azure.ocr_printed as ocrprint

ROOT_DIR = os.path.abspath("../")
RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")
WELL_LOG_DIR = os.path.join(RESOURCE_DIR, "logs")
OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output")
BOXES_DIR = os.path.join(RESOURCE_DIR, "boxes")
COMBO_DIR = os.path.join(RESOURCE_DIR, "combo")
PROBLEM_DIR = os.path.join(RESOURCE_DIR, "problem_children")


LOG_DIVISOR = os.path.join(RESOURCE_DIR, "logs - Copy")
LOGS_01 = os.path.join(LOG_DIVISOR, "1")
LOGS_02 = os.path.join(LOG_DIVISOR, "2")
LOGS_03 = os.path.join(LOG_DIVISOR, "3")
LOGS_04 = os.path.join(LOG_DIVISOR, "4")
LOGS_05 = os.path.join(LOG_DIVISOR, "5")
LOGS_06 = os.path.join(LOG_DIVISOR, "6")
LOGS_07 = os.path.join(LOG_DIVISOR, "7")
LOGS_08 = os.path.join(LOG_DIVISOR, "8")
LOGS_09 = os.path.join(LOG_DIVISOR, "9")
LOGS_10 = os.path.join(LOG_DIVISOR, "10")
LOGS_11 = os.path.join(LOG_DIVISOR, "11")


def datatable_extract(img):
    return tabex.extract_data(img)


def scan():
    count = 0
    for entry in os.scandir(LOGS_01):
        if (entry.path.endswith(".tif")) and entry.is_file():
            path = entry.path
            name = os.path.splitext(entry.name)[0]
            print("Analyzing image: ", name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            datatable_extract(img)
            print("\n\n\n")
            # count += 1
            # if count > 30:
            #     break

if __name__ == "__main__":
    json_analysis = scan()
    # time.sleep(3)
    # print("Beginning processing!")
    # path = "C:\\Users\\Godonan\\Pictures\\testing.png"
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # datatable_extract(img)
    # ocrhand.execute(img)
    # ocrprint.exec_orig()

