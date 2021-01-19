import json

import cv2
import os
import sys
import numpy as np
import Levenshtein

import img_utils
from table_extractor.Cell import Cell
import ocr_azure.ocr_handwriting as ocrhand

ROOT_DIR = os.path.abspath("../")
RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")
OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output")

MIN_AREA_PERCENTAGE = 0.10
MAX_AREA_PERCENTAGE = 0.50
sys.path.append(ROOT_DIR)
DEBUG = False


def get_table_lines(img, kernel_divisor):
    img_vert = img_hori = img.copy()
    kernel_vert_len = np.array(img).shape[0] // kernel_divisor
    kernel_hori_len = np.array(img).shape[1] // kernel_divisor

    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_vert_len))
    kernel_hori = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_hori_len, 1))

    img_vert = cv2.morphologyEx(img_vert, cv2.MORPH_OPEN, kernel_vert)
    img_hori = cv2.morphologyEx(img_hori, cv2.MORPH_OPEN, kernel_hori)

    img_vert = cv2.morphologyEx(img_vert, cv2.MORPH_CLOSE, kernel_vert)
    img_hori = cv2.morphologyEx(img_hori, cv2.MORPH_CLOSE, kernel_hori)

    return img_vert, img_hori


def extract_data(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                    # Greyscale image
    img_bitwise = cv2.bitwise_not(img_grey)                             # Invert image
    img_thresh = cv2.adaptiveThreshold(img_bitwise, 255,                #
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       15, -2
                                       )

    img_vert, img_hori = get_table_lines(img_thresh, 100)

    img_table_mask = cv2.addWeighted(img_vert, 0.5, img_hori, 0.5, 0.0)

    intersections = cv2.bitwise_and(img_vert, img_hori)
    contours = cv2.findContours(img_table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # img_contours = img.copy()
    # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    tables = []

    img_table = img.copy()
    (height, width, depth) = img.shape
    img_area = height * width

    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area < 10000: # img_area * AREA_PERCENTAGE:
        #     continue
        # print(area)

        poly_curve = cv2.approxPolyDP(contour, 3, True)
        rect = cv2.boundingRect(poly_curve)

        pos_region = intersections[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        intersects = cv2.findContours(pos_region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(intersects) <= 4:
            continue

        table = Cell(rect[0], rect[1], rect[2], rect[3])
        intersect_coords = []
        for i in range(len(intersects)):
            intersect_coords.append(intersects[i][0][0])
        intersect_coords = np.asarray(intersect_coords)

        # Returns indices of coordinates in sorted order
        # Sorts based on parameters (aka keys) starting from the last parameter, then second-to-last, etc
        sorted_indices = np.lexsort((intersect_coords[:, 0], intersect_coords[:, 1]))
        intersect_coords = intersect_coords[sorted_indices]

        table.set_intersects(intersect_coords)
        area = table.w * table.h
        if img_area * MAX_AREA_PERCENTAGE < area:
            continue
        if img_area * MIN_AREA_PERCENTAGE > area:
            continue
        print(area)
        tables.append(table)
        cv2.rectangle(
            img_table,
            (table.x, table.y),
            (table.x + table.w, table.y + table.h),
            (255, 0, 0),
            2
        )

    for table in tables:
        img_cropped = img_bitwise[table.y: table.y + table.h, table.x: table.x + table.w]
        json_analysis = ocrhand.execute(img_cropped)

        if "analyzeResult" not in json_analysis:
            print("No result :(")
            return None

        # Check the first three words
        pos_from = json_analysis["analyzeResult"]["readResults"][0]["lines"][0]["text"]
        pos_desc = json_analysis["analyzeResult"]["readResults"][0]["lines"][2]["text"]
        rat_from = Levenshtein.ratio(pos_from, "FROM")
        rat_desc = Levenshtein.ratio(pos_desc, "DESCRIPTION")

        print("\nValidating...")
        rat_avg = rat_from * 0.5 + rat_desc * 0.5  # Possibly tweak weights, maybe not
        if rat_avg < 0.9:  # Invalid region
            print("Invalid region!")
            # cv2.imshow("Analysis", img_cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            continue

        print("Great Success!")

        # print(json.dumps(json_analysis, indent=4))

        table_vert, table_hori = get_table_lines(img_cropped, 10)

        #cv2.imshow("table_vert_before", table_vert)
        #cv2.imshow("table_hori_before", table_hori)
        #cv2.waitKey(0)

        kernel_vert_len_long = np.array(img).shape[0] // 5
        kernel_hori_len_long = np.array(img).shape[1] // 5
        kernel_vert_len_short = np.array(img).shape[0] // 40
        kernel_hori_len_short = np.array(img).shape[1] // 40

        kernel_vert_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_vert_len_long))
        kernel_hori_long = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_hori_len_long, 1))
        kernel_vert_short = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_vert_len_short))
        kernel_hori_short = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_hori_len_short, 1))

        table_vert = cv2.erode(table_vert, kernel_vert_len_long, iterations=2)
        table_hori = cv2.erode(table_hori, kernel_vert_len_long, iterations=2)


        # cv2.imshow("table_vert_erode", table_vert)
        # cv2.imshow("table_hori_erode", table_hori)
        # cv2.waitKey(0)

        table_vert = cv2.dilate(table_vert, kernel_vert_len_long, iterations=100)
        table_hori = cv2.dilate(table_hori, kernel_vert_len_long, iterations=100)

        #img_vert = cv2.morphologyEx(img_vert, cv2.MORPH_OPEN, kernel_vert)
        #img_hori = cv2.morphologyEx(img_hori, cv2.MORPH_OPEN, kernel_hori)


        #table_vert = cv2.dilate(table_vert, kernel_vert, iterations=2)
        #table_hori = cv2.dilate(table_hori, kernel_hori, iterations=2)


        table_mask = cv2.addWeighted(table_vert, 0.5, table_hori, 0.5, 0.0)

        # cv2.imshow("img_cropped", img_cropped)
        # cv2.imshow("table_vert", table_vert)
        # cv2.imshow("table_hori", table_hori)
        # cv2.imshow("table_mask", table_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        table_mask = cv2.erode(~table_mask, kernel, iterations=2)
        thresh, img_vh = cv2.threshold(table_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
        # idx = 0

        img_cropped_orig = img[table.y: table.y + table.h, table.x: table.x + table.w]
        imgshow = img_cropped_orig.copy()
        cv2.imshow("All Marked Image!", imgshow)
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # idx += 1
            if (w > 20 and h > 20):
                cv2.rectangle(imgshow, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #cv2.imshow("All Marked Image!", imgshow)
                #cv2.imshow("Marked Image!", ~img_cropped[y:y+h, x:x+w])
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
        cv2.imshow("All Marked Image!", imgshow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    """
    for table in tables:
        cropped = cv2.adaptiveThreshold(~cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # cropped = cv2.GaussianBlur(cropped, (5, 5), 0)
        # resized = img_utils.resizeImg(cropped, width=10000)
        text = ocr_textractor.text_from_image(cropped)
        #print(text)
        text = " ".join(text.split()).upper()
        #print(text)

        # cv2.imshow("WINDOW", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if "FROM" not in text or "DESCRIPTION" not in text:
            continue

        cv2.imshow("WINDOW", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

    # cv2.imshow("img Tables", img_utils.resizeImg(img_table, height=1000))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_table
