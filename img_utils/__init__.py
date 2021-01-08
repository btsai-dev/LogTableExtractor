import cv2


def display_table(img, table):
    crop_img = img[table.y: table.y + table.h, table.x: table.x + table.w]

    cv2.imshow("TABLE", resizeImg(crop_img, height=1000))
    cv2.waitKey(0)


def resizeImg(img, width=None, height=None):
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

