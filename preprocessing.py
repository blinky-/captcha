import cv2 as cv
import numpy as np
import splitting


def preprocess(img):
    out = cv.inRange(img, 0, 0)

    # Get area around digits
    from_x = 21
    from_y = 4
    to_x = 130
    to_y = 38
    out = out[from_y:to_y, from_x:to_x]

    out = _rm_thin_lines(out, 2)

    cross_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    out = cv.morphologyEx(out, cv.MORPH_OPEN, cross_kernel)

    out = vertical_thickness_threshold(out, 6)

    return out


def preprocess_digit_img(img):
    cntrs, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cntrs = zip(cntrs, [cv.boundingRect(cntr) for cntr in cntrs])
    max_cntr = max(cntrs, key=lambda el: cv.contourArea(el[0]))

    cntr_height = max_cntr[1][3]
    cntr_width = max_cntr[1][2]
    if cntr_height >= splitting.MIN_DIGIT_HEIGHT and cntr_width >= splitting.MIN_DIGIT_WIDTH:
        img = contour_to_img(img, max_cntr[0])

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img = cv.erode(img, kernel)

    return img


def contour_to_img(img, contour):
    mask = np.zeros_like(img)
    cv.drawContours(mask, [contour], -1, 255, -1)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    x, y, w, h = cv.boundingRect(contour)

    return out[y:y + h, x:x + w]


def _rm_thin_lines(img, threshold):
    width = img.shape[1]
    height = img.shape[0]

    for x in range(width):
        white_count = 0
        y = 0
        for y in range(height):
            if img[y, x] == 255:
                white_count += 1
            else:
                if white_count <= threshold:
                    img[y - white_count:y + 1, x] = 0
                white_count = 0

        if white_count <= threshold:
            img[y - white_count:y + 1, x] = 0

    return img


def vertical_thickness_threshold(img, threshold):
    thickness = cv.reduce(img, 0, cv.REDUCE_SUM, dtype=cv.CV_32S) / 255
    img[:, thickness[0] < threshold] = 0
    return img
