import cv2 as cv
import numpy as np
import os
import requests
from collections import namedtuple
from typing import List
from typing import Any

Rect = namedtuple("Rect", "x y w h")


def recognize(url: str) -> (str, Any, Any):
    img = _fetch_from_url(url)
    preprocessed = _preproc(img.copy())
    digit_imgs = _get_digit_rectangles(preprocessed)

    return ''.join(_match_templates(digit_rect) for digit_rect in digit_imgs), img, preprocessed


def show_images_and_wait_for_key(images) -> int:
    for i, img in enumerate(images):
        cv.imshow('%d' % i, img)

    return cv.waitKey(0)


def _fetch_from_url(url: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36'
    }

    r = requests.get(url, headers=headers)
    img_array = np.asarray(bytearray(r.content), dtype=np.uint8)

    return cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)


def _load_templates():
    templates = {}
    templates_directory = 'digit_templates/'
    for file in os.listdir(templates_directory):
        if file.endswith('.png'):
            templates[os.path.splitext(file)[0]] = cv.imread(
                os.path.join(templates_directory, file),
                cv.IMREAD_GRAYSCALE
            )

    return templates


_templates = _load_templates()


def _match_templates(digit_img) -> str:
    result = []
    for n, template in _templates.items():
        result.append((n, _get_similarity_rate(digit_img, template)))

    result.sort(key=lambda el: el[1])

    return result[-1][0]


def _get_similarity_rate(digit_img, template) -> float:
    resized = cv.resize(digit_img, (template.shape[1], template.shape[0]))
    subtracted = cv.subtract(resized, template)

    common_rate = np.sum(cv.bitwise_and(template, resized)) / np.sum(template)
    excess_rate = np.sum(subtracted) / np.sum(resized)

    return common_rate - excess_rate


def _preproc(img):
    out = cv.inRange(img, 0, 1)
    out = out[7:36, 25:130]
    out = cv.resize(out, (0, 0), fx=6.0, fy=2.0)

    kernel = np.ones((3, 3), np.float)
    kernel /= 9

    out = cv.filter2D(out, -1, kernel)
    _, out = cv.threshold(out, 195, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    out = cv.erode(out, kernel, iterations=1)
    _, out = cv.threshold(out, 195, 255, cv.THRESH_BINARY)

    x_thickness = cv.reduce(out, 0, cv.REDUCE_AVG, dtype=cv.CV_32S)
    for idx, h in enumerate(x_thickness[0]):
        if h < 18:
            out[:, idx] = 0

    return out


def _get_digit_rectangles(img):
    cntrs, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    half_height = img.shape[0] / 2

    areas = []
    for contour in cntrs:
        (x, y, w, h) = cv.boundingRect(contour)
        rect = Rect(x, y, w, h)
        if rect.h > half_height:
            areas.append(rect)

    areas = _split_areas(areas)

    areas.sort(key=lambda el: el[0])

    for (x, y, w, h) in areas:
        cv.rectangle(img, (x, y), (x + w, y + h), (127, 127, 127), 1)

    return [img[y:y + h, x:x + w] for (x, y, w, h) in areas]


def _split_areas(areas: List[Rect]) -> List[Rect]:
    if len(areas) >= 5:
        return areas

    areas.sort(key=lambda el: el.x)
    avg_digit_size = (areas[-1].x + areas[-1].w - areas[0].x) / 5
    avg_digit_size *= 1.4

    areas.sort(key=lambda el: el.w)

    split_digits = []
    for rect in areas:
        if rect.w > avg_digit_size:
            if len(areas) == 1:
                split_digits += _split_area(rect, 5)
            elif len(areas) == 2:
                if len(split_digits) == 1:
                    split_digits += _split_area(rect, 4)
                elif len(split_digits) == 2:
                    split_digits += _split_area(rect, 3)
                else:
                    split_digits += _split_area(rect, 2)
            elif len(areas) == 3 and len(split_digits) == 2:
                split_digits += _split_area(rect, 3)
            else:
                split_digits += _split_area(rect, 2)
        else:
            split_digits.append(rect)

    return split_digits


def _split_area(rect: Rect, split: int) -> List[Rect]:
    new_width = rect.w // split
    x = rect.x

    result = []
    for i in range(split):
        result.append((x, rect.y, new_width, rect.h))
        x += new_width

    return result
