import cv2 as cv
import os
import numpy as np


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


def match_templates(digit_img) -> str:
    result = [(n, _get_similarity_rate(digit_img, template)) for n, template in _templates.items()]

    return max(result, key=lambda el: el[1])[0]


def _get_similarity_rate(digit_img, template) -> float:
    resized = cv.resize(digit_img, (template.shape[1], template.shape[0]))
    subtracted = cv.subtract(resized, template)

    common_rate = np.sum(cv.bitwise_and(template, resized)) / np.sum(template)
    excess_rate = np.sum(subtracted) / np.sum(resized)

    return common_rate - excess_rate
