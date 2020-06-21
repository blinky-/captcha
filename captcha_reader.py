import cv2 as cv
import numpy as np
import splitting
import requests
import preprocessing
import recognition


def recognize_from_url(url: str) -> str:
    return _recognize(_fetch_from_url(url))


def recognize_from_file(filename: str) -> str:
    return _recognize(cv.imread(filename, cv.IMREAD_GRAYSCALE))


def _recognize(img) -> str:
    preprocessed = preprocessing.preprocess(img)
    digit_imgs = splitting.get_digit_rectangles(preprocessed)
    digit_imgs = [preprocessing.preprocess_digit_img(dimg) for dimg in digit_imgs]

    return ''.join(recognition.match_templates(digit_rect) for digit_rect in digit_imgs)


def _fetch_from_url(url: str):
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'User-Agent': 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    r = requests.get(url, headers=headers)
    img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
    return cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
