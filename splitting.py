import preprocessing
import cv2 as cv

MIN_DIGIT_HEIGHT = 20
MIN_DIGIT_WIDTH = 10


def get_digit_rectangles(img):
    cntrs, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cntrs = zip(cntrs, [cv.boundingRect(cntr) for cntr in cntrs])

    digit_cntrs = [(x, cntr) for (cntr, (x, _, w, h)) in cntrs if h > MIN_DIGIT_HEIGHT and w > MIN_DIGIT_WIDTH]
    digit_imgs = [(x, preprocessing.contour_to_img(img, cntr)) for (x, cntr) in digit_cntrs]

    digit_imgs = _split_areas_to_five(digit_imgs)

    digit_imgs.sort(key=lambda el: el[0])

    return [dimg for _, dimg in digit_imgs]


def _split_areas_to_five(areas):
    if len(areas) == 0:
        return []

    if len(areas) >= 5:
        return areas

    areas.sort(key=lambda el: el[1].shape[1], reverse=True)

    result = []

    divide_num = 5 - len(areas)
    for idx, area in enumerate(areas):
        if divide_num == 0:
            result.append(area)
        elif divide_num > 1 and idx + 1 < len(areas) and areas[idx + 1][1].shape[1] * 2 > area[1].shape[1]:
            result += _split_img(area, divide_num)
            divide_num = 1
        else:
            result += _split_img(area, divide_num + 1)
            divide_num = 0

    return result


def _split_img(x_img, split):
    x_pos, img = x_img

    preprocessing.vertical_thickness_threshold(img, 6)

    width = img.shape[1]
    width_step = width // split
    x = 0

    # Split evenly
    add_rem_after = split - (width % split)
    even_areas = []
    for i in range(split):
        rem = 0
        if add_rem_after == 0:
            rem = 1
        else:
            add_rem_after -= 1

        digit_img = img[:, x:x + width_step + rem]

        even_areas.append((x_pos, digit_img))
        x_pos += 1
        x += width_step + rem

    return even_areas
