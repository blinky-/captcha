import captcha_reader
import cv2 as cv
import os
import sys

url = "http://example.com"

def run_tests():
    good = 0
    bad = []

    tests_dir = "test_imgs/"
    for imgfile in os.listdir(tests_dir):
        expected = os.path.splitext(imgfile)[0]
        full_filename = os.path.join(tests_dir, imgfile)
        actual = captcha_reader.recognize_from_file(full_filename)

        if actual != expected:
            bad.append(expected)
            print("Actual: " + actual + ", Expected: " + expected)
        else:
            good += 1

    print("Successful: " + str(good) + " \nFailures: " + str(len(bad)) + "\n", bad)


def download():
    cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.resizeWindow("img", 520, 144)

    run = True
    while run:
        img = captcha_reader._fetch_from_url(url)
        cv.imshow("img", img)
        cv.waitKey(1)

        line = sys.stdin.readline()
        line = line.rstrip()
        cv.imwrite("test_imgs/%s.png" % line, img)
        print("Writing %s" % line)


def main():
    # download()
    run_tests()


if __name__ == '__main__':
    main()
