import captcha_reader

url = 'https://rosreestr.ru/wps/portal/p/cc_ib_portal_services/online_request/!ut/p/z1' \
      '/pZDPCoJAEIefpQeImdks6eBh2cL_imFle5FFSBdKl5AOPX3rA6SH5jbM94P5fiChAtmrt27VqIdePex' \
      '-k7vaL5wjCYdivyAXechj2pCPWG7hOgvkBPKfvAWmPP4YjhAtAdaAvVKRtiCNGru17u8DVMYeQunWSAE' \
      'n7rA4P5R75KcNXpKMMUTyxOxfXhY1yoxNpzzrIGc1BVsAppqWPMzzXH2SAHXIV1-6hWlh/p0/IZ7_01H' \
      'A1A42KODT90AR30VLN22001=CZ6_GQ4E1C41KGQ170AIAK131G00T5=NJcaptcha=/?refresh=true&time=1592737752036'


def main():
    run = True
    while run:
        text, src_img, split_img = captcha_reader.recognize(url)

        print(text)

        key = captcha_reader.show_images_and_wait_for_key([src_img, split_img])
        if key == 27:
            run = False


if __name__ == '__main__':
    main()
