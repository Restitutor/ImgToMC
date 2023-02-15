import gzip
import requests
import numpy as np
import cv2 as cv


def readUrl(url):
    return requests.get(url, stream=True).raw.read()


def getImage(arr):
    return cv.imdecode(np.asarray(bytearray(arr), dtype=np.uint8), cv.IMREAD_COLOR)


def resizeImage(image, height=None, width=160):
    if height is None:
        height = round((width / image.shape[1] * image.shape[0]) / 5)
        if height >= 32:
            height = 31

    return cv.resize(
        image,
        (width, height),
        cv.INTER_LANCZOS4,
    )


def makeRGBArray(img):
    for row in img:
        for pixel in row:
            yield reversed(pixel)


def hexify(colorList):
    for color in colorList:
        yield ("{:02X}" * 3).format(*color)


def compactOut(hexList):
    return "".join(hexList)


def onfimHandler(arr):
    return gzip.compress(
        compactOut(hexify(makeRGBArray(resizeImage(getImage(arr), height=20)))).encode(
            "utf-8"
        )
    )


def generateHover(hexList, character='▏', width=160):
    out = [
        r'tellraw @a {"text":"Hover to view image.","hoverEvent":{"action":"show_text","contents":["\n"'
    ]

    index = 0
    for x in hexList:
        out.append(r',{"text":"')
        if index >= width:
            out.append(r"\n")
            index -= width
        out.append(character + r'","color":"#' + x + r'"}')
        index += 1
    out.append(r"]}}")
    return "".join(out)


def generateText(hexList, character='▏'):
    out = ['tellraw @a ["\n"']
    for x in hexList:
        out.append(",")
        out.append(r'{"text":"' + character + r'","color":"#' + x + r'"}')
    out.append("]")
    return "".join(out)


def write(text, path="tellraw.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# idk where you wanna get the url but this'll do it
# requires string of url as an argument

if __name__ == "__main__":
    write(
        generateText(
            hexify(makeRGBArray(resizeImage(getImage(readUrl(input("URL: "))))))
        )
    )
