import gzip
import requests
import numpy as np
import cv2 as cv


def getImage(url):
    response = requests.get(url, stream=True).raw
    return cv.resize(
        cv.imdecode(
            np.asarray(bytearray(response.read()), dtype=np.uint8), cv.IMREAD_COLOR
        ),
        (53, 20),
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


def onfimHandler(link):
    return gzip.compress(
        compactOut(hexify(makeRGBArray(getImage(link.strip())))).encode("utf-8")
    )


def generateTellraw(hexList):
    out = []
    out.append('tellraw @a [""')
    for x in hexList:
        out.append(",")
        out.append(r'{"text":"#","color":"#' + x + r'"}')
    out.append("]")
    return "".join(out)


def write(text):
    with open("tellraw.txt", "w", encoding="utf-8") as f:
        f.write(text)


# idk where you wanna get the url but this'll do it
# requires string of url as an argument

if __name__ == "__main__":
    write(generateTellraw(hexify(makeRGBArray(getImage(input("URL: "))))))
