import gzip
import requests
import numpy as np
import cv2 as cv

try:
    import aiohttp
    import asyncio
except ImportError:
    aiohttp = None


def isInvalidHeader(headers):
    try:
        size = int(headers['Content-Length'])
        if size > 2e7:  # 20 MB
            return f"Size is {size} bytes"
    except Exception:
        pass

    try:
        ct = headers['Content-Type']
        if 'image' not in ct:
            return f"Link is {ct}."
    except Exception:
        pass

    return False  # No error

async def readUrlAsync(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if not isInvalidHeader(response.headers.get('Content-Length')):
                return await response.read()


def readUrl(url):
    if aiohttp is None:
        response = requests.get(url, stream=True)
        if not isInvalidHeader(response.headers):
            return response.raw.read()
    else:
        return asyncio.run(readUrlAsync(url))


def getImage(arr):
    return cv.imdecode(np.asarray(bytearray(arr), dtype=np.uint8), cv.IMREAD_COLOR)


def resizeImage(image, height=None, width=240):
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


def generateJson(hexList, character='▏', width=240):
    out = [r'["\n"']
    index = 0
    for x in hexList:
        out.append(r',{"text":"')
        if index >= width:
            out.append(r"\n")
            index -= width
        out.append(character + r'","color":"#' + x + r'"}')
        index += 1
    out.append("]")
    return "".join(out)

def generateHoverTellraw(json):
    return r'tellraw @a {"text":"Hover to view image.","hoverEvent":{"action":"show_text","contents":' + json + r'}}'

def generateTellraw(json):
    return r'tellraw @a ' + json


def write(text, path="tellraw.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# <1.20 the width should be 160 to fit most screens.
# 1.20 changed the width of the character we use, here are new values to implement:
# GUI Scale x1: ~<960
# GUI Scale x2: ~<480
# GUI Scale x3: ~<320
# GUI Scale x4: =<240
# these numbers work for most screens (16:9)

# idk where you wanna get the url but this'll do it
# requires string of url as an argument (+ optionally width)

if __name__ == "__main__":
    width = input("width (press enter for default): ")
    if (width == ""):
        width = 240
    else:
        width = int(width)

    write(
        generateHoverTellraw(
            generateJson(
                hexify(makeRGBArray(resizeImage(getImage(readUrl(input("URL: "))),width=width))),width=width
            )
        )
    )
    
