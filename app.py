import numpy as np
from summarizer import Summarizer
from PIL import Image
from pytesseract import pytesseract
#Define path to tessaract.exe

path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
#Define path to image
path_to_image = 'WhatsApp Image 2022-11-05 at 19.37.07.jpeg'
#Point tessaract_cmd to tessaract.exe
pytesseract.tesseract_cmd = path_to_tesseract
#Open image with PIL
img = Image.open(path_to_image)
#Extract text from image
texts = pytesseract.image_to_string(img)

from transformers import logging
logging.set_verbosity_error()
model = Summarizer()
result = model(texts, min_length=20)
summary = "".join(result)
print(summary)