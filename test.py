# import easyocr
# import cv2
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# reader = easyocr.Reader(['en'],gpu = False) # load once only in memory.

# image_file_name="article.png" 
# image = cv2.imread(image_file_name)

# # sharp the edges or image.
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
# thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# r_easy_ocr=reader.readtext(thresh,detail=0)
# print(type(r_easy_ocr))

# stringg=""
# print(stringg.join(r_easy_ocr))
# string_f=stringg.join(r_easy_ocr)
from PIL import Image
from pytesseract import pytesseract
#Define path to tessaract.exe
path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
#Define path to image
path_to_image = 'article.png'
#Point tessaract_cmd to tessaract.exe
pytesseract.tesseract_cmd = path_to_tesseract
#Open image with PIL
img = Image.open(path_to_image)
#Extract text from image
texts = pytesseract.image_to_string(img)
# print(text)

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency= max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=(''.join(final_summary)).replace('\\',"")
    return summary


print(summarize(texts, 0.1))
# print(final)