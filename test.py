# import easyocr
# import cv2
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from happytransformer import HappyTextToText
from happytransformer import TTSettings
from textblob import TextBlob

from PIL import Image
from pytesseract import pytesseract
#Define path to tessaract.exe
path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
#Define path to image
path_to_image = 'samp_4.jpg'
#Point tessaract_cmd to tessaract.exe
pytesseract.tesseract_cmd = path_to_tesseract
#Open image with PIL
img = Image.open(path_to_image)
#Extract text from image
texts = pytesseract.image_to_string(img)
# print(text)
happy_tt = HappyTextToText("T5",  "prithivida/grammar_error_correcter_v1")
settings = TTSettings(do_sample=True, top_k=10, temperature=0.5,  min_length=1, max_length=100)

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


sum_string = summarize(texts , 0.20)
string_list = sum_string.split(".")
result_new=""
for i in string_list:
    result = happy_tt.generate_text("gec : "+i, args=settings)
    
    sentence = TextBlob(result.text)
    result_new = result_new+ "." +str(sentence.correct())
print(result_new)



# result = happy_tt.generate_text("gec : "+summarize(texts, 0.1), args=settings)
# print(result.text)
