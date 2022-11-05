from summarizer import Summarizer
from PIL import Image
from pytesseract import pytesseract
from flask import Flask,request
from flask import render_template
from flask_cors import CORS
from transformers import logging
import os


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/result',methods=["POST"])
def result():
    path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    # path_to_image = 'WhatsApp Image 2022-11-05 at 19.37.07.jpeg'
    pytesseract.tesseract_cmd = path_to_tesseract
    # img = Image.open(path_to_image)
    img = Image.open(request.files['img'])
    texts = pytesseract.image_to_string(img)
    logging.set_verbosity_error()
    model = Summarizer()
    result = model(texts, min_length=20)
    summary = "".join(result)
    return summary

@app.route("/")
def home():
    return render_template("index.html")

port = int(os.environ.get("PORT", 5000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
