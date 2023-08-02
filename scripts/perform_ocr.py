#get rid of all the imports from earlier versions of the program 
import os
import subprocess
from PIL import Image
import glob
import io
#import pdf2image
#from pdf2image import convert_from_bytes, convert_from_path
import pytesseract 
from PyPDF2 import PdfReader
import fitz
import argparse 
import zipfile
import json 


parser = argparse.ArgumentParser()
parser.add_argument("--input", dest='zip_file', help="zip data for the ocr")
parser.add_argument("--output", dest= "out_file", help = "output json file")
                    
args = parser.parse_args()
output = args.out_file
input_file = args.zip_file

def make_ocr(the_file,the_name):
    name = the_name 
    text_file = []
    #doc = the_file  #fitz.open(the_file)
    for i in range(len(the_file)):
        page = the_file[i]
        #there are settings here which might optimize ocr accuracy
        pixmap = page.get_pixmap()
        png_bytes = pixmap.tobytes()
        image = Image.open(io.BytesIO(png_bytes))
        #there are also definitely modes to further test/explore here, including page/column segmentation mode 
        text = pytesseract.image_to_string(image)
        text_file.append(text)
    text_file = "".join(text_file)
    name = the_name
    name = name.replace('-', ":", 1)
    name = name.replace("-",".")
    name = name.strip('.pdf')
    text_dictionary = {}
    text_dictionary[name] = text_file
    return text_dictionary

json_list = []
text_list = []
new_json_list = []
with zipfile.ZipFile(input_file, 'r') as zip_file:
    for line in zip_file.namelist():
        with zip_file.open(line) as x: 
            if line.endswith(".jsonl"):
                for line in io.TextIOWrapper(x):
                    data = json.loads(line)
                    json_list.append(data)
            elif line.endswith('.pdf'):
                doc = fitz.open("pdf",x.read())
                out_text = make_ocr(doc,line)
                text_list.append(out_text)
print(len(json_list))
print(len(text_list))
for y in json_list:
    for z in text_list:
        name1 = z.keys()
        for name in name1:
            if y["levy_pid"] == name:
                y["full_text"] = z[name]
                new_json_list.append(y)
                print("matched" + name)
with open(output, "w") as json_file:
    for entry in new_json_list:
        json.dump(entry, json_file)
        json_file.write('\n')
