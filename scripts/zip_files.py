import json
import argparse
import glob
from zipfile import ZipFile
import zipfile
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument("--glob_input", dest='glob_input', default = '/home/sbacker2/from_aws/my_project/Tin_Pan_Alley_Topic_Model/*.pdf', help="pattern for glob" )
parser.add_argument("--output_name", dest="output", help="name of output")
parser.add_argument("--test", dest = "test", type = int, default = 0 , choices = range(1,500), help = "write an integer to initiate test mode") 
args = parser.parse_args()
input_glob = args.glob_input
output = args.output
tester = args.test
pdf_list = glob.glob(input_glob)
with ZipFile(output,'w',compression = zipfile.ZIP_DEFLATED, compresslevel = 9) as zips:
    zips.write('/home/sbacker2/Tues_May_2/data/levy_metadata.jsonl', arcname = "levy_metadata.jsonl")
    if tester != 0:
        print("test in action!")
        for x in islice(pdf_list, tester):
            y = x.split("/")
            y = y[-1]
            zips.write(x, arcname = y)
            print("wrote" + y)
    else: 
        for x in pdf_list:
            y = x.split("/")
            y = y[-1]
            zips.write(x, arcname = y)
            print("wrote" + y)

