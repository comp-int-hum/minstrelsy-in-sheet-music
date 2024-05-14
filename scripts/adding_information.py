import json
import csv 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    dest="the_input",
    required=True,
    help="counts file program will be applied to."
)

parser.add_argument(
    "--output_file",
    dest="the_output",
    required=True,
    help="counts file program will be applied to."
)

args = parser.parse_args()


original_data_list = []
secondary_list = []
output_list = []
counter = 0
with open(args.the_input,  "r") as original_file:
    for x in original_file: 
        original_data_list.append(json.loads(x))



with open("/home/sbacker2/projects/minstrelsy_in_sheet_music/minstrelsy-in-sheet-music/data/1774_2-2085collectionMetadata_2023-09-07.csv", "r", newline = "") as in_file:
    reader = csv.DictReader(in_file)
    for row in reader:
        secondary_list.append(row)

for x in original_data_list:
    for y in secondary_list:
        if x["title"] == y["dc.title"]:
            x.update(y)
            print(x)
            print(" ")
            output_list.append(x)
print("this is the original data list")
print(len(original_data_list))
print("this is the updated list")
print(len(output_list))




with open(args.the_output, "w") as out_file:
    json.dump(output_list, out_file)
