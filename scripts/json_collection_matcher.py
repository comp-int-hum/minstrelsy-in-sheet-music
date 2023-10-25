import json
import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "--input_file1",
    dest="data",
    help="Data file model will be applied to."
)


#this one is write for json vs. jsonl
parser.add_argument(
    "--input_file2",
    dest="data2",
    help="Data file model will be applied to"
)

parser.add_argument(
    "--output_file",
    dest="output",
    help="Counts output file."
)

#parser.add_argument(
 #   "--topic_of_interest",
  #  dest="topic_of_interest",
   # type = int,
   # help="topic you want to look at"
#)

args = parser.parse_args()
data_one_list = []
data_two_list = []

with open(args.data, "r") as in_file:
    data_one_list = json.load(in_file)

with open(args.data2, "r") as in_file:
    data_two_list = json.load(in_file)
    data_two_list = data_two_list[0]

matched_dictionary_list = []
unmatched_dictionary_list = []

print(len(data_two_list))
print(len(data_one_list))
for entry in data_two_list:
    #print(type(entry))
    matched = False 
    for composition in data_one_list:
        #print(type(composition))
        if entry["levy_pid"] == composition[0]["levy_pid"]:
            matched_dictionary_list.append(entry)
            matched = True
            print("matched!")
    if matched == False:
          unmatched_dictionary_list.append(entry)
          print("ummatched")

report = {}
          
report["length_of_matched_dictionary"] = len(matched_dictionary_list)
report["length_of_unmatched_dictionary"] = len(unmatched_dictionary_list)
report["matched_dictionary_list"] = matched_dictionary_list
report["ummatched_dictionary_list"] = unmatched_dictionary_list

        
with open( args.output, "w") as out_file:
    json.dump(report, out_file)
