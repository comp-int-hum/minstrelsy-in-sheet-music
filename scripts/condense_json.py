import json
import glob
import os
import os.path
import argparse 
from collections import defaultdict
import re
import csv 
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    dest="input_file",
    required=True,
    nargs = "+",
    help="list of variance files")



parser.add_argument(
    "--output_file",
    dest="output",
    required=True,
    nargs = "+",
    help="Counts output file."
)

parser.add_argument(
    "--json_output",
    dest="json_output",
    help="json output file."
)



args = parser.parse_args()
csv_name_list = args.input_file
holding_list = []
sorted_results = defaultdict(lambda: defaultdict(list))
for x in args.input_file:
    
    type_dict = {}
    res = [int(i) for i in re.findall(r'\d+', x)]
    
    if len(res) > 2:
        seed = res[0]
        resolution = res[1]
        topic_no = res[2]
    else:
        resolution = res[0]
        topic_no = res[1]
    with open(x, "r") as json_file:
        variance = json.load(json_file)
        variance_dict = {}

        variance_dict["overall_variance_by_topic"] = variance["overall_variance_by_topic"]
        #variance_dict["overall_variance"] = variance["overall_variance"]
        variance_dict["seed"] = seed
        variance_dict["name"] = x    
        
        sorted_results[topic_no][resolution].append(variance_dict)



with open(args.json_output, "w") as outfile:
    json.dump(sorted_results, outfile)

#dictionary of topic numbers and resolution dictionaries 
for topic_no, resolutions in sorted_results.items():
    
    
    
    for resolution_no, variances in resolutions.items():
    #for resolution_no_list in resolution_keys:
        csv_list = []
        for entry in variances: 
            #print(entry)
            topics = entry["overall_variance_by_topic"]
            sorted_topic_dict = sorted(topics.items(), key=lambda x: int(x[0]))
            #print(sorted_topic_dict)
            name = entry["name"]
            name = ("name", name)
            #print(name)
            sorted_topic_dict.insert(0, name)
            csv_list.append([sorted_topic_dict,entry])
            output = "work/variance_csv_{}_topics_{}_resolution.csv".format(topic_no,resolution_no)
            
        with open(output, 'w', newline='') as csvfile:
            list_format = csv_list[0]
            name = list_format[1]
            name = name["name"]
            name = ("name", name)
            fieldnames = list_format[0]
            #fieldnames.insert(0, name)
            fieldnames = dict(fieldnames)
            fieldnames = fieldnames.keys()
     #       print(fieldnames)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for a_list in csv_list:
                #print(a_list)
                row_dict = dict(a_list[0])
      #          print(row_dict)
                writer.writerow(row_dict)    
                    
                    
                
        






    














