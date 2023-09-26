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
    help="Counts output file."
)

no_of_topics = [5,10,15,20,30]
group_resolutions = [5, 10, 25]

args = parser.parse_args()
holding_list = []
sorted_results = defaultdict(lambda: defaultdict(list))
for x in args.input_file:
    print(x)
    type_dict = {}
#    res = [int(i) for i in x.split() if i.isdigit()]
    res = [int(i) for i in re.findall(r'\d+', x)]
    print(res)
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
        variance_dict["overall_variance"] = variance["overall_variance"]
        variance_dict["seed"] = seed
        variance_dict["name"] = x    
        sorted_results[topic_no][resolution].append(variance_dict)



for topic_no, resolutions in sorted_results.items():
    #resolution_keys = sorted_results[key].keys()
    #print(resolution_keys)
    for resolution_no, variances in resolutions.items():
    #for resolution_no_list in resolution_keys:
        csv_list = []
        for entry in variances: 
            print(entry)
            topics = entry["overall_variance_by_topic"]
            sorted_topic_dict = sorted(topics.keys(), key= lambda x: int(x))
            csv_list.append([sorted_topic_dict,entry])
            output = "work/variancecsv{}topics{}resolution.csv".format(topic_no,resolution_no)
            #output = args.output
        with open(output, 'w', newline='') as csvfile:
            name_list = []
            for entry in csv_list:
                spec_dict = entry[1]
                name = spec_dict["name"]
                name_list.append(name)
            fieldnames = ["key"] + name_list
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key in csv_list[0][0]:
                row_dict = {"key" : key}
                for d in csv_list:
                    name_dict = d[1]
                    name_dict = name_dict["name"]
                    num_dict = d[0]
                    row_dict["name_dict"] = num_dict[key]
            writer.writerow(row_dict)    
                    
                    
                
        






    














