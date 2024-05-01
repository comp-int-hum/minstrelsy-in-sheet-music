import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--counts",
    dest="data",
    required=True,
    help="counts file program will be applied to."
)

parser.add_argument(
    "--output",
    dest="output",
    required=True,
    help="output json file ."
)

args = parser.parse_args()
input_data = args.data 



def compute_percentages_single_list(input_file):
        data_list = input_file
        topic_proportions_per_year = {}
        average_topic_proportions = {}
        for year, count in data_list:
            year_total = sum(count.values())
            for topic, topic_count in count.items():
                if topic not in topic_proportions_per_year:
                     topic_proportions_per_year[topic] = []
                topic_proportions_per_year[topic].append(topic_count / year_total)
        print(topic_proportions_per_year)
        
        for topic, proportions in topic_proportions_per_year.items():
            average_topic_proportions[topic] = sum(proportions) / len(proportions)
        results = {
            "overall_variance_by_topic": average_topic_proportions,
                }
        return results
        

def compute_percentages(input_data):
    results_list = []
    with open(input_data, 'r') as f:
        data_list = []
        for line in f:
            print("this is the line")
            line = line.strip()
            print(line)
            data_list.append(json.loads(line))
            
      #  sub_list = data_list[0]
      #  sub_sub_list = sub_list[0]
       # results_list = []
       # if isinstance(sub_sub_list, list) == True:
        #    print("evaluated to true")
        for x in data_list:
            results_list.append(compute_percentages_single_list(x))
        #else:
         #   print("evaluated to false")
          #  results_list.append(compute_percentages_single_list(data_list))

        return results_list
        
        
        
        

results = compute_percentages(input_data)

# If you want to save the results to a JSON file:

with open(args.output, 'w') as f:
    json.dump(results, f,  indent=4)

