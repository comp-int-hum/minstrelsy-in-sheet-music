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

def calculate_variance(data):
    # Step 1: Find the mean
    mean = sum(data) / len(data)
    
    # Step 2, 3, and 4: Find the squared difference from the mean, and find their average
    variance = sum((xi - mean) ** 2 for xi in data) / len(data)
    
    return variance


def compute_percentages_single_list(input_file):
        data_list = input_file
        
        topic_proportions_per_year = {}
        average_variance = {}
        for year, count in data_list:
            year_total = sum(count.values())
            for topic, topic_count in count.items():
                if topic not in topic_proportions_per_year:
                     topic_proportions_per_year[topic] = []
                topic_proportions_per_year[topic].append(topic_count / year_total)
        print(topic_proportions_per_year)
        
        for topic, proportions in topic_proportions_per_year.items():
            average_variance[topic] = calculate_variance(proportions)
            
        results = {
            "overall_variance_by_topic": average_variance,
                }
        return results
        

def compute_percentages(input_data):
    with open(input_data, 'r') as f:
        data_list = []
        results_list = []
        for line in f:
            #print("this is the line")
            line = line.strip()
            #print(line)
            data_list.append(json.loads(line))

        

        #sub_list = data_list[0]
        #sub_sub_list = sub_list[0]
        #results_list = []
        #if isinstance(sub_sub_list, list) == True:
            #print("evaluated to true")
        for x in data_list:
            results_list.append(compute_percentages_single_list(x))
        #else:
         #   print("evaluated to false")
          #  results_list.append(compute_percentages_single_list(data_list))

        return results_list
        
        
        
        

results = compute_percentages(args.data)

# If you want to save the results to a JSON file:

with open(args.output, 'w') as f:
    json.dump(results, f,  indent=4)

