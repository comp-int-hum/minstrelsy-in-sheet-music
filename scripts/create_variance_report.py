import json
import statistics
import argparse
import math
import re
from gensim.models import LdaModel
import pickle
import gzip
import torch
from detm import DETM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    dest="input_file",
    required=True,
    help="input file that will be analyzed to."
)
parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')   
parser.add_argument(
    "--output_file",
    dest = "output_file",
    help = "names out output files")

parser.add_argument(
    "--model",
    dest = "model",
    nargs = "*",
    required = True, )

args = parser.parse_args()
args.device = "cpu"
output_dictionary = {}


with open(args.input_file, 'r') as f:
    data = json.load(f)

# Iterating through your data
# Note: Modify data access based on your actual structure if this doesn’t fit.
for number_of_topics, resolutions in data.items():
    resolution_dict = {}
    for x in args.model:
        print(x)
        integer_match = re.search(r'\d+', x)
        print(integer_match)
        integer_value = int(integer_match.group())
        print(integer_value)
        print(number_of_topics)
        if int(integer_value) == int(number_of_topics):
            print("match!")
            with gzip.open(x, "rb") as ifd:
                model = torch.load(ifd, map_location=torch.device(args.device))
                print("model loaded!")
    print("this is number of topics")
    print(number_of_topics)
    #print(resolutions)
    for resolution, variance_dict_context in resolutions.items():
        print("this is resolution")
        print(resolution)
        #print("this is variance_dict_context")
        #print(variance_dict_context)
        variance_collector = {}
        select_collector = {}
        report = {}
        for variance_dict in variance_dict_context:
            print(variance_dict)

            #creating two dictionaries -- one that collects all the variance values, one that collects the selected/minstrel values

            topic_no_list = variance_dict["overall_variance_by_topic"].keys()
            for topic in topic_no_list:
                if variance_collector.get(int(topic)):
                    pass 
                else:
                    variance_collector[int(topic)] = []
            if 'random' in variance_dict['name']:
                for topic, variance in variance_dict['overall_variance_by_topic'].items():
                    variance_collector[int(topic)].append(variance)        
                    #print(int(topic))
                
            elif 'selected' in variance_dict["name"]:
                for topic, variance in variance_dict['overall_variance_by_topic'].items():
                    print("this is the topic in select")
                    print(topic)
                    print(type(topic))
                    print("this is the variance")
                    print(variance)
                    select_collector[int(topic)] = variance
                    
        print("this is the get cycle")
        print(integer_value)
        for x in range(integer_value):

            print(x)
            select_collector[x] = select_collector.get(x, 0)         
        print(select_collector.keys())            
                    
        for topic_no, variance_values in variance_collector.items():
            print(topic_no)
            #print(type(topic_no))
            print(variance_values)

            divisor = statistics.stdev(variance_values)
            
            if statistics.stdev(variance_values) ==0: 
                   assert select_collector[topic_no] == 0
                   divisor = statistics.stdev(variance_values) + 1e-10

            
            
            minstrel_z_score = float(((select_collector[topic_no] - statistics.mean(variance_values)) / divisor))
            
            #this is more information that's nice to have here, but need to find a way to bring it in with the detm 

            #topic_terms = model.show_topic(topic_no)
            #topic_terms = [{'word': word, 'probability': float(prob)} for word, prob in topic_terms]
            report[topic_no] = {
                "mean": float(statistics.mean(variance_values)),
                "standard_deviation":float(statistics.stdev(variance_values)),
                "minstrel_value" :(float(select_collector[topic_no])),
                "minstrel_z_score" : minstrel_z_score,
                "resolution" : resolution,
                "topic_no" : topic_no,
                #"topic_terms" : topic_terms,
                "number of topics" : number_of_topics
            }
        resolution_dict[int(resolution)] = report
        #print(resolution_dict)
    output_dictionary[int(number_of_topics)] = resolution_dict
# Output or further use the report as per your use-case
print(output_dictionary)
with open (args.output_file, "w") as wf:
    json.dump(output_dictionary,wf, indent = 4)

