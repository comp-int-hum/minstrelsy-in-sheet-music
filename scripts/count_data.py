import json
import argparse
import random
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be applied to."
)

parser.add_argument(
    "--group_resolution",
    dest="group_resolution",
    default=10,
    type=int,
    help="The size of each group (e.g. number of years, or whatever units 'group_field' uses)."
)

parser.add_argument(
    "--date_cutoff",
    dest="cutoff_date",
    default=1930,
    type=int,
    help="final date counted for groups"
)

parser.add_argument(
    "--date_start",
    dest="start_date",
    default=1800,
    type=int,
    help="final date counted for groups"
)


parser.add_argument(
    "--counts",
    dest="counts",
    required=True,
    help="Counts output file."
)


counter = 0
pub_date_lost = 0
lost_words = []
counts_list = []
data_dictionary_list = []

args = parser.parse_args()



with open(args.data, "rt") as ifd:
    if args.data.endswith(".jsonl"):
        data_list = []
        for line in ifd:
            dictionary = json.loads(line.strip())
            data_list.append(dictionary)
        data_dictionary_list.append(data_list)
    else: 
        data_dictionary_list = json.load(ifd)
        
    for collection in data_dictionary_list:
        groupwise_topic_counts = {}
        for individual_dictionary in collection: 
            local_counter = 0
            print(individual_dictionary)
        
           
            group_value = individual_dictionary["time"]
            group = group_value - (group_value % args.group_resolution)
            groupwise_topic_counts[group] = groupwise_topic_counts.get(group, {})
            document_topics_by_word = individual_dictionary["text"]
                
        
            for word in document_topics_by_word:      
                word_text = word[0]
                     
                topic = word[1]
                  
                #if len(topic) > 0:
                 #   topic = max(topics, key = lambda x: x[1])
                  #  topic = topic[0]
                if group_value  < args.cutoff_date and group_value > args.start_date:
                    groupwise_topic_counts[group][topic] = groupwise_topic_counts[group].get(topic, 0) + 1
                    
        counts_list.append(groupwise_topic_counts)                
    
#prints out a series of jsonl count objects 

with open(args.counts, "wt") as ofd:
    for count in counts_list: 
        ofd.write(
            json.dumps(
                [(k, v) for k, v in sorted(count.items()) if len(v) > 0]
            ) + "\n"
        )

