import csv
import re
import logging
import pickle
import json
import argparse
import gensim

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    dest="data",
    help="Data file model will be applied to."
)

parser.add_argument(
    "--output_file",
    dest="output",
    help="Counts output file."
)

args = parser.parse_args()

report_data = []
with open(args.data, "r") as in_file:
    report_data = json.load(in_file)

interest_list = []
    
for num_of_topics, year_buckets in report_data.items():
    print(num_of_topics)
    for year_bucket, bucket in year_buckets.items():
        for topic, entry in bucket.items():

            if abs(entry["minstrel_z_score"]) > 1:
                interest_dictionary = { "topic_num" : num_of_topics,
                                        "year_bucket" : year_bucket,
                                        "topic_report" : entry}
                interest_list.append(interest_dictionary)

with open(args.output, "w") as out_file:
    json.dump(interest_list, out_file, indent = 2)
