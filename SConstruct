import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from steamroller import Environment
import glob
import custom

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables( 
    ("NUMBERS_OF_TOPICS", "", [5,10,15,20,30]),
    ("DATA_LOCATION", "", "/home/sbacker2/minstrelsy-in-sheet-music/data/levy_zip.zip"),
    ("GROUP_RESOLUTIONS", "", [5, 10, 25]),
    ("CHUNK_SIZE", "", [500]),
    ("EXISTING_JSON", "", True),
    ("EXISTING_JSON_LOCATION","","/home/sbacker2/git_test/minstrelsy-in-sheet-music/data/json_metadata.jsonl")
)    
# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    BUILDERS={ 
        "PerformOcr" : Builder(
            action="python scripts/perform_ocr.py --input ${SOURCES[0]} --output ${TARGETS[0]}", chdir = False),
        "TrainModel" : Builder(
            action="python scripts/train_model.py --data ${SOURCES[0]}  --output_file ${TARGETS[0]} --topic_num ${NUMBER_OF_TOPICS}", chdir = False            
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py --model ${SOURCES[0]} --data ${DATA} --topic_num ${NUMBER_OF_TOPICS} --json_out ${TARGETS[0]}"
        ),
	"GroupData" : Builder(action="python scripts/group_data.py --data ${SOURCES[0]}  --output_file ${TARGETS[0]} --random_on ${RANDOM_ON} --sub_category ${SUB_CAT} --reverse_sub_category ${REVERSE_SUB_CAT} --seed ${SEED}"),
	"CountData" : Builder(action= "python scripts/count_data.py --data ${SOURCES[0]}  --group_resolution ${GROUP_RESOLUTION} --counts ${TARGETS[0]} --lost_words ${LOST_WORDS} --lost_words_check ${LOST_WORDS_CHECK}"),
	"InspectModel" : Builder (action = "python scripts/inspect_model.py --counts ${SOURCES[0]} --figure ${TARGETS[0]} --model ${MODEL}"),
	"CalculateVariance" : Builder (action = "python scripts/calculate_count_variances.py --counts ${SOURCES[0]} --output ${TARGETS[0]}"),
	"CalculatePercentages" : Builder (action = "python scripts/calculate_count_percentages.py --counts ${SOURCES[0]} --output ${TARGETS[0]}"),
	"CondenseJson" : Builder ( action= "python scripts/condense_json.py --output_file ${OUTPUT} --json_output ${TARGETS[0]} --input ${SOURCES}"),
	"CreateVarianceReport": Builder (action = "python scripts/create_variance_report.py --output_file ${TARGETS[0]} --input_file ${SOURCES[0]} --model ${MODEL}", chdir=False) 
}
)

# The basic pattern for invoking a build rule is:
#
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"

topic_model_list = []
if env["EXISTING_JSON"] == False: 
    json_metadata_including_text_ocr = env.PerformOcr(["work/json_metadata.jsonl"],env["DATA_LOCATION"])
else:
    json_metadata_including_text_ocr = env["EXISTING_JSON_LOCATION"]

for number in env["NUMBERS_OF_TOPICS"]:
    topic_model_list.append([number, (env.TrainModel("work/model_with_no_{}_topics.bin".format(number),json_metadata_including_text_ocr, NUMBER_OF_TOPICS = number))])
results = []

# model format model [0] = number of topics, model[1] = trained model 

for model in topic_model_list:
    results.append([ model, (env.ApplyModel("work/levy_json_{}_topics.jsonl".format(model[0]), model[1],  DATA = json_metadata_including_text_ocr, NUMBER_OF_TOPICS = model[0]))])

output = []
minstrel_output = []
#results format [[number, model],apply model]

random_segmentation = []
random_segmentation_counts = []

for result in results:
    random_segmentation.append([env.GroupData("work/random_data_segments_topic_no_{}.json".format(result[0][0]), result[1], SEED = 1, RANDOM_ON = 100,REVERSE_SUB_CAT = 0),result])

for random_segment in random_segmentation:    
    for resolution in env["GROUP_RESOLUTIONS"]:
    	random_segmentation_counts.append([(env.CountData("work/random_data_counts_{}_resolution_topic_no_{}.json".format(resolution,random_segment[1][0][0]),random_segment[0], GROUP_RESOLUTION = resolution,  LOST_WORDS_CHECK = False, LOST_WORDS = "work/placeholder")), random_segment, resolution])
	
variance_name_list = []
percentage_name_list = []

random_variance_list = []
random_percentage_list = []
for entry in random_segmentation_counts:
    print(entry[0])
    random_variance_list.append([env.CalculateVariance("work/random_variance_list_{}_resolution_{}_topic_no.json".format(entry[2],entry[1][1][0][0]),entry[0]), entry])
    variance_name_list.append("work/random_variance_list_{}_resolution_{}_topic_no.json".format(entry[2],entry[1][1][0][0]))
    random_percentage_list.append([env.CalculatePercentages("work/random_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2],entry[1][1][0][0]), entry[0]), entry])
    percentage_name_list.append("work/random_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2],entry[1][1][0][0]))


full_set_counts = []
for result in results: 
    for resolution in env["GROUP_RESOLUTIONS"]:
    	full_set_counts.append(
	[env.CountData("work/full_group_data_counts_{}_resolution_topic_no_{}.json".format(resolution,result[0][0]), result[1], GROUP_RESOLUTION = resolution, LOST_WORDS_CHECK = True,
	LOST_WORDS = "work/full_group_data_set_lost_words{}_resolution_topic_no_{}.json".format(resolution,result[0][0])),
	result, resolution])
	
 


full_group_variance_list = []
full_group_percentage_list = []
for entry in full_set_counts:
    full_group_variance_list.append([env.CalculateVariance("work/full_group_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]),entry[0]), entry])
    variance_name_list.append("work/full_group_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]))
    full_group_percentage_list.append([env.CalculatePercentages("work/full_group_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]), entry[0]), entry])
    percentage_name_list.append("work/full_group_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]))


selected_segmentation_counts = []
selected_segmentation = []
for result in results:
    selected_segmentation.append([env.GroupData("work/selected_data_segments_topic_no_{}.json".format(result[0][0]), result[1], SEED= 0, RANDOM_ON= 0, SUB_CAT = "subjectSearched Minstrel shows", REVERSE_SUB_CAT = 0), result])
for segmentation in selected_segmentation:
    for resolution in env["GROUP_RESOLUTIONS"]:
        selected_segmentation_counts.append([(env.CountData("work/selected_data_counts_{}_resolution_topic_no_{}.json".format(resolution,
	segmentation[1][0][0]), segmentation[0], GROUP_RESOLUTION=resolution, LOST_WORDS_CHECK=False, LOST_WORDS="placeholder")),
	segmentation, resolution])


selected_variance_list = []
selected_percentage_list = []
for entry in selected_segmentation_counts:
    selected_variance_list.append([env.CalculateVariance("work/selected_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][0][0]), entry[0]), entry])
    variance_name_list.append("work/selected_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][0][0]))
    selected_percentage_list.append([env.CalculatePercentages("work/selected_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][0][0]),entry[0]), entry])
    percentage_name_list.append("work/selected_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][0][0]))


no_minstrel_counts = []
no_minstrel_segmentation = []
for result in results:
    no_minstrel_segmentation.append([env.GroupData("work/no_minstrel_data_segments_topic_no_{}.json".format(result[0][0]), result[1], SEED=0, RANDOM_ON=0, LOST_WORDS_CHECK=False,REVERSE_SUB_CAT = 1, LOST_WORDS="work/dummy", SUB_CAT="subjectSearched Minstrel shows"), result])


for segment in no_minstrel_segmentation:
    for resolution in env["GROUP_RESOLUTIONS"]:
        no_minstrel_counts.append([(env.CountData("work/no_minstrel_data_counts_{}_resolution_topic_no_{}.json".format(resolution,
	result[0][0]), no_minstrel_segmentation[0][0], GROUP_RESOLUTION=resolution, LOST_WORDS_CHECK=False, LOST_WORDS="placeholder")),
	segment,resolution])



no_minstrel_variance_list = []
no_minstrel_percentage_list = []
for entry in no_minstrel_counts:
    no_minstrel_variance_list.append([env.CalculateVariance("work/no_minstrel_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]),entry[0]), entry])
    variance_name_list.append("work/no_minstrel_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]))
    no_minstrel_percentage_list.append([env.CalculatePercentages("work/no_minstrel_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]),entry[0]), entry])
    percentage_name_list.append("work/no_minstrel_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][0][0]))


#creating name list for resolution
csv_variance_output_name_list = []
csv_percentage_output_name_list = []
for topic_no in env["NUMBERS_OF_TOPICS"]:
    for resolution in env["GROUP_RESOLUTIONS"]:
        csv_variance_output_name_list.append("work/variance_csv_{}_topics_{}_resolution.csv".format(topic_no,resolution))
        csv_percentage_output_name_list.append("work/percentages_csv_{}_topics_{}_resolution.csv".format(topic_no, resolution))

#creating name list for json

#csv_variance_output_list = []
#csv_percentage_output_list = []

#for topic_no in env["NUMBERS_OF_TOPICS"]:
 #   for resolution in env["GROUP_RESOLUTIONS"]:
  #      csv_variance_output_list.append("variance_csv_{}_topics_{}_resolution.csv".format(topic_no,resolution))
   #     csv_percentage_output_list.append("work/percentages_csv_{}_topics_{}_resolution.csv".format(topic_no, resolution))


#new_variance_input_list = []

#for x in full_group_variance_list:
 #   new_variance_input_list.append(x[0])
#for x in no_minstrel_variance_list:
 #   new_variance_input_list.append(x[0])
#for x in selected_variance_list:
 #   new_variance_input_list.append(x[0])
#for x in random_variance_list:
 #   new_variance_input_list.append(x[0])

#new_percentage_input_list = []

#for x in full_group_percentage_list:
 #   new_percentage_input_list.append(x[0])
#for x in no_minstrel_percentage_list:
 #   new_percentage_input_list.append(x[0])
#for x in selected_percentage_list:
 #   new_percentage_input_list.append(x[0])
#for x in random_percentage_list:
 #   new_percentage_input_list.append(x[0])

variance_json_output = []
percentages_json_output = []

variance_json_output.append(env.CondenseJson("work/condensed_variance_json_dictionary.json", variance_name_list, OUTPUT = csv_variance_output_name_list))
percentages_json_output.append(env.CondenseJson("work/condensed_percentages_json_dictionary.json",percentage_name_list, OUTPUT = csv_percentage_output_name_list))

graph = []


for entry in full_set_counts:
    #print(entry)
    graph.append([env.InspectModel("work/full_graph_with_{}_resolution_{}_topics.png".format(entry[2],entry[1][0][0]), entry[0],
    MODEL = entry[1][0][1]), entry])

for entry in selected_segmentation_counts:
    graph.append([env.InspectModel("work/graph_of_minstrel_texts_with_{}_resolution_{}_topics.png".format(entry[2],entry[1][1][0][0]),
    entry[0],MODEL = entry[1][1][0][1]), entry])

for entry in no_minstrel_counts:
   graph.append([env.InspectModel("work/graph_of_no_minstrel_texts_with_{}_resolution_{}_topics.png".format(entry[2],entry[1][1][0][0]),
   entry[0],MODEL = entry[1][1][0][1]), entry])

just_models_list = []
for model in topic_model_list:
    just_models_list.append(model[1])
print(just_models_list)

print(variance_json_output)
variance_report_list = []
for thing in variance_json_output:
    print(thing)
    print(thing[0])
    variance_report_list.append(env.CreateVarianceReport("work/variance_report.json", thing[0], MODEL = just_models_list))

for thing in percentages_json_output:
    variance_report_list.append(env.CreateVarianceReport("work/percentage_report.json", thing[0], MODEL = just_models_list))




    














