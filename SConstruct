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
import steamroller
import glob

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
    ("CHUNK_SIZE", "", [500])
)    
# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    BUILDERS={ 
        "PerformOcr" : Builder(
            action="python scripts/perform_ocr.py --input ${SOURCES[0]} --output ${TARGETS[0]}", chdir = False),
        "TrainModel" : Builder(
            action="python scripts/train_model.py --data ${SOURCES[0]}  --output_file ${TARGETS[0]} --topic_num ${NUMBER_OF_TOPICS}", chdir = False            
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py --model ${SOURCES[0]} --data ${DATA} --topic_num ${NUMBER_OF_TOPICS} --json_out ${TARGETS[0]}"
        ),
	"GroupData" : Builder(action="python scripts/group_data.py --data ${SOURCES[0]} --group_resolution ${GROUP_RESOLUTION} --counts ${TARGETS[0]} --lost_words ${LOST_WORDS} --lost_words_check ${LOST_WORDS_CHECK} --sub_category ${SUB_CATEGORY} --seed ${SEED}"),
	"InspectModel" : Builder (action = "python scripts/inspect_model.py --counts ${SOURCES[0]} --figure ${TARGETS[0]} --model ${MODEL}"),
	"CalculateVariance" : Builder (action = "python scripts/calculate_variance.py --counts ${SOURCES[0]} --output ${TARGETS[0]}"),
	"CondenseJson" : Builder ( action= "python scripts/condense_json.py --output ${TARGETS} --input ${SOURCES}")
	
}
)

# The basic pattern for invoking a build rule is:
#
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"

topic_model_list = []
json_metadata_including_text_ocr = env.PerformOcr(["work/json_metadata.jsonl"],env["DATA_LOCATION"])
for number in env["NUMBERS_OF_TOPICS"]:
    topic_model_list.append([number, (env.TrainModel("work/model_with_{}_topics.bin".format(number),json_metadata_including_text_ocr, NUMBER_OF_TOPICS = number))])
results = []

# model format model [0] = number of topics, model[1] = trained model 

for model in topic_model_list:
    results.append([ model, (env.ApplyModel("work/levy_json_{}_topics.jsonl".format(model[0]), model[1],  DATA = json_metadata_including_text_ocr, NUMBER_OF_TOPICS = model[0]))])

output = []
minstrel_output = []
#results format [[number, model],apply model]

random_segmentation = []

for x in range(100): 
    for result in results: 
    	for resolution in env["GROUP_RESOLUTIONS"]:
	     random_segmentation.append([result,x,resolution,env.GroupData("work/segmented_data_counts_seed_{}_resolution_{}_topic_no_{}.json".format(x,resolution,result[0][0]), result[1], GROUP_RESOLUTION = resolution, SEED = x, LOST_WORDS_CHECK = False,	LOST_WORDS = "work/fake_words{}_{}".format(x,resolution))])

full_set = []
for result in results: 
    for resolution in env["GROUP_RESOLUTIONS"]:
    	full_set.append([result, resolution, env.GroupData("work/full_group_data_counts_{}_resolution_topic_no_{}.json".format(resolution,result[0][0]), result[1], GROUP_RESOLUTION = resolution, LOST_WORDS_CHECK = True, LOST_WORDS = "work/full_group_data_set_lost_words{}_resolution_topic_no_{}.json".format(resolution,result[0][0]))])
minstrel_set = []
for result in results:
    for resolution in env["GROUP_RESOLUTIONS"]:
       minstrel_set.append([result,resolution,env.GroupData("work/minstrel_full_group_data_counts_{}_resolution_topic_no_{}.json".format(resolution,result[0][0]), result[1],LOST_WORDS_CHECK = False, LOST_WORDS = "work/minstrel_fake_words{}_{}".format(resolution, result[0][0]), GROUP_RESOLUTION = resolution, SUB_CATEGORY = "subjectSearched Minstrel shows")]) 

#set append format = [result[number,model]apply_model], resolution, group_data]

random_variance_list = []
csv_output_list = []
for set in random_segmentation:
    random_variance_list.append([env.CalculateVariance("work/variance_{}_seed_{}_resolution_{}_topic_no.json".format(set[1],set[2],set[0][0][0]), set[3]),set])
    csv_output_list.append("work/variance_{}_seed_{}_resolution_{}_topic_no.json".format(set[1],set[2],set[0][0][0]))
full_set_variance_list = []
for set in full_set:
    full_set_variance_list.append([env.CalculateVariance("work/variance_{}_resolution_{}_topic_no.json".format(set[1],set[0][0][0]), set[2]), set])
    csv_output_list.append("work/variance_{}_resolution_{}_topic_no.json".format(set[1],set[0][0][0]))
minstrel_set_variance_list = []
for set in minstrel_set:
    minstrel_set_variance_list.append([env.CalculateVariance("work/minstrel_variance_{}_resolution_{}_topic_no.json".format(set[1],set[0][0][0]), set[2]), set])
    csv_output_list.append("work/minstrel_variance_{}_resolution_{}_topic_no.json".format(set[1],set[0][0][0]))
print(minstrel_set)
print("this is the minstrel variance list")    
print(minstrel_set_variance_list)
print(csv_output_list)

csv_name_list = []

for topic_no in env["NUMBERS_OF_TOPICS"]:
    for resolution in env["GROUP_RESOLUTIONS"]:
    	csv_name_list.append("variance_csv_{}_topics_{}_resolution.csv".format(topic_no,resolution))

csv_output = env.CondenseJson(csv_name_list, csv_output_list) 


graph = []
#entry format [ [[number, trained model], applied model], resolution, groupdata)
# [  [ [5, [<SCons.Node.FS.File object at 0x1b844a0>]], [<SCons.Node.FS.File object at 0x1b85940>]], 5, [<SCons.Node.FS.File object at 0x1f07cd0>]]

for entry in full_set:
    print(entry)
    graph.append([entry,env.InspectModel("work/graph_with_{}_resolution_{}_topics.png".format(entry[1],entry[0][0][0]), entry[2], MODEL = entry[0][0][1])])

for entry in minstrel_set:
    graph.append([results, env.InspectModel("work/graph_of_minstrel_texts_with_{}_resolution_{}_topics.png".format(entry[1],entry[0][0][0]), entry[2], MODEL = entry[0][0][1])])

variance = []

#for put in output:
 #   variance.append(env.CalculateVariance("work/variance_with_{}_resolution_{}_topics.json".format(put[0][0], put[0][1][0]), "work/counts_with_{}_resolution_{}_topics".f#ormat(put[0][0], put[0][1][0])))
#for put in minstrel_output:
 #   variance.append(env.CalculateVariance("work/variance_with_{}_resolution_{}_topics.json".format(put[0][0], put[0][1][0]), "work/counts_with_{}_resolution_{}_topics".f#ormat(
#put[0][0], put[0][1][0])))
    
#experimental_variance = []
#experimental_results = []
#for model in topic_model_list:
 #   for resolution in env["GROUP_RESOLUTIONS"]:
  #  	for x in range(100):
   #         experimental_results.append([x,resolution, model, (env.ApplyModelRandomSegmentation("work/random_counts{}_with_{}_resolution_{}_topics.json".format(x,resolut#ion, model[0]), model[1], GROUP_RESOLUTION = resolution, DATA = json_metadata_including_text_ocr, NUMBER_OF_TOPICS = model[0]))])
#for result in experimental_results:
 #   experimental_variance.append(env.CalculateVariance("work/experimental_variance{}_with_{}_resolution_{}_topics.json".format(result[0],result[1],result[2][0]),"work/ra#ndom_counts{}_with_{}_resolution_{}_topics.json".format(result[0],result[1],result[2][0])))

# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).
#report = env.GenerateReport(
#    "work/report.txt",
 #   results)
