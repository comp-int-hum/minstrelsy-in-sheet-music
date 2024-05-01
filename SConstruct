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
    ("CHUNK_SIZE", "", [500]),
    ("EXISTING_JSON", "", True),
    ("EXISTING_JSON_LOCATION","","/home/sbacker2/projects/minstrelsy_in_sheet_music/minstrelsy-in-sheet-music/data/json_metadata.jsonl"),
    ("USE_GRID","",1),
    ("RANDOM_LENGTH","",2000),
    ("GRID_TYPE","", "slurm"),
    ("GRID_GPU_COUNT","", 1),
    ("GRID_MEMORY", "", "64G"),
    ("GRID_TIME", "", "24:00:00"),
    ("WINDOW_SIZE","", [20])
    #("FOLDS", "", 1),
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
	"TrainEmbeddings" : Builder(
	    action = "python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"), 
	   "TrainModel" : Builder(
            action="python scripts/train_detm.py --embeddings ${SOURCES[0]} --train ${DATA_FILE}  --output ${TARGETS[0]} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE} --min_word_occurrence ${MIN_WORD_OCCURANCE} --window_size ${WINDOW_SIZE} --max_word_proportion ${MAX_WORD_PROPORTION}",chdir = False            
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${DATA_FILE} --output ${TARGETS[0]}"
        ),
	"GroupData" : Builder(action="python scripts/group_data.py --data ${SOURCES[0]}  --output_file ${TARGETS[0]} --random_on ${RANDOM_ON} --sub_category ${SUB_CAT} --reverse_sub_category ${REVERSE_SUB_CAT} --seed ${SEED} --sub_category_search ${SUB_CATEGORY_SEARCH} --sub_category_search_value ${SUB_CATEGORY_SEARCH_VALUE} --random_length ${RANDOM_LENGTH}"),
	"CountData" : Builder(action= "python scripts/count_data.py --data ${SOURCES[0]}  --group_resolution ${GROUP_RESOLUTION} --counts ${TARGETS[0]}"),
	"InspectModel" : Builder (action = "python scripts/inspect_model.py --counts ${SOURCES[0]} --figure ${TARGETS[0]} --model ${MODEL}"),
	"CalculateVariance" : Builder (action = "python scripts/calculate_count_variances.py --counts ${SOURCES[0]} --output ${TARGETS[0]}"),
	"CalculatePercentages" : Builder (action = "python scripts/calculate_count_percentages.py --counts ${SOURCES[0]} --output ${TARGETS[0]}"),
	"CondenseJson" : Builder ( action= "python scripts/condense_json.py --output_file ${OUTPUT} --json_output ${TARGETS[0]} --input ${SOURCES}"),
	"CreateVarianceReport": Builder (action = "python scripts/create_variance_report.py --output_file ${TARGETS[0]} --input_file ${SOURCES[0]} --model ${MODEL}", chdir=False), 
	"ConstructNumpyArray": Builder (action = "python scripts/construct_numpy_array.py --output_file ${TARGETS[0]} --input_file ${SOURCES[0]} --model ${MODEL} --year_bucket ${YEAR_BUCKET} --number ${NUMBER} "), 
	"ConstructBimodalityVarianceChart": Builder (action = "python scripts/generate_graph_of_bimodality_and_variance.py --output_dictionary ${TARGETS[0]} --matrix ${SOURCES[0]} --output_chart ${OUTPUT_CHART} --model ${MODEL}"),
	"AnalyzeTopWordsPerTopic" : Builder (action = "python scripts/numpy_analyze_top_topic_words_per_time.py --output_file ${TARGETS[0]} --matrix ${SOURCES[0]} --model ${MODEL}")

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

embeddings_list = []
embeddings_list.append(env.TrainEmbeddings("work/word_2_vec_embeddings.bin", json_metadata_including_text_ocr))


topic_model_list = []

for entry in embeddings_list: 
    for number in env["NUMBERS_OF_TOPICS"]:
    	for group in env["WINDOW_SIZE"]:
    	    topic_model_list.append([(env.TrainModel("work/applied_detm_with_{}_topics.bin".format(number) , entry , DATA_FILE = json_metadata_including_text_ocr, NUMBER_OF_TOPICS = number, BATCH_SIZE = 10, MIN_WORD_OCCURANCE = 1, MAX_WORD_PROPORTION = .99)),number, group])

applied_results_levy_list = []

# topic model list = [applied data, number window]
#applied results levy list = [applied levy, [model_list]
#random seg = [random seg [applied results]]
#random seg counts [count, [random seg list]

for model in topic_model_list:
    applied_results_levy_list.append([(env.ApplyModel("work/levy_json_{}_topics.jsonl".format(model[1]), model[0],  DATA_FILE = json_metadata_including_text_ocr)), model])


output = []
minstrel_output = []
#results format [[number, model],apply model]


#this part is for the random segmentation 


random_segmentation = []
random_segmentation_counts = []

for result in applied_results_levy_list:
    random_segmentation.append([env.GroupData("work/random_data_segments_topic_no_{}.json".format(result[1][1]), result[0], SEED = 1, RANDOM_ON = 100,REVERSE_SUB_CAT = 0, SUB_CATEGORY_SEARCH = "None",SUB_CATEGORY_SEARCH_VALUE="Junk"),result])

#need to fix potential future window-size disjunct


for random_segment in random_segmentation:    
    	random_segmentation_counts.append([(env.CountData("work/random_data_counts_{}_resolution_topic_no_{}.json".format(random_segment[1][1][2],random_segment[1][1][1]),random_segment[0], GROUP_RESOLUTION = random_segment[1][1][2])), random_segment])


variance_name_list = []
percentage_name_list = []

random_variance_list = []
random_percentage_list = []
for entry in random_segmentation_counts:
    print(entry[0])
    random_variance_list.append([env.CalculateVariance("work/random_variance_list_{}_resolution_{}_topic_no.json".format(entry[1][1][1][2],entry[1][1][1][1]),entry[0]), entry])
    variance_name_list.append("work/random_variance_list_{}_resolution_{}_topic_no.json".format(entry[1][1][1][2],entry[1][1][1][1]))
    random_percentage_list.append([env.CalculatePercentages("work/random_percentage_list_{}_resolution_{}_topic_no.json".format(entry[1][1][1][2],entry[1][1][1][1]), entry[0]), entry])
    percentage_name_list.append("work/random_percentage_list_{}_resolution_{}_topic_no.json".format(entry[1][1][1][2],entry[1][1][1][1]))


#this part is for the full data set 


full_set_counts = []
full_set_numpy_arrays = []
for result in applied_results_levy_list: 
  
  full_set_counts.append(
  	[env.CountData("work/full_group_data_counts_{}_resolution_topic_no_{}.json".format(result[1][2],result[1][1]), result[0], GROUP_RESOLUTION = result[1][2]),
	result])

  full_set_numpy_arrays.append(
	[env.ConstructNumpyArray("work/full_group_data_counts_numpy_array_{}_resolution_topic_no_{}.npy".format(result[1][2],result[1][1]), result[0], MODEL = result[1][0],
        YEAR_BUCKET =result[1][2], NUMBER = result[1][1]),result])

 	

full_group_variance_list = []
full_group_percentage_list = []
for entry in full_set_counts:
    full_group_variance_list.append([env.CalculateVariance("work/full_group_variance_list_{}_resolution_{}_topic_no.json".format(entry[1][1][2], entry[1][1][1]),entry[0]), entry])
    variance_name_list.append("work/full_group_variance_list_{}_resolution_{}_topic_no.json".format(entry[1][1][2], entry[1][1][1]))
    full_group_percentage_list.append([env.CalculatePercentages("work/full_group_percentage_list_{}_resolution_{}_topic_no.json".format(entry[1][1][2], entry[1][1][1]), entry[0]), entry])
    percentage_name_list.append("work/full_group_percentage_list_{}_resolution_{}_topic_no.json".format(entry[1][1][2], entry[1][1][1]))


#this part is for the selected data_set

selected_segmentation_numpy_arrays = []
selected_segmentation_counts = []
selected_segmentation = []
for result in applied_results_levy_list:
    selected_segmentation.append([env.GroupData("work/selected_data_segments_topic_no_{}.json".format(result[1][1]), result[0], SEED= 0, RANDOM_ON= 0, SUB_CAT = "subjectSearched Minstrel shows", REVERSE_SUB_CAT = 0, SUB_CATEGORY_SEARCH = "None",SUB_CATEGORY_SEARCH_VALUE="Junk"), result])
for segmentation in selected_segmentation:
    for resolution in env["WINDOW_SIZE"]:
        selected_segmentation_counts.append([(env.CountData("work/selected_data_counts_{}_resolution_topic_no_{}.json".format(resolution,
	segmentation[1][1][1]), segmentation[0], GROUP_RESOLUTION=resolution)),
	segmentation, resolution])

        selected_segmentation_numpy_arrays.append(
        [env.ConstructNumpyArray("work/selected_segmentation_numpy_array_{}_resolution_topic_no_{}.npy".format(resolution,segmentation[1][1][1]), segmentation[0], MODEL = segmentation[1][1][0], YEAR_BUCKET = resolution,NUMBER= segmentation[1][1][1]),segmentation,resolution])

selected_variance_list = []
selected_percentage_list = []
for entry in selected_segmentation_counts:
    selected_variance_list.append([env.CalculateVariance("work/selected_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]), entry[0]), entry])
    variance_name_list.append("work/selected_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]))
    selected_percentage_list.append([env.CalculatePercentages("work/selected_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]),entry[0]), entry])
    percentage_name_list.append("work/selected_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]))

#this part is for everything but the selected dataset


no_minstrel_counts = []
no_minstrel_segmentation = []
no_minstrel_numpy_array = []
for result in applied_results_levy_list:
    no_minstrel_segmentation.append([env.GroupData("work/no_minstrel_data_segments_topic_no_{}.json".format(result[1][1]), result[0], SEED=0, RANDOM_ON=0,REVERSE_SUB_CAT = 1, SUB_CAT="subjectSearched Minstrel shows",  SUB_CATEGORY_SEARCH = "None",SUB_CATEGORY_SEARCH_VALUE="Junk"), result])


for segment in no_minstrel_segmentation:
    for resolution in env["WINDOW_SIZE"]:
        no_minstrel_counts.append([(env.CountData("work/no_minstrel_data_counts_{}_resolution_topic_no_{}.json".format(resolution,
	segment[1][1][1]), segment[0], GROUP_RESOLUTION=resolution)),
	segment,resolution]) 

        no_minstrel_numpy_array.append(
        [env.ConstructNumpyArray("work/no_minstrel_numpy_array_{}_resolution_topic_no_{}.npy".format(resolution,segment[1][1][1]), segment[0], MODEL = segment[1][1][0], YEAR_BUCKET = resolution, NUMBER = segment[1][1][1]),segment,resolution])


no_minstrel_variance_list = []
no_minstrel_percentage_list = []
#entry = 
for entry in no_minstrel_counts:
    no_minstrel_variance_list.append([env.CalculateVariance("work/no_minstrel_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]),entry[0]), entry])
    variance_name_list.append("work/no_minstrel_variance_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]))
    no_minstrel_percentage_list.append([env.CalculatePercentages("work/no_minstrel_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]),entry[0]), entry])
    percentage_name_list.append("work/no_minstrel_percentage_list_{}_resolution_{}_topic_no.json".format(entry[2], entry[1][1][1][1]))


#creating name list for resolution
csv_variance_output_name_list = []
csv_percentage_output_name_list = []
for topic_no in env["NUMBERS_OF_TOPICS"]:
    for resolution in env["WINDOW_SIZE"]:
        csv_variance_output_name_list.append("work/variance_csv_{}_topics_{}_resolution.csv".format(topic_no,resolution))
        csv_percentage_output_name_list.append("work/percentages_csv_{}_topics_{}_resolution.csv".format(topic_no, resolution))

#creating name list for json



variance_json_output = []
percentages_json_output = []

variance_json_output.append(env.CondenseJson("work/condensed_variance_json_dictionary.json", variance_name_list, OUTPUT = csv_variance_output_name_list))
percentages_json_output.append(env.CondenseJson("work/condensed_percentages_json_dictionary.json",percentage_name_list, OUTPUT = csv_percentage_output_name_list))

graph = []


#inspection part to make charts -- these charts have issues, replacing with other format

#for entry in full_set_counts:
    #print(entry)
 #   graph.append([env.InspectModel("work/full_graph_with_{}_resolution_{}_topics.png".format(entry[2],entry[1][0][0]), entry[0],
  #  MODEL = entry[1][0][1]), entry])

#for entry in selected_segmentation_counts:
 #   graph.append([env.InspectModel("work/graph_of_minstrel_texts_with_{}_resolution_{}_topics.png".format(entry[2],entry[1][1][0][0]),
  #  entry[0],MODEL = entry[1][1][0][1]), entry])

#for entry in no_minstrel_counts:
 #  graph.append([env.InspectModel("work/graph_of_no_minstrel_texts_with_{}_resolution_{}_topics.png".format(entry[2],entry[1][1][0][0]),
  # entry[0],MODEL = entry[1][1][0][1]), entry])

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


full_set_variance_bimodal = []
full_set_top_words_per_topic = []
for numpy_chart in full_set_numpy_arrays: 
    full_set_variance_bimodal.append(env.ConstructBimodalityVarianceChart("work/variance_bimodal_chart_{}_resolution_{}_topics.json".format(numpy_chart[1][1][2], numpy_chart[1][1][1]), numpy_chart[0], OUTPUT_CHART = "work/variance_bimodal_chart_{}_resolution_{}_topics.png".format(numpy_chart[1][1][2], numpy_chart[1][1][1]), MODEL = numpy_chart[1][1][0]))
    full_set_top_words_per_topic.append(env.AnalyzeTopWordsPerTopic("work/top_words_per_topic__{}_resolution_{}_topics.json".format(numpy_chart[1][1][2], numpy_chart[1][1][1]), numpy_chart[0], MODEL = numpy_chart[1][1][0]))
no_minstrel_variance_bimodal = []
no_minstrel_top_words_per_topic = []
for numpy_chart in no_minstrel_numpy_array:
    no_minstrel_variance_bimodal.append(env.ConstructBimodalityVarianceChart("work/no_minstrel_variance_bimodal_chart_{}_resolution_{}_topics.json".format(numpy_chart[2], numpy_chart[1][1][1][1]), numpy_chart[0],OUTPUT_CHART = "work/no_minstrel_variance_bimodal_chart_{}_resolution_{}_topics.png".format(numpy_chart[2], numpy_chart[1][1][1][1]), MODEL =numpy_chart[1][1][1][0]))
    no_minstrel_top_words_per_topic.append(env.AnalyzeTopWordsPerTopic("work/no_minstrel_top_words_per_topic__{}_resolution_{}_topics.json".format(numpy_chart[2], numpy_chart[1][1][1][1]), numpy_chart[0], MODEL =numpy_chart[1][1][1][0]))
minstrel_variance_bimodal = []
minstrel_top_words_per_topic = []
for numpy_chart in minstrel_variance_bimodal:
    minstrel_variance_bimodal.append(env.ConstructBimodalityVarianceChart("work/minstrel_variance_bimodal_chart_{}_resolution_{}_topics.json".format(numpy_chart[2], numpy_chart[1][1][1][1]), numpy_chart[0],OUTPUT_CHART = "work/minstrel_variance_bimodal_chart_{}_resolution_{}_topics.png".format(numpy_chart[2], numpy_chart[1][1][1][1]), MODEL = numpy_chart[1][1][1][0]))
    minstrel_top_words_per_topic.append(env.AnalyzeTopWordsPerTopic("work/minstrel_top_words_per_topic__{}_resolution_{}_topics.json".format(numpy_chart[2], numpy_chart[1][1][1][1]), numpy_chart[0], MODEL =numpy_chart[1][1][1][0]))












