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
            action="python scripts/apply_model.py --model ${SOURCES[0]} --data ${DATA} --group_resolution ${GROUP_RESOLUTION}  --counts ${TARGETS[0]} --topic_num ${NUMBER_OF_TOPICS} --json_out ${TARGETS[1]} --minstrel_counts ${TARGETS[2]}"
        ),       
	"InspectModel" : Builder (action = "python scripts/inspect_model.py --counts ${SOURCES[0]} --figure ${TARGETS[0]} --model ${MODEL}"),
	"CalculateVariance" : Builder (action = "python scripts/calculate_variance.py --counts ${SOURCES[0]} --output ${TARGETS[0]}")
}
)



# OK, at this point we have defined all the builders and variables, so it's
# time to specify the actual experimental process, which will involve
# running all combinations of datasets, folds, model types, and parameter values,
# collecting the build artifacts from applying the models to test data in a list.
#
# The basic pattern for invoking a build rule is:
#
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"
#
# Note how variables are specified in each invocation, and their values used to fill
# in the build commands *and* determine output filenames.  It's a very flexible system,
# and there are ways to make it less verbose, but in this case explicit is better than
# implicit.
#
# Note also how the outputs ("targets") from earlier invocation are used as the inputs
# ("sources") to later ones, and how some outputs are also gathered into the "results"
# variable, so they can be summarized together after each experiment runs.

topic_model_list = []
json_metadata_including_text_ocr = env.PerformOcr(["work/json_metadata.jsonl"],env["DATA_LOCATION"])
for number in env["NUMBERS_OF_TOPICS"]:
    topic_model_list.append([number, (env.TrainModel("work/model_with_{}_topics.bin".format(number),json_metadata_including_text_ocr, NUMBER_OF_TOPICS = number))])
results = []

# model format model [0] = number of topics, model[1] = trained model 

for model in topic_model_list:
    for resolution in env["GROUP_RESOLUTIONS"]:
    	results.append([resolution, model, (env.ApplyModel(["work/counts_with_{}_resolution_{}_topics".format(resolution, model[0]),"work/levy_json_{}_topics_per_word{}.jsonl".format(model[0],resolution),"work/minstrel_counts_with_{}_resolution_{}_topics".format(resolution, model[0])], model[1], GROUP_RESOLUTION = resolution, DATA = json_metadata_including_text_ocr, NUMBER_OF_TOPICS = model[0]))])
output = []
#results format [resolution, [number, model],apply model]
for result in results:
    output.append([results,env.InspectModel("work/graph_with_{}_resolution_{}_topics.png".format(result[0],result[1][0]), "work/counts_with_{}_resolution_{}_topics".format(result[0], result[1][0]), MODEL = result[1][1])])
    output.append(env.InspectModel("work/graph_of_minstrel_texts_with_{}_resolution_{}_topics.png".format(result[0],result[1][0]), "work/minstrel_counts_with_{}_resolution_{}_topics".format(result[0], result[1][0]), MODEL = result[1][1]))

variance = []


for put in output:
    variance.append(env.CalculateVariance("work/variance_with_{}_resolution_{}_topics.json".format(put[0][0], put[0][1][0]), "work/minstrel_counts_with_{}_reso\
lution_{}_topics".format(put[0][0], put[0][1][0])))
    


# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).
#report = env.GenerateReport(
#    "work/report.txt",
 #   results)
