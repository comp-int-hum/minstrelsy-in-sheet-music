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
            action="python scripts/apply_model.py --model ${SOURCES[0]} --data ${DATA} --group_resolution ${GROUP_RESOLUTION}  --counts ${TARGETS[0]} --topic_num ${NUMBER_OF_TOPICS}"
        )       
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
for model in topic_model_list:
    for resolution in env["GROUP_RESOLUTIONS"]:
    	results.append(env.ApplyModel("work/counts_with_{}_resolution_{}".format(resolution, model[0]), model[1], GROUP_RESOLUTION = resolution, DATA = json_metadata_including_text_ocr, NUMBER_OF_TOPICS = model[0]))
	

#for dataset_name in env["DATASETS"]:
 #   data = env.CreateData("work/${DATASET_NAME}/data.txt", [], DATASET_NAME=dataset_name)
  #  for fold in range(1, env["FOLDS"] + 1):
   #     train, dev, test = env.ShuffleData(
    #        [
     #           "work/${DATASET_NAME}/${FOLD}/train.txt",
      #          "work/${DATASET_NAME}/${FOLD}/dev.txt",
       #         "work/${DATASET_NAME}/${FOLD}/test.txt",
        #    ],
         #   data,
          #  FOLD=fold,
           # DATASET_NAME=dataset_name,
       # )
       # for model_type in env["MODEL_TYPES"]:
        #    for parameter_value in env["PARAMETER_VALUES"]:
         #       model = env.TrainModel(
          #          "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/model.bin",
           #         [train, dev],
            #        FOLD=fold,
             #       DATASET_NAME=dataset_name,
              #      MODEL_TYPE=model_type,
               #     PARAMETER_VALUE=parameter_value,
               # )
               # results.append(
                #    env.ApplyModel(
                 #       "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/applied.txt",
                  #      [model, test],
                   #     FOLD=fold,
                    #    DATASET_NAME=dataset_name,
                     #   MODEL_TYPE=model_type,
                      #  PARAMETER_VALUE=parameter_value,                        
                   # )
               # )

# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).
#report = env.GenerateReport(
#    "work/report.txt",
 #   results
#)
