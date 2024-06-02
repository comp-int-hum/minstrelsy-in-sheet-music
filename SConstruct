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

vars = Variables("custom.py")
vars.AddVariables( 
    ("NUMBERS_OF_TOPICS", "", [100]),
    ("DATA_PATH", "", os.path.expanduser("~/corpora")),
    ("SHEET_MUSIC_ARCHIVE", "", "${DATA_PATH}/levy.zip"),
    ("WINDOW_SIZES", "", [25]),
    ("MAX_SUBDOC_LENGTHS", "", [50]),    
    ("EXISTING_JSON", "", False),
    ("NUM_ID_SPLITS", "", 500),
    ("LIMIT_SPLITS", "", None),
    ("GPU_ACCOUNT", "", None),
    ("GPU_QUEUE", "", None),
    ("PRECOMPUTED_OCR", "", False),
    ("MIN_YEAR", "", 1800),
    ("MAX_YEAR", "", 1920),
    ("EPOCHS", "", 1000),
    ("LEARNING_RATE", "", .015),
    ("RANDOM_SEED", "", 1),
    ("NUM_TOP_WORDS", "", 8),
    ("MIN_WORD_OCCURRENCE", "", 50),
)    

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    BUILDERS={
        "Scatter" : Builder(
            action="python scripts/scatter.py --input ${SOURCES[0]} ${TARGETS}"
        ),
        "Gather" : Builder(
            action="python scripts/gather.py --output ${TARGETS[0]} ${SOURCES}"
        ),        
        "PerformOcr" : Builder(
            action="python scripts/perform_ocr.py --input_archive ${SOURCES[0]} --id_file ${SOURCES[1]} --output ${TARGETS[0]}"
        ),
        "CleanText" : Builder(
            action="python scripts/clean_text.py --input ${SOURCES[0]} --output ${TARGETS[0]} --min_year ${MIN_YEAR} --max_year ${MAX_YEAR}"
        ),
	"TrainEmbeddings" : Builder(
	    action = "python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ), 
	"TrainDETM" : Builder(
            action="python scripts/train_detm.py --embeddings ${SOURCES[0]} --train ${SOURCES[1]}  --output ${TARGETS[0]} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE} --min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROPORTION} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --epochs ${EPOCHS} --learning_rate ${LEARNING_RATE} --random_seed ${RANDOM_SEED}"
        ),
        "ApplyDETM" : Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --max_subdoc_length ${MAX_SUBDOC_LENGTH}"
        ),
        "GenerateWordSimilarityTable" : Builder(
            action="python scripts/generate_word_similarity_table.py --embeddings ${SOURCES[0]} --output ${TARGETS[0]} --target_words ${WORD_SIMILARITY_TARGETS} --top_neighbors ${TOP_NEIGHBORS}"
        ),
        "CreateMatrices" : Builder(
            action="python scripts/create_matrices.py --topic_annotations ${SOURCES[0]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE}"
        ),
        "CreateFigures" : Builder(
            action="python scripts/create_figures.py --input ${SOURCES[0]} --latex ${TARGETS[0]} --temporal_image ${TARGETS[1]} --num_top_words ${NUM_TOP_WORDS}"
        )
    }
)

if env["PRECOMPUTED_OCR"]:
    all_ocr = env.File("work/all_ocr.jsonl.gz")
else:
    id_splits = env.Scatter(
        ["work/id_splits/{}.txt.gz".format(i + 1) for i in range(env["NUM_ID_SPLITS"])],
        env["SHEET_MUSIC_ARCHIVE"]
    )

    ocr_outputs = []
    for i, split in enumerate(id_splits[0:env["LIMIT_SPLITS"]] if env.get("LIMIT_SPLITS") else id_splits):
        ocr_outputs.append(
            env.PerformOcr("work/ocr_splits/{}.jsonl.gz".format(i + 1), [env["SHEET_MUSIC_ARCHIVE"], split])
        )

    all_ocr = env.Gather(
        "work/all_ocr.jsonl.gz",
        ocr_outputs
    )

cleaner_ocr = env.CleanText(
    "work/cleaner_ocr.jsonl.gz",
    all_ocr    
)
    
embeddings = env.TrainEmbeddings(
    [
        "work/word2vec_embeddings.bin",
        "work/word2vec_embeddings.bin.syn1neg.npy",
        "work/word2vec_embeddings.bin.wv.vectors.npy"
    ],
    cleaner_ocr
)[0]

env.GenerateWordSimilarityTable(
    "work/word_similarity.tex",
    embeddings,
    WORD_SIMILARITY_TARGETS=["mother", "war", "south", "love", "toil", "dem"],
    TOP_NEIGHBORS=5
)


topic_models = {}
for number_of_topics in env["NUMBERS_OF_TOPICS"]:
    for window_size in env["WINDOW_SIZES"]:
        for max_subdoc_length in env["MAX_SUBDOC_LENGTHS"]:
            model = env.TrainDETM(
                "work/detm_model_${NUMBER_OF_TOPICS}_${MAX_SUBDOC_LENGTH}_${WINDOW_SIZE}.bin",
                [embeddings, cleaner_ocr],
                NUMBER_OF_TOPICS=number_of_topics,
                BATCH_SIZE=2000,
                MAX_SUBDOC_LENGTH=max_subdoc_length,
                WINDOW_SIZE=window_size,
                MAX_WORD_PROPORTION=0.7,
                RANDOM_SEED=env["RANDOM_SEED"],
                STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                STEAMROLLER_GPU_COUNT=1,
                STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                STEAMROLLER_MEMORY="64G"
            )

            labeled = env.ApplyDETM(
                "work/results_${NUMBER_OF_TOPICS}_${MAX_SUBDOC_LENGTH}_${WINDOW_SIZE}.jsonl.gz",
                [model, cleaner_ocr],
                NUMBER_OF_TOPICS=number_of_topics,
                MAX_SUBDOC_LENGTH=max_subdoc_length,
                WINDOW_SIZE=window_size,                
                STEAMROLLER_MEMORY="64G"        
            )

            matrices = env.CreateMatrices(
                "work/matrices_${NUMBER_OF_TOPICS}_${MAX_SUBDOC_LENGTH}_${WINDOW_SIZE}.pkl.gz",
                labeled,
                NUMBER_OF_TOPICS=number_of_topics,
                MAX_SUBDOC_LENGTH=max_subdoc_length,
                WINDOW_SIZE=window_size,                
                STEAMROLLER_MEMORY="64G"        
            )
    
            figures = env.CreateFigures(
                [
                    "work/tables_${NUMBER_OF_TOPICS}_${MAX_SUBDOC_LENGTH}_${WINDOW_SIZE}.tex",
                    "work/temporal_image_${NUMBER_OF_TOPICS}_${MAX_SUBDOC_LENGTH}_${WINDOW_SIZE}.png",
                ],
                matrices,
                NUMBER_OF_TOPICS=number_of_topics,
                MAX_SUBDOC_LENGTH=max_subdoc_length,
                WINDOW_SIZE=window_size
            )
            
