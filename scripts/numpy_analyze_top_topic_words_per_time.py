import numpy
import argparse
import json
import pickle
import gensim
import gensim
import csv
import gzip
import torch
from detm import DETM



parser = argparse.ArgumentParser()
parser.add_argument(
    "--matrix",
    dest="matrix",
    required=True,
    help="numpy matrix")

parser.add_argument(
    "--output_file",
    dest="output_file",
    help="where does the file go?"
)

parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="model used for creating dataset"
)
parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')  
args = parser.parse_args()

matrix = numpy.load(args.matrix)
args.device = "cpu" 
with gzip.open(args.model, "rb") as ifd:
    model = torch.load(ifd, map_location=torch.device(args.device))


    
output_dictionary = {}
    
for topic in range(matrix.shape[0]):
    output_dictionary[topic] = {}
    for time_slice in range(matrix.shape[2]):
        word_distribution = matrix[topic, :, time_slice]
        indices = numpy.argpartition(word_distribution, -10)[-10:]
        indices_sorted = indices[numpy.argsort(word_distribution[indices])[::-1]]
        output_list = []

        for element in indices_sorted:
            # Convert numpy.int64 to Python's int before appending
            element_index = int(element)  # Convert index to native Python int
            
            word = model.id2token[element_index]  # Assuming model.id2word[element] is already a Python native type
            output_list.append([element_index, word])
        
        output_dictionary[topic][time_slice] = output_list

print(output_dictionary)

with open(args.output_file, "w") as output_file:
    json.dump(output_dictionary, output_file, indent = 4)
