import json
import numpy
import argparse
import pickle
import gzip
import torch
from detm import DETM
from gensim.models import Word2Vec
import logging

logger = logging.getLogger("construct numpy array ")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    dest="input_file",
    required=True,
    help="json database")

parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="topic model used"
)

parser.add_argument(
    "--number",
    dest="number",
    required=True,
    type = int, 
    help="how many topics"
)


parser.add_argument(
    "--output_file",
    dest="output_file",
    required=True,
    help="where does the file go?"
)

parser.add_argument(
    "--year_range",
    dest="year_range",
    type = int,
    default = 130, 
    help="how many years, for bucket division"
)

parser.add_argument(
    "--year_bucket",
    dest="year_bucket",
    required=True,
    type = int, 
    help="how are you organizing years?"
)

parser.add_argument(
    "--date_cutoff",
    dest="cutoff_date",
    default=1930,
    type=int,
    help="final date counted for groups"
)


parser.add_argument(
    "--start_year",
    dest="start_year",
    default = 1800, 
    type = int,
    help="initial year"
)


parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')

logger.warning("Setting device to CPU because CUDA isn't available")



args = parser.parse_args()
args.device = "cpu"
database = {}
year = {}

word_dictionary = {} 
counter = 0 

# Read in the model that was previously trained and serialized.                                                                            
#with open(args.model, "rb") as ifd:
 #   model = pickle.loads(ifd.read())
#print(len(model.id2word))
#word_len= len(model.id2word)




#note -- this is a BAD FIX -- you should probably change the initial number 

year_range = 130

buckets = (args.year_range // args.year_bucket) + (1 if args.year_range  % args.year_bucket != 0 else 0)
print(buckets)

#with open(args.input_file, "r") as in_file:
 #   for x in in_file:

  #      local_comp = json.loads(x)
   #     for entry in local_comp["text"]:
    #        word = entry[0]
     #       check = word_dictionary.get(word,counter)
      #      if check == counter:
       #         counter = counter + 1

#print(len(word_dictionary))

with gzip.open(args.model, "rb") as ifd:
    model = torch.load(ifd, map_location=torch.device(args.device))

model.id2token
#print(model.id2token)
token2id = {v : k for k, v in model.id2token.items()}
#print(token2id)
word_len = len(model.id2token)
print(word_len)
topics = args.number
overall_array= numpy.zeros((topics,word_len,buckets))

counter = 0

with open(args.input_file, "r") as in_file:
    #counter = 0
    local_list = []
    if args.input_file.endswith(".jsonl"):
        for x in in_file:
     #       counter = counter + 1
            composition =  json.loads(x)
            local_list.append(composition)
            #someday find a way to include the guesses? 
       #composition = composition[0]
       #print(type(composition))
       #print(composition["time"])
       #if type(composition) == list:
           #print(composition)
           #print(composition[0])
           #print(composition[1])
         #  composition = composition[0]
       #if type(composition) == list:                                                                                                                                                                            
           #print(composition)
        #   print(composition[0])
           #print(composition[1])                                                                                                                                                                                
    else:
        compositions = json.load(in_file)

        for x in compositions[0]:
            print(type(x))
            print(x)
            #if type(x) == list:
             #   y = x[0]
            local_list.append(x)
        #print(local_list)
        #second_counter = 0 
        #for thing in local_list:
         #   print(thing)
          #  print(second_counter)
           # second_counter = second_counter + 1 
    for song in local_list:    

        if song["time"] < args.cutoff_date and song["time"] > args.start_year:
            counter = counter + 1       
            pub_date_bucket = (song["time"] - args.start_year) // args.year_bucket  


            #this dealt with a "final bucket" issue, not sure if that will still be at play with the detm
            #if composition["date"]) >= 1925:
            #   pub_date_bucket = 4
            print("this is the date bucket")
            print(pub_date_bucket)
            for word in song["text"]:
                if word[1] != None: 
           
                    word_num = token2id[word[0]]
           
            
                    top_topic = word[1]
               
                    print("this is the number of buckets")
                     
                    print(buckets)
                    print("this is the publication date")
                    print(song["time"])
                    overall_array[top_topic,word_num, pub_date_bucket] = overall_array[top_topic,word_num, pub_date_bucket] + 1
                    print(overall_array[ top_topic,word_num, pub_date_bucket])
                    print(counter)
#numpy.savetxt("work/test_slice.csv", overall_array[:, :, 1], delimiter=",")
numpy.save(args.output_file, overall_array)
    
                  
                 
