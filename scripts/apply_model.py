from itertools import islice
import csv
import re
import logging
import random
import pickle
import json
import argparse
import gensim 
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim.parsing.preprocessing as gpp


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    dest="model",
    required=True,
    help="Model file generated by training script.")

parser.add_argument(
    "--topic_number",
    dest="topic",
    required=True,
    help="number of topics"
)

parser.add_argument(
    "--data",
    dest="data",
    required=True,
    help="Data file model will be applied to."
)

parser.add_argument(
    "--json_out",
    dest="json_out",
    required=True,
    help="Counts output file."
)

parser.add_argument(
    "--minimum_word_length",
    dest="minimum_word_length",
    default=3,
    type=int,
    help="Minimum word length"
)
args = parser.parse_args()

# A silly default setting in the csv library needs to be changed 
# to handle larger fields.
csv.field_size_limit(1000000000)
no_topics = args.topic
# For each "group", we'll collect the number of times each topic
# occurs.
groupwise_topic_counts = {}
data_dictionary_list = []
group_names = {}
output_json_l = []
# Read in the model that was previously trained and serialized.
with open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())


with open(args.data, "rt") as ifd:

    for line in ifd:
        dictionary = json.loads(line)
        data_dictionary_list.append(dictionary)



for row in  data_dictionary_list:
    new_json_row = {}

        
    # We want to prepare the data the same way we prepared the data that
    # trained the model (there may be situations where we'd do something
    # different, but only with a particularly good reason!).
    tokens = gpp.split_on_space(
            gpp.strip_short(
                gpp.remove_stopwords(
                    gpp.strip_non_alphanum(
                        row["full_text"].lower()
                    ),
                ),
                minsize=args.minimum_word_length
            )
    )
                  

    # Turn the subdocument tokens into integers and count them, using the
    # trained model (so it employs the same mapping as it was trained with).
          
           
    subdocument_bow = model.id2word.doc2bow(tokens)
        
    # It will be useful to have the "bag-of-words" counts as a dictionary, too.
    subdocument_bow_lookup = dict(subdocument_bow)
        
    
        
    doc_topic, test , labeled_subdocument = model.get_document_topics(
        subdocument_bow,
        per_word_topics=True,minimum_phi_value = 0.0 
        )
    labeled_word_list = []
    topic_list = []
    for word in labeled_subdocument:
        #       print("this is the phi output per word")
        #      print(word)
        english_word = model.id2word.get(word[0])
        #labeled_word_list.append([english_word,word])
        word_id = word[0]
        non_float_topics_list = []
                
            
        for topic_prob in word[1]:
            new = topic_prob[0]
            no_float = (float(topic_prob[1]))
            new_t = [new,no_float]
            non_float_topics_list.append(new_t)
        labeled_word_list.append([english_word,word_id,non_float_topics_list])
    for topic in doc_topic:
        new_t = topic[:1] + (float(topic[1]),)
        topic_list.append(new_t)

    row["topics_for_word_phi"] = labeled_word_list
    row["document_topics"] = topic_list
    row["topics_for_word"] = test
    row["sub_doc_bow_dict"] = subdocument_bow_lookup
    output_json_l.append(row)
            

with open(args.json_out, "wt") as out_thing:
  for line in output_json_l:
      out_thing.write(json.dumps(line) + "\n")













  






    








